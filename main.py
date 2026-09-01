import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

# SEARCH FIELDS
SEARCH_FIELDS = [
    "name", "firstName", "lastName", "middleName", "fullName",
    "fathersName", "motherName", "spouseName", "guardianName",
    "gender", "dob", "age", "bloodGroup", "nationality", "religion", "caste",
    "phoneNumber", "alternatePhone", "otherNumber", "emergencyContact",
    "email", "alternateEmail",
    "address", "permanentAddress", "currentAddress", "officeAddress",
    "city", "town", "district", "state", "pincode", "country",
    "aadharNumber", "panNumber", "voterId", "passportNumber", 
    "drivingLicense", "rationCard", "ssn", "taxId", "employeeId",
    "occupation", "profession", "designation", "companyName", "officeName",
    "education", "qualification", "instituteName", "university", "graduationYear",
    "bankAccount", "ifscCode", "accountNumber", "upiId", "creditCardNumber",
    "familyMembers", "childrenCount", "maritalStatus",
    "locality", "landmark", "area", "sector", "colony", "village", "tehsil",
    "mandal", "zilla", "parish", "municipality", "ward",
    "source", "createdDate", "updatedDate", "status", "activeFlag",
    "relationship", "category", "subCategory", "type", "notes", "comments"
]

NUMBER_FIELDS = [
    "phoneNumber", "aadharNumber", "otherNumber", "alternatePhone",
    "emergencyContact", "panNumber", "voterId", "passportNumber",
    "drivingLicense", "bankAccount", "accountNumber", "creditCardNumber",
    "upiId", "ssn", "taxId", "employeeId", "rationCard", "emergencyContactNumber"
]

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    name = (row.get("name") or row.get("fullName") or row.get("firstName") or "").strip()
    fname = (row.get("fathersName") or row.get("fatherName") or "").strip()
    
    if ph or ad:
        return (ph, ad)
    if name or fname:
        return (name, fname)
    return (str(row.get("email", "")).strip(), "")


def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


def _get_index_for_field(field: str):
    mapping = {
        "phoneNumber": "phone",
        "alternatePhone": "phone",
        "otherNumber": "phone",
        "emergencyContact": "phone",
        "aadharNumber": "aadhar",
    }
    return mapping.get(field)


def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    
    v = value.replace("'", "''")
    index_kind = _get_index_for_field(field)
    view_name = None
    
    if mode == "exact":
        if index_kind and _idx_ready(index_kind):
            view_name = f"people_{index_kind}"
        else:
            for kind in REMOTE_INDEXES:
                if _idx_ready(kind):
                    view_name = f"people_{kind}"
                    break
        
        if view_name:
            sql = f"SELECT * FROM {view_name} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
    
    elif mode == "contains":
        for kind in REMOTE_INDEXES:
            if _idx_ready(kind):
                view_name = f"people_{kind}"
                break
        
        if view_name:
            v2 = v.replace("%", r"\%").replace("_", r"\_")
            sql = f"SELECT * FROM {view_name} WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
    else:
        raise ValueError(f"Unknown mode: {mode}")

    try:
        con = _get_conn()
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
        return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}
    except Exception as e:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": [], "error": str(e)}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}
    
    is_num = q.isdigit() and len(q) >= 8
    is_email = '@' in q and '.' in q
    
    all_rows = []
    searched = []
    
    if is_num:
        for field in ["phoneNumber", "alternatePhone", "otherNumber", "emergencyContact"]:
            if field in SEARCH_FIELDS:
                r = _run_field_search(field, q, "exact", limit)
                if r.get("results"):
                    all_rows.extend(r["results"])
                    searched.append(field)
                    break
        
        if not all_rows and "aadharNumber" in SEARCH_FIELDS:
            r = _run_field_search("aadharNumber", q, "exact", limit)
            if r.get("results"):
                all_rows.extend(r["results"])
                searched.append("aadharNumber")
    
    elif is_email:
        for field in ["email", "alternateEmail"]:
            if field in SEARCH_FIELDS:
                r = _run_field_search(field, q, "exact", limit)
                if r.get("results"):
                    all_rows.extend(r["results"])
                    searched.append(field)
                    break
    
    else:
        for field in ["name", "fullName", "firstName", "lastName"]:
            if field in SEARCH_FIELDS:
                r = _run_field_search(field, q, "contains", limit)
                if r.get("results"):
                    all_rows.extend(r["results"])
                    searched.append(field)
                    break
        
        if not all_rows:
            for field in ["address", "district", "city", "state", "town", "locality"]:
                if field in SEARCH_FIELDS:
                    r = _run_field_search(field, q, "contains", limit)
                    if r.get("results"):
                        all_rows.extend(r["results"])
                        searched.append(field)
                        break
    
    all_rows = _cap_duplicates(all_rows)[:limit]
    return {
        "query": q, 
        "searched_fields": searched,
        "count": len(all_rows), 
        "results": all_rows,
        "query_type": "number" if is_num else "email" if is_email else "text"
    }


# ── FastAPI ──────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API - Extended")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API (Extended)",
        "records": "2.5 Billion+",
        "fields_supported": len(SEARCH_FIELDS),
        "fields": SEARCH_FIELDS,
        "number_fields": NUMBER_FIELDS,
        "indexes": {k: _idx_ready(k) for k in REMOTE_INDEXES},
        "index_source": INDEX_SOURCE,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "fields_available": len(SEARCH_FIELDS),
        "indexes": {k: _idx_ready(k) for k in REMOTE_INDEXES},
        "index_source": INDEX_SOURCE
    }


@fastapi_app.get("/fields")
def get_fields():
    return {
        "total_fields": len(SEARCH_FIELDS),
        "search_fields": SEARCH_FIELDS,
        "number_fields": NUMBER_FIELDS,
        "indexed_fields": list(REMOTE_INDEXES.keys())
    }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile parameter")
    
    loop = asyncio.get_running_loop()
    
    if field:
        if field not in SEARCH_FIELDS:
            raise HTTPException(400, f"Field '{field}' not supported")
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    
    result = {
        "success": bool(data.get("count", 0)),
        **data,
        "number": q_val,
        "total": data.get("count", 0)
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _run_field_search,
                             item.get("field", "phoneNumber"),
                             item.get("value", ""),
                             item.get("mode", "exact"),
                             int(item.get("limit", req.limit)))
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({
            "searches": len(req.queries),
            "results": list(results)
        }, indent=2, ensure_ascii=False),
        media_type="application/json"
    )


# ── Pinger ──────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"[Pinger] OK")
            except Exception as e:
                print(f"[Pinger] Error: {e}")


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    
    personal_fields = ["name", "fullName", "firstName", "lastName", "fathersName", 
                      "motherName", "spouseName", "gender", "dob", "age", "bloodGroup"]
    personal = [f"**{f}:** {row[f]}" for f in personal_fields if row.get(f)]
    if personal:
        lines.append("### Personal Information")
        lines.extend(personal)
        lines.append("")
    
    contact_fields = ["phoneNumber", "alternatePhone", "otherNumber", "emergencyContact",
                     "email", "alternateEmail", "address", "city", "district", "state", "pincode"]
    contact = [f"**{f}:** {row[f]}" for f in contact_fields if row.get(f)]
    if contact:
        lines.append("### Contact Information")
        lines.extend(contact)
        lines.append("")
    
    id_fields = ["aadharNumber", "panNumber", "voterId", "passportNumber", 
                "drivingLicense", "rationCard", "ssn"]
    ids = [f"**{f}:** {row[f]}" for f in id_fields if row.get(f)]
    if ids:
        lines.append("### ID Documents")
        lines.extend(ids)
        lines.append("")
    
    prof_fields = ["occupation", "profession", "designation", "companyName", 
                  "education", "qualification", "instituteName"]
    prof = [f"**{f}:** {row[f]}" for f in prof_fields if row.get(f)]
    if prof:
        lines.append("### Professional")
        lines.extend(prof)
        lines.append("")
    
    fin_fields = ["bankAccount", "ifscCode", "accountNumber", "upiId"]
    fin = [f"**{f}:** {row[f]}" for f in fin_fields if row.get(f)]
    if fin:
        lines.append("### Financial")
        lines.extend(fin)
        lines.append("")
    
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    
    all_fields = set(SEARCH_FIELDS)
    displayed = set(personal_fields + contact_fields + id_fields + prof_fields + fin_fields)
    remaining = [f"**{f}:** {row[f]}" for f in all_fields if row.get(f) and f not in displayed]
    if remaining:
        lines.append("### Additional Information")
        lines.extend(remaining)
    
    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "Please enter a search query"
    
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"Error: {str(e)}"
    
    count = data.get("count", 0)
    results = data.get("results", [])
    searched = ", ".join(data.get("searched_fields", []))
    query_type = data.get("query_type", "unknown")
    
    if not results:
        return f"Query: {q}\nType: {query_type}\nSearched: {searched or 'None'}\n\nNo data found."
    
    header = f"Query: {q}\nType: {query_type}\nFound: {count} results\nSearched in: {searched}\nAvailable fields: {len(SEARCH_FIELDS)}\n"
    
    parts = [header]
    for i, row in enumerate(results, 1):
        parts.append(f"---\nResult {i}\n{format_result(row)}")
    
    return "\n\n".join(parts)


def build_ui():
    with gr.Blocks(
        title="ICMR Search API - Extended",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        .footer { text-align: center; color: #888; margin-top: 20px; }
        """
    ) as demo:
        gr.Markdown("# ICMR + HITEK Search API (Extended)", elem_classes="main-title")
        gr.Markdown(f"Search 2.5 Billion+ records with {len(SEARCH_FIELDS)} searchable fields", elem_classes="subtitle")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Phone, Aadhaar, Name, Email, PAN, Voter ID, or any other field...",
                    lines=2
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results"
                )
        
        search_btn = gr.Button("Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        
        search_btn.click(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output
        )
        query_input.submit(
            fn=search_ui,
            inputs=[query_input, limit_slider],
            outputs=output
        )
        
        gr.Markdown("---")
        
        with gr.Accordion("Available Search Fields", open=False):
            fields_html = "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px;'>"
            for field in SEARCH_FIELDS:
                fields_html += f"<span style='background: #f0f0f0; padding: 3px 8px; border-radius: 4px; font-size: 0.8em;'>{field}</span>"
            fields_html += "</div>"
            gr.Markdown(f"Total Fields: {len(SEARCH_FIELDS)}\n\n{fields_html}")
        
        with gr.Accordion("API Info", open=False):
            gr.Markdown("Endpoints:\n- GET /search?q=query\n- GET /search?field=field&q=value\n- GET /fields\n- GET /health\n- GET /docs")
        
        gr.Markdown(
            "---\n"
            "<div class='footer'>"
            "Developer: @kzr0x | Channel: @api_wallah | "
            f"{len(SEARCH_FIELDS)} Fields Supported"
            "</div>",
            elem_classes="footer"
        )
    
    return demo


# ── Mount Gradio on FastAPI ─────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
