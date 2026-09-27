"""
Stage-by-stage Laya question bench for the pipeline-confidence dataset.  Edit bench/questions.py.

    bin/python bench/bench.py [--split train|val] [--stage NAME] [--force] [--limit N] [--case ID]
                              [--preview STAGE.PARAM] [--check-enums] [--workers 8] [--no-cache]
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = {"train": "fine_tuning.jsonl", "val": "validation.jsonl"}


def _load_dotenv(path):
    """KEY=value lines from bench/.env (gitignored) -> os.environ, without overriding real env vars."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(HERE, ".env"))
API_KEY = os.environ.get("LAYA_API_KEY", "")
URL = os.environ.get("LAYA_URL", "https://laya-server.internal.generalmind.com/v1/system-one")
DATA_DIR = os.environ.get("CONFIDENCE_DATA", os.path.expanduser("~/Downloads/datasets/confidence"))
CONTEXT_LIMIT = 1024  # server truncates input at this many tokens
CACHE_DIR = os.environ.get("BENCH_CACHE_DIR", os.path.join(HERE, ".cache"))
RUNS_DIR = os.environ.get("BENCH_RUNS_DIR", os.path.join(HERE, "runs"))
HEADER_FIELDS = ["document_type", "document_number", "document_date", "currency", "supplier_id", "buyer_id",
                 "ship_to_id", "payment_terms_days", "freight"]


def load_questions():
    path = os.environ.get("QUESTIONS_FILE", os.path.join(HERE, "questions.py"))
    spec = importlib.util.spec_from_file_location("questions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_split(split):
    with open(os.path.join(DATA_DIR, SPLITS[split])) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- stage documents

LINE_SPLIT = re.compile(r"(?=\bLine \d+:)")
LINE_STMT = re.compile(r"^Line (\d+):\s*([^,;]+?),\s*(-?[\d.,]+)\s+(\w+)\s+at\s+([\d.,]+)\s+each")


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def parser_view(inp):
    """Per-line view of what the parser produced: the text statement and the table row."""
    p = inp["parser_output"]
    stmts = {}
    for b in p["text_blocks"]:
        for seg in LINE_SPLIT.split(b["text"]):
            m = re.match(r"Line (\d+):", seg)
            if not m:
                continue
            text = re.split(r"\n|\s;\s", seg.strip())[0].strip(" .;>")
            # prefer the base statement (description, quantity, rate) over notes that also start "Line NN:"
            if m.group(1) not in stmts or (LINE_STMT.match(text) and not LINE_STMT.match(stmts[m.group(1)])):
                stmts[m.group(1)] = text
    rows = {}
    for t in p.get("tables", []):
        for row in t["rows"]:
            rows.setdefault(str(row[0]), dict(zip([h.lower() for h in t["headers"]], row)))
    ids = sorted(set(stmts) | set(rows), key=lambda x: (len(x), x))
    return {"lines": [{"line_id": i, "text": stmts.get(i), "table": rows.get(i)} for i in ids]}


def parser_flags(inp):
    """Code-computed parser mistakes.  severity "error" = the parse contradicts itself (labels the parser
    stage wrong); "warning" = worth a look, never used as a label."""
    p = inp["parser_output"]
    flags = []
    all_text = "\n".join(b["text"] for b in p["text_blocks"])
    mentioned = set(LINE_REF.findall(all_text))
    for line in parser_view(inp)["lines"]:
        lid, text, row = line["line_id"], line["text"], line["table"]
        if row and lid not in mentioned:
            flags.append({"type": "table_line_not_in_text", "severity": "error", "line_id": lid,
                          "detail": f"table has line {lid}, text never mentions it"})
        if text and not row and p.get("tables"):
            flags.append({"type": "line_not_in_table", "severity": "warning", "line_id": lid,
                          "detail": f"text states line {lid}, table has no row for it"})
        m = LINE_STMT.match(text or "")
        if m and row:
            for name, tv, sv in (("description", row.get("description"), m.group(2)),
                                 ("quantity", row.get("quantity"), m.group(3)),
                                 ("rate", row.get("rate"), m.group(5))):
                if tv is None:
                    continue
                same = (str(tv).strip().lower() == sv.strip().lower()) if name == "description" \
                    else (_num(tv) is not None and _num(tv) == _num(sv))
                if not same:
                    flags.append({"type": "table_text_conflict", "severity": "error", "line_id": lid,
                                  "detail": f"{name}: table {tv!r} vs text {sv!r}"})
    spans = [b["span_id"] for b in p["text_blocks"]]
    order = p.get("reading_order") or []
    if order and set(order) != set(spans):
        flags.append({"type": "reading_order_gap", "severity": "error", "line_id": None,
                      "detail": f"reading order {sorted(set(order) ^ set(spans))} missing/extra"})
    texts = [b["text"].strip() for b in p["text_blocks"]]
    if len(set(texts)) != len(texts):
        flags.append({"type": "duplicate_block", "severity": "warning", "line_id": None,
                      "detail": "identical text block appears twice"})
    for w in p.get("warnings", []):
        flags.append({"type": "parser_warning", "severity": "warning", "line_id": None, "detail": str(w)})
    return flags


TAMPER_SUFFIX = "#tampered"


def tamper_parser(inp, case_id):
    """Deterministic synthetic parser mistake: one table cell changed so it contradicts the text.
    Used only by the parser stage (the dataset itself has no parser errors)."""
    out = json.loads(json.dumps(inp))
    stmts = {l["line_id"] for l in parser_view(inp)["lines"] if LINE_STMT.match(l["text"] or "")}
    rows = [r for t in out["parser_output"].get("tables", []) for r in t["rows"] if str(r[0]) in stmts]
    if not rows:  # nothing verifiable to tamper with
        return out
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16)
    row = rows[h % len(rows)]
    kind = (h // 7) % 3
    if kind == 0:  # rate
        row[3] = f"{(_num(row[3]) or 1) * 1.5 + 1:.2f}"
    elif kind == 1:  # quantity
        row[2] = str(int((_num(row[2]) or 1) + 4))
    else:  # description
        row[1] = "stainless " + row[1] if not row[1].startswith("stainless") else "brass " + row[1][10:]
    return out


def tampered_records(cases, fraction):
    """Tampered copies of a deterministic `fraction` of cases, ids suffixed with #tampered."""
    out = []
    for r in cases:
        if int(hashlib.sha256(("t" + r["case_id"]).encode()).hexdigest(), 16) % 1000 < fraction * 1000:
            t = tamper_parser(r["input"], r["case_id"])
            if t != r["input"]:
                out.append({"case_id": r["case_id"] + TAMPER_SUFFIX, "input": t, "target": r["target"],
                            "synthetic": True})
    return out


def get_record(split, case_id):
    base = case_id[: -len(TAMPER_SUFFIX)] if case_id.endswith(TAMPER_SUFFIX) else case_id
    rec = next(r for r in load_split(split) if r["case_id"] == base)
    if base != case_id:
        return {"case_id": case_id, "input": tamper_parser(rec["input"], base), "target": rec["target"],
                "synthetic": True}
    return rec


def stage_document(kind, inp):
    if kind == "parser":
        return parser_view(inp)
    if kind == "extractor":
        return inp["extractor_output"]["document"]
    if kind == "normalization":
        return inp["normalization_output"]["document"]
    if kind == "final":
        return inp["final_output"]
    if kind == "matching":
        m = inp["matching_output"]
        base = inp["normalization_output"]["document"]["lines"]
        return {
            "supplier_id": m.get("selected_supplier_id"),
            "buyer_id": m.get("selected_buyer_id"),
            "lines": [{"line_id": l["line_id"], "product_id": pid}
                      for l, pid in zip(base, m.get("selected_product_ids", []))],
        }
    raise ValueError(f"unknown stage document '{kind}'")


def field_value(doc, field, line=None):
    return (line or doc).get(field)


def norm_value(v):
    return None if v is None else str(v)


# --------------------------------------------------------------------------- evidence slices

LINE_REF = re.compile(r"\blines?\s+(\d+)", re.I)


def _parser_blocks(inp):
    blocks = {b["span_id"]: b["text"] for b in inp["parser_output"]["text_blocks"]}
    order = inp["parser_output"].get("reading_order") or list(blocks)
    return [blocks[s] for s in order if s in blocks]


def _tables_text(inp, line_id=None):
    out = []
    for t in inp["parser_output"].get("tables", []):
        rows = [r for r in t["rows"] if line_id is None or (r and str(r[0]) == line_id)]
        if rows:
            out.append(" | ".join(t["headers"]))
            out += [" | ".join(map(str, r)) for r in rows]
    return "\n".join(out)


def _strip_line(line):
    return {k: v for k, v in line.items() if v not in ([], None)}


def _header(doc):
    return {k: doc[k] for k in HEADER_FIELDS if k in doc and k != "freight"}


def _totals(doc):
    return {k: doc[k] for k in ("subtotal", "tax", "freight", "total") if k in doc}


def _line_ids(doc):
    return [{k: x[k] for k in ("line_id", "product_id", "quantity") if k in x} for x in doc["lines"]]


def _stage_line(inp, stage, line_id):
    for line in inp[stage]["document"]["lines"]:
        if line["line_id"] == line_id:
            return _strip_line(line)
    return "line missing at this stage"


def _line_pairs(line):
    """This line's description / quantity / rate as the text states it next to the table cell."""
    if not line or not line.get("table"):
        return "no table row for this line (the table only lists some lines)"
    m = LINE_STMT.match(line.get("text") or "")
    if not m:
        return {"text": line.get("text") or "(not mentioned in the text)", "table": line["table"]}
    t = line["table"]

    def n(x):  # same number, same spelling: 27 / 27.00 / 1,234.5 -> "27.00" / "1234.50"
        v = _num(x)
        return x if v is None else f"{v:.2f}"
    return {"description": {"text": m.group(2), "table": t.get("description")},
            "quantity": {"text": n(m.group(3)), "table": n(t.get("quantity"))},
            "rate": {"text": n(m.group(5)), "table": n(t.get("rate"))}}


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def statement_value(inp, lid, field):
    """The value this line's own base statement gives for `field`, normalised like the stage documents."""
    view = next((l for l in parser_view(inp)["lines"] if l["line_id"] == lid), None)
    full = re.match(r"^Line (\d+):\s*([^,;]+?),\s*(-?[\d.,]+)\s+(\w+)\s+at\s+([\d.,]+)\s+each"
                    r"(?:,\s*VAT\s+(\d+)%)?(?:,\s*due\s+(\d{1,2})\s+(\w+)\s+(\d{4}))?", (view or {}).get("text") or "")
    if not full:
        return None
    if field == "quantity":
        return f"{_num(full.group(3)):g}"
    if field == "unit_price":
        return f"{_num(full.group(5)):.2f}"
    if field == "delivery_date" and full.group(7):
        return f"{full.group(9)}-{MONTHS.get(full.group(8).lower(), 0):02d}-{int(full.group(7)):02d}"
    if field == "schedule":
        return "[]"
    return None


def _same(field, a, b):
    if a is None or b is None:
        return False
    if field == "schedule":
        return (a in ([], "[]")) == (b in ([], "[]"))
    na, nb = _num(a), _num(b)
    return (na == nb) if (na is not None and nb is not None and field != "delivery_date") else str(a) == str(b)


def _stage_check(name, inp, line):
    """Code-side consistency rules for detection questions: what the stage value must look like when the
    situation the question detects applies.  (The model detects; code checks the value.)"""
    if not line:
        return False
    num = _num(line.get("quantity") or 0) or 0.0
    rules = {
        "quantity_negative": lambda: num < 0,
        "quantity_zero": lambda: num == 0,
        "discount_100": lambda: _num(line.get("discount_percent") or 0) == 100,
        "tax_zero": lambda: _num(line.get("tax_rate") or 0) == 0,
        "schedule_split": lambda: len(line.get("schedule") or []) > 1,
        "price_basis_not_1": lambda: (_num(line.get("price_basis") or 1) or 1) != 1,
        "quantity_changed": lambda: not _same("quantity", line.get("quantity"),
                                              statement_value(inp, line["line_id"], "quantity")),
        "unit_price_changed": lambda: not _same("unit_price", line.get("unit_price"),
                                                statement_value(inp, line["line_id"], "unit_price")),
        "delivery_date_changed": lambda: not _same("delivery_date", line.get("delivery_date"),
                                                   statement_value(inp, line["line_id"], "delivery_date")),
    }
    return rules[name]()


def _is_base_block(text):
    return len(re.findall(r"\bLine \d+:", text)) >= 2


def _line_source(inp, lid):
    """Only what concerns this line: its own statement, its table row, and notes that mention it."""
    view = next((l for l in parser_view(inp)["lines"] if l["line_id"] == lid), {})
    notes = [b for b in _parser_blocks(inp) if not _is_base_block(b) and lid in LINE_REF.findall(b)]
    out = {"line": lid, "statement": view.get("text") or "(no statement for this line)"}
    if view.get("table"):
        out["table row"] = {k: v for k, v in view["table"].items() if k != "line"}
    out["notes about this line"] = notes or "none"
    return out


def _header_source(inp):
    """Document-level facts: the base block without its line statements, plus notes that name no line."""
    head = []
    for b in _parser_blocks(inp):
        if _is_base_block(b):
            head.append(LINE_SPLIT.split(b)[0].strip())
    notes = [b for b in _parser_blocks(inp) if not _is_base_block(b) and not LINE_REF.findall(b)]
    return {"header": " ".join(head), "notes": notes or "none"}


def _pair_sentence(line, field):
    """One value, text vs table, as a sentence -- the model compares single pairs reliably, not several at once."""
    pairs = _line_pairs(line)
    if not isinstance(pairs, dict) or field not in pairs:
        return "The text statement for this line could not be matched to the table."
    return f"The text says {pairs[field]['text']}. The table says {pairs[field]['table']}."


def slice_value(name, inp, doc, line=None):
    fo, md = inp["final_output"], inp["master_data"]
    lid = line["line_id"] if line else None
    simple = {
        "doc_header": lambda: _header(doc),
        "doc_lines": lambda: [_strip_line(x) for x in doc["lines"]],
        "doc_line_ids": lambda: _line_ids(doc),
        "doc_totals": lambda: _totals(doc),
        "doc_line": lambda: _strip_line(line),
        "doc_parties": lambda: {k: doc.get(k) for k in ("supplier_id", "buyer_id")},
        "line_ref": lambda: {"line_id": lid},
        "parser_line": lambda: {"text says": (line or {}).get("text") or "(line not mentioned in the text)",
                                "table says": (line or {}).get("table")
                                or "(no table row -- the table only lists some lines, which is normal)"},
        "parser_line_pairs": lambda: _line_pairs(line),
        "line_source": lambda: _line_source(inp, lid),
        "line_discount_notes": lambda: [n for n in (_line_source(inp, lid)["notes about this line"]
                                                    if isinstance(_line_source(inp, lid)["notes about this line"], list) else [])
                                        if re.search(r"discount|allowance|promotion|free|rebate", n, re.I)] or "none",
        "header_source": lambda: _header_source(inp),
        "general_notes": lambda: _header_source(inp)["notes"],
        "parser_description_pair": lambda: _pair_sentence(line, "description"),
        "parser_quantity_pair": lambda: _pair_sentence(line, "quantity"),
        "parser_rate_pair": lambda: _pair_sentence(line, "rate"),
        "parser_table_pairs": lambda: {l["line_id"]: _line_pairs(l) for l in parser_view(inp)["lines"] if l["table"]},
        "parser_table_check": lambda: [
            {"line": l["line_id"], "text says": l["text"] or "(line not mentioned in the text)",
             "table says": {k: v for k, v in l["table"].items() if k != "line"}}
            for l in parser_view(inp)["lines"] if l["table"]],
        "parser_tables": lambda: inp["parser_output"].get("tables", []),
        "parser_warnings": lambda: inp["parser_output"].get("warnings", []),
        "parser_reading_order": lambda: inp["parser_output"].get("reading_order", []),
        "validation_checks": lambda: inp["validation_output"].get("checks", []),
        "final_header": lambda: _header(fo),
        "final_lines": lambda: [_strip_line(x) for x in fo["lines"]],
        "final_line_ids": lambda: _line_ids(fo),
        "final_totals": lambda: _totals(fo),
        "final_line": lambda: next((_strip_line(x) for x in fo["lines"] if x["line_id"] == lid), "not in final output"),
        "extractor_line": lambda: _stage_line(inp, "extractor_output", lid),
        "normalization_line": lambda: _stage_line(inp, "normalization_output", lid),
        "normalization_ops": lambda: inp["normalization_output"].get("applied_operations", []),
        "matching_line": lambda: next((pc["candidates"] for pc in inp["matching_output"].get("product_candidates", [])
                                       if pc["line_id"] == lid), []),
        "email": lambda: inp["source_context"]["email"],
        "master_parties": lambda: md["parties"],
        "master_locations": lambda: md["locations"],
        "master_products": lambda: md["products"],
        "master_products_brief": lambda: [{k: v for k, v in p.items()
                                           if k in ("id", "description", "revision", "customer_mappings")}
                                          for p in md["products"]],
        "master_product_line": lambda: next((p for p in md["products"] if p["id"] == line.get("product_id")),
                                            "not in master data"),
        "customer_profile": lambda: md.get("customer_profile", {}),
        "commercial_policy": lambda: md.get("commercial_policy", {}),
    }
    if name in simple:
        return simple[name]()
    if name == "parser_text":
        tables = _tables_text(inp)
        return "\n".join(_parser_blocks(inp)) + ("\nTABLE:\n" + tables if tables else "")
    if name == "parser_text_line":
        keep = [b for b in _parser_blocks(inp) if not LINE_REF.findall(b) or lid in LINE_REF.findall(b)]
        tables = _tables_text(inp, lid)
        return "\n".join(keep) + ("\nTABLE:\n" + tables if tables else "")
    raise ValueError(f"unknown evidence slice '{name}' (see questions.py docstring)")


# --------------------------------------------------------------------------- questions

def dynamic_criteria(source, inp):
    """Per-case options from master data: ids are unique, so options are exclusive and exhaustive."""
    md = inp["master_data"]
    if source == "parties":
        return {p["id"]: f"{p['name']} ({p['role']})" for p in md["parties"]}
    if source == "locations":
        return {l["id"]: f"{l['name']} ({l['role']})" for l in md["locations"]}
    if source == "products":
        out = {}
        for p in md["products"]:
            desc = p["description"]
            if p.get("revision"):
                desc += f", revision {p['revision']}"
            for cm in p.get("customer_mappings", []):
                desc += f", buyer code {cm['code']} for {cm['buyer_id']}"
            out[p["id"]] = desc
        return out
    raise ValueError(f"unknown criteria_from '{source}'")


def criteria_for(p, inp):
    if p.get("criteria_from"):
        return dynamic_criteria(p["criteria_from"], inp)
    return p.get("criteria")


def build_question(p, inp):
    q = {"type": p["type"], "instructions": p["instructions"]}
    if p["type"] == "choice":
        q["criteria"] = criteria_for(p, inp)
    return q


def build_payloads(pname, p, record, doc):
    """[(line_id or None, payload, stage value of answer_field or None)] for one param on one case."""
    inp = record["input"]
    targets = doc["lines"] if p["scope"] == "line" else [None]
    if p.get("only_table_lines"):  # parser: lines without a table row have nothing to compare
        targets = [l for l in targets if l.get("table")]
    out = []
    for line in targets:
        value = norm_value(field_value(doc, p["answer_field"], line)) if p.get("answer_field") else None
        if p.get("no_value") == "statement":
            f = p["answer_field"]
            raw = field_value(doc, f, line)
            value = (norm_value(raw), _same(f, raw, statement_value(inp, line["line_id"], f)))
        if p.get("stage_check"):
            value = _stage_check(p["stage_check"], inp, line)
        if "ask_unless_value" in p and value == p["ask_unless_value"]:
            continue  # e.g. the discount amount is only asked when the stage has a non-zero discount
        if p.get("per_note"):
            # one note per question; the line has the situation if ANY of its notes says so
            src = _line_source(inp, line["line_id"]) if line else {}
            notes = src.get("notes about this line") if isinstance(src.get("notes about this line"), list) else []
            if "general_notes" in p["evidence"]:
                g = _header_source(inp)["notes"]
                notes = notes + (g if isinstance(g, list) else [])
            payloads = [{"state": {"line": line["line_id"] if line else None, "note": n},
                         "questions": {pname: build_question(p, inp)}} for n in notes]
            out.append((line["line_id"] if line else None, payloads, value))
            continue
        state = {name: ({f: (line or doc).get(f) for f in p["fields"]} if name == "doc_fields"
                        else slice_value(name, inp, doc, line)) for name in p["evidence"]}
        out.append((line["line_id"] if line else None,
                    {"state": state, "questions": {pname: build_question(p, inp)}}, value))
    return out


# --------------------------------------------------------------------------- API

_SLOTS_DIR = os.path.join(CACHE_DIR, ".slots")
MAX_IN_FLIGHT = int(os.environ.get("LAYA_MAX_IN_FLIGHT", "24"))  # shared by every bench/probe process


def _acquire_slot():
    """Machine-wide cap on concurrent API requests (the server returns 504s when flooded)."""
    import fcntl
    os.makedirs(_SLOTS_DIR, exist_ok=True)
    while True:
        for i in range(MAX_IN_FLIGHT):
            f = open(os.path.join(_SLOTS_DIR, f"{i}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except OSError:
                f.close()
        time.sleep(0.05)


def call_api(payload, use_cache=True, cache_only=False):
    """-> (response, was_cached).  cache_only: never call the API; (None, False) when not cached."""
    blob = json.dumps(payload, sort_keys=True)
    key = hashlib.sha256((URL + blob).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".json")
    if (use_cache or cache_only) and os.path.exists(path):
        with open(path) as f:
            return json.load(f), True
    if cache_only:
        return None, False
    if not API_KEY:
        raise RuntimeError("LAYA_API_KEY is not set -- put it in bench/.env (see bench/.env.example)")
    req = urllib.request.Request(URL, data=blob.encode(), headers={"x-api-key": API_KEY, "content-type": "application/json"})
    attempts = 7
    for attempt in range(attempts):
        slot = _acquire_slot()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode())
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if code and 400 <= code < 500 and code != 429:
                body = e.read().decode() if hasattr(e, "read") else ""
                raise RuntimeError(f"HTTP {code}: {body}") from e
            if attempt == attempts - 1:
                raise
            wait = min(60, 2 ** attempt) + random.random()  # 504 / 429 / network: back off and retry
        finally:
            slot.close()
        time.sleep(wait)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(resp, f)
    os.replace(tmp, path)  # atomic: parallel processes never read a half-written answer
    return resp, False


def answer(payload, use_cache=True, cache_only=False):
    """call_api for one payload, or OR-combine a list of per-note noul payloads (empty list = no notes = NO)."""
    if not isinstance(payload, list):
        return call_api(payload, use_cache=use_cache, cache_only=cache_only)
    pname = None
    probs, cached, tokens = [], True, 0
    for pl in payload:
        pname = next(iter(pl["questions"]))
        r, c = call_api(pl, use_cache=use_cache, cache_only=cache_only)
        if r is None:
            return None, False
        probs.append(r["answers"][pname]["noul"])
        cached &= c
        tokens = max(tokens, r.get("usage", {}).get("input_tokens", 0))
    p_any = 1 - math.prod(1 - x for x in probs)
    return {"answers": {"*": {"noul": p_any, "per_note": [round(x, 4) for x in probs]}},
            "usage": {"input_tokens": tokens}}, cached


def read_answer(pname, p, resp, value):
    """-> (P(stage value/verdict is correct), model's pick or None)."""
    a = resp["answers"].get(pname) or resp["answers"]["*"]
    if p["type"] == "noul" and p.get("stage_check"):
        # detection question: YES = the situation applies.  The stage value is consistent when the code rule
        # holds exactly when the situation applies.
        p_yes = a["noul"]
        return (p_yes if value else 1 - p_yes), ("applies" if p_yes >= 0.5 else "does not apply")
    if p["type"] == "noul" and p.get("no_value") == "statement":
        # "does a note change this line's X from its statement?": value arrives as (stage value, unchanged?)
        p_yes = a["noul"]
        unchanged = value[1] if isinstance(value, (list, tuple)) else True
        return (1 - p_yes if unchanged else p_yes), ("changed" if p_yes >= 0.5 else "as stated")
    if p["type"] == "noul" and "no_value" in p:
        # yes/no "is there any X?": NO means the field should be `no_value`, YES means anything else
        p_yes = a["noul"]
        pick = "other" if p_yes >= 0.5 else p["no_value"]
        return (1 - p_yes if value == p["no_value"] else p_yes), pick
    if p["type"] == "noul" and "yes_value" in p:
        # yes/no question about a two-value field: YES means the field should be `yes_value`
        p_yes = a["noul"]
        pick = p["yes_value"] if p_yes >= 0.5 else next(k for k in p["criteria"] if k != p["yes_value"])
        return (p_yes if value == p["yes_value"] else 1 - p_yes), pick
    if p["type"] == "noul":
        p_yes = a["noul"]
        return (p_yes if p.get("polarity", "correct") == "correct" else 1 - p_yes), None
    probs = a["probabilities"]
    pick = max(probs, key=probs.get)
    if p.get("answer_is_other"):
        # the question identifies the OTHER option of a two-way split (e.g. asks for the supplier to score the
        # buyer): the stage value is right when it is not the option the model picked
        if value not in probs:
            return 0.0, None
        other = next((k for k in probs if k != pick), pick)
        return 1.0 - probs[value], other
    if p.get("answer_field"):
        return probs.get(value, 0.0), pick  # a stage value outside the options gets 0
    p_yes = probs[p["correct_choice"]]
    return (p_yes if p.get("polarity", "correct") == "correct" else 1 - p_yes), pick


# --------------------------------------------------------------------------- ground truth

def _field_ok(f, doc, exp, line=None):
    if f == "__line_set__":
        return [x["line_id"] for x in doc["lines"]] == [x["line_id"] for x in exp["lines"]]
    if line is None:
        if f in HEADER_FIELDS:
            return doc.get(f) == exp.get(f)
        el = {x["line_id"]: x for x in exp["lines"]}  # line field on a case param: every line must agree
        return all(x["line_id"] in el and el[x["line_id"]].get(f) == x.get(f) for x in doc["lines"])
    e = next((x for x in exp["lines"] if x["line_id"] == line["line_id"]), None)
    return e is not None and e.get(f) == line.get(f)


def param_labels(p, record, doc):
    """{line_id or None: bool correct}, or None without annotations (validation split)."""
    if p.get("label_from") == "parser_flags":  # code-computed, so available on every split
        errs = [f for f in parser_flags(record["input"]) if f["severity"] == "error"
                and (not p.get("label_field") or f["detail"].startswith(p["label_field"] + ":"))]
        if p["scope"] == "case":
            return {None: not errs}
        return {l["line_id"]: not any(f["line_id"] == l["line_id"] for f in errs) for l in doc["lines"]}
    ann = record.get("training_annotations")
    if not ann:
        return None
    exp = ann["expected_final_output"]
    lines = doc["lines"] if p["scope"] == "line" else [None]
    if p.get("label_compare") == "is_no_value":  # only "none vs some" matters for this question
        f, nv = p["answer_field"], p["no_value"]
        el = {x["line_id"]: x for x in exp["lines"]}
        return {l["line_id"]: l["line_id"] in el and (norm_value(l.get(f)) == nv) == (norm_value(el[l["line_id"]].get(f)) == nv)
                for l in lines}
    by_fields = {(l["line_id"] if l else None): all(_field_ok(f, doc, exp, l) for f in p["fields"]) for l in lines}
    if not p.get("mechanism"):
        return by_fields
    if p["mechanism"] not in ann["failure_mechanisms"]:
        return {k: True for k in by_fields}
    if p["scope"] == "case" or all(by_fields.values()):  # mechanism failed; blame the lines whose fields differ
        return {k: False for k in by_fields}
    return by_fields


def expected_value(p, record, line_id):
    ann = record.get("training_annotations")
    if not ann or not p.get("answer_field"):
        return None
    exp = ann["expected_final_output"]
    if line_id is None:
        return norm_value(exp.get(p["answer_field"]))
    e = next((x for x in exp["lines"] if x["line_id"] == line_id), None)
    return norm_value(e.get(p["answer_field"])) if e else None


def field_diff(p, record, doc, line_id):
    if p.get("label_from") == "parser_flags":
        return "; ".join(f["detail"] for f in parser_flags(record["input"])
                         if f["severity"] == "error" and (line_id is None or f["line_id"] == line_id)
                         and (not p.get("label_field") or f["detail"].startswith(p["label_field"] + ":")))
    exp = record["training_annotations"]["expected_final_output"]
    if line_id is None:
        out = []
        for f in p["fields"]:
            if f == "__line_set__":
                a, b = [x["line_id"] for x in doc["lines"]], [x["line_id"] for x in exp["lines"]]
                if a != b:
                    out.append(f"lines: got {a} expected {b}")
            elif f in HEADER_FIELDS and doc.get(f) != exp.get(f):
                out.append(f"{f}: got {doc.get(f)!r} expected {exp.get(f)!r}")
        return "; ".join(out) or "mechanism failed elsewhere in the case"
    line = next(x for x in doc["lines"] if x["line_id"] == line_id)
    e = next((x for x in exp["lines"] if x["line_id"] == line_id), None)
    if e is None:
        return "line should not exist"
    return "; ".join(f"{f}: got {line.get(f)!r} expected {e.get(f)!r}" for f in p["fields"]
                     if e.get(f) != line.get(f)) or "mechanism failed elsewhere in the case"


def doc_correct(stage_params, record, doc):
    """Is this stage's document right on every field its params cover?"""
    if any(p.get("label_from") == "parser_flags" for p in stage_params.values()):
        return not any(f["severity"] == "error" for f in parser_flags(record["input"]))
    ann = record.get("training_annotations")
    if not ann:
        return None
    exp = ann["expected_final_output"]
    for f in {f for p in stage_params.values() for f in p["fields"]}:
        if f == "__line_set__" or f in HEADER_FIELDS:
            if not _field_ok(f, doc, exp):
                return False
        elif not all(_field_ok(f, doc, exp, l) for l in doc["lines"]):
            return False
    return True


# --------------------------------------------------------------------------- stage input vs output

PREVIOUS_DOC = {"normalization": "extractor", "extractor": None, "matching": "normalization", "final": "normalization"}


def _parsed_source(inp):
    tables = _tables_text(inp)
    return "\n\n".join(_parser_blocks(inp)) + ("\n\nTABLE:\n" + tables if tables else "")


def stage_io(kind, record):
    """What a stage received, what it produced, and a field-by-field comparison with the previous stage's
    value and the expected output (training split)."""
    inp = record["input"]
    ann = record.get("training_annotations")
    exp = ann["expected_final_output"] if ann else None
    if kind == "parser":
        view = parser_view(inp)
        return {
            "input": {"source_context": inp["source_context"]},
            "input_text": None,
            "output": inp["parser_output"],
            "fields": [{"field": "line text", "line_id": l["line_id"], "value": l["text"], "previous": None,
                        "expected": None, "ok": None} for l in view["lines"]]
                      + [{"field": "table row", "line_id": l["line_id"],
                          "value": json.dumps(l["table"]) if l["table"] else None, "previous": None,
                          "expected": None, "ok": None} for l in view["lines"]],
            "flags": parser_flags(inp),
        }
    doc = stage_document(kind, inp)
    prev_kind = PREVIOUS_DOC.get(kind)
    prev = stage_document(prev_kind, inp) if prev_kind else None
    if kind in ("extractor", "normalization"):
        received = {"parser_output": {"tables": inp["parser_output"].get("tables", []),
                                      "warnings": inp["parser_output"].get("warnings", [])}}
        input_text = _parsed_source(inp)
        output = {"document": doc}
        if kind == "normalization":
            output["applied_operations"] = inp["normalization_output"].get("applied_operations", [])
            output["raw_extractor_document"] = inp["extractor_output"]["document"]
    elif kind == "matching":
        received = {"extracted_document": inp["normalization_output"]["document"],
                    "master_data": {k: inp["master_data"][k] for k in ("parties", "products")}}
        input_text = None
        output = inp["matching_output"]
    else:  # final
        received = {"extracted_document": inp["normalization_output"]["document"],
                    "matching": {k: v for k, v in inp["matching_output"].items() if k != "product_candidates"},
                    "validation_output": inp["validation_output"]}
        input_text = None
        output = inp["final_output"]

    def cmp(v, e):
        return None if exp is None else (norm_value(v) == norm_value(e))

    fields = []
    for f in HEADER_FIELDS:
        if f in doc:
            fields.append({"field": f, "line_id": None, "value": doc.get(f),
                           "previous": prev.get(f) if prev else None,
                           "expected": exp.get(f) if exp else None, "ok": cmp(doc.get(f), exp.get(f) if exp else None)})
    ids = [l["line_id"] for l in doc["lines"]]
    exp_ids = [l["line_id"] for l in exp["lines"]] if exp else None
    fields.append({"field": "line ids", "line_id": None, "value": ids,
                   "previous": [l["line_id"] for l in prev["lines"]] if prev else None,
                   "expected": exp_ids, "ok": None if exp is None else ids == exp_ids})
    for line in doc["lines"]:
        el = next((x for x in exp["lines"] if x["line_id"] == line["line_id"]), None) if exp else None
        pl = next((x for x in prev["lines"] if x["line_id"] == line["line_id"]), None) if prev else None
        for f, v in line.items():
            if f == "line_id":
                continue
            e = el.get(f) if el else None
            fields.append({"field": f, "line_id": line["line_id"], "value": v,
                           "previous": pl.get(f) if pl else None, "expected": e,
                           "ok": None if exp is None else (el is not None and cmp(v, e))})
    return {"input": received, "input_text": input_text, "output": output, "fields": fields,
            "flags": parser_flags(inp)}


# --------------------------------------------------------------------------- enum audit

def check_enums(Q, verbose=True):
    """Every value any stage document or expected output takes must be exactly one option of its enum."""
    records = load_split("train") + load_split("val")
    issues, report = [], []
    seen = {}
    for sname, scfg in Q.STAGES.items():
        for pname, p in scfg["params"].items():
            if not p.get("answer_field") or "no_value" in p:
                continue
            key = (p["answer_field"], p.get("ask_unless_value"), p.get("criteria_from") or json.dumps(p.get("criteria"), sort_keys=True))
            if key in seen:
                continue
            seen[key] = f"{sname}.{pname}"
            counts, outside = {}, {}
            for r in records:
                inp = r["input"]
                opts = criteria_for(p, inp) or {}
                descs = [d.strip().lower() for d in opts.values()]
                if p.get("criteria_from") and len(set(descs)) != len(descs):
                    issues.append(f"{p['answer_field']}: duplicate option descriptions in case {r['case_id']}")
                docs =[stage_document(k, inp) for k in ("extractor", "normalization", "matching", "final")]
                if r.get("training_annotations"):
                    docs.append(r["training_annotations"]["expected_final_output"])
                for d in docs:
                    targets = d["lines"] if p["scope"] == "line" else [None]
                    for l in targets:
                        v = field_value(d, p["answer_field"], l)
                        if v is None:
                            continue  # e.g. matching doc has no currency
                        v = norm_value(v)
                        if v == p.get("ask_unless_value"):
                            continue
                        counts[v] = counts.get(v, 0) + 1
                        if v not in opts:
                            outside[v] = outside.get(v, 0) + 1
            name = f"{p['answer_field']} ({'per-case ' + p['criteria_from'] if p.get('criteria_from') else 'ENUM'})"
            entry = {"field": p["answer_field"], "source": p.get("criteria_from") or "enum", "used_by": seen[key],
                     "value_counts": counts if not p.get("criteria_from") else {"distinct": len(counts)},
                     "outside_enum": outside, "unused_options": [], "duplicate_descriptions": [],
                     "ambiguous_keys": []}
            if not p.get("criteria_from"):
                crit = p["criteria"]
                entry["unused_options"] = [k for k in crit if k not in counts]
                descs = [d.strip().lower() for d in crit.values()]
                entry["duplicate_descriptions"] = sorted({d for d in descs if descs.count(d) > 1})
                numeric = {}
                for k in crit:
                    try:
                        numeric.setdefault(float(k), []).append(k)
                    except ValueError:
                        pass
                entry["ambiguous_keys"] = [v for v in numeric.values() if len(v) > 1]
            if outside:
                issues.append(f"{name}: values not in options {outside}")
            if entry["duplicate_descriptions"] or entry["ambiguous_keys"]:
                issues.append(f"{name}: options not exclusive {entry['duplicate_descriptions'] or entry['ambiguous_keys']}")
            report.append(entry)
    if verbose:
        print("ENUM AUDIT (all stage documents + expected outputs, both splits)")
        for e in report:
            ok = "OK " if not e["outside_enum"] and not e["duplicate_descriptions"] and not e["ambiguous_keys"] else "!! "
            extra = f"  unused options: {e['unused_options']}" if e["unused_options"] else ""
            print(f" {ok}{e['field']:20} {e['source']:10} {e['value_counts']}{extra}")
            if e["outside_enum"]:
                print(f"     values outside options: {e['outside_enum']}")
        print("exhaustive + exclusive" if not issues else "\n".join("ISSUE " + i for i in issues))
    return report, issues


# --------------------------------------------------------------------------- metrics

def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def brier(scores, labels):
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores) if scores else None


def balanced_accuracy(scores, labels, cut=0.5):
    rates = []
    for cls in (True, False):
        xs = [s for s, y in zip(scores, labels) if y == cls]
        if xs:
            rates.append(sum((s >= cut) == cls for s in xs) / len(xs))
    return sum(rates) / len(rates) if rates else None


def combine(probs, mode):
    if not probs:
        return None
    if mode == "product":
        return math.prod(probs)
    if mode == "min":
        return min(probs)
    if mode == "geomean":
        return math.prod(max(x, 1e-9) for x in probs) ** (1 / len(probs))
    raise ValueError(mode)


def threshold_sweep(scores, labels, target_risk):
    """Accept a case when score >= t, for t at every distinct score (products of many P's get tiny)."""
    rows, suggested, n, good_total = [], None, len(scores), sum(labels)
    for t in sorted(set(scores)):
        acc = [y for s, y in zip(scores, labels) if s >= t]
        row = {
            "threshold": t, "accepted": len(acc), "coverage": len(acc) / n if n else 0,
            "incorrect_among_accepted": (sum(not y for y in acc) / len(acc)) if acc else None,
            "acceptable_recall": (sum(acc) / good_total) if good_total else None,
        }
        rows.append(row)
        if suggested is None and row["incorrect_among_accepted"] <= target_risk:
            suggested = t
    return rows, suggested


def at_threshold(scores, labels, t):
    acc = [y for s, y in zip(scores, labels) if s >= t]
    return {"threshold": t, "accepted": len(acc), "coverage": len(acc) / len(scores) if scores else 0,
            "incorrect_among_accepted": (sum(not y for y in acc) / len(acc)) if acc else None,
            "acceptable_recall": (sum(acc) / sum(labels)) if sum(labels) else None}


def param_gate(m, gate):
    if not m["written"]:
        return False, ["no question written"]
    reasons = []
    if m["truncated"] and not gate.get("allow_truncation"):
        reasons.append(f"{m['truncated']} calls truncated")
    if m["errors"] >= gate["min_errors"] and (m["auroc"] or 0) < gate["min_auroc"]:
        reasons.append(f"AUROC {m['auroc'] or 0:.2f} < {gate['min_auroc']}")
    if m["labeled"] and (m["balanced_accuracy"] or 0) < gate["min_balanced_accuracy"]:
        reasons.append(f"balanced acc {m['balanced_accuracy'] or 0:.2f} < {gate['min_balanced_accuracy']}")
    return not reasons, reasons


def stage_hash(stage_cfg, gate):
    return hashlib.sha256(json.dumps([stage_cfg, gate], sort_keys=True).encode()).hexdigest()[:16]



# --------------------------------------------------------------------------- run log

RUN_LOG = os.path.join(RUNS_DIR, "run_log.txt")


def run_log_entry(run_id, s):
    """Human-readable record of one run: what was run, every question as asked, and how it scored."""
    def f(x, nd=3):
        return "-" if x is None else f"{x:.{nd}f}"

    scope = f"only {s['only']}" if s.get("only") else f"up to {s['last_stage']}"
    L = ["=" * 100,
         f"RUN {run_id}   {s['created']}   split={s['split']}"
         + (f" ({s['subset']} half)" if s.get("subset") else "") + f"   cases={s['n_cases']}   {scope}"
         + (f"   case={s['case']}" if s.get("case") else "")
         + ("" if s.get("gate_eligible", True) else "   [sample run: does not count for gates]"),
         "=" * 100]
    for st in s["stages"]:
        cum = st["cumulative"][s["cumulative_mode"]]
        status = "from cache" if st.get("context") else ("PASSED" if st["passed"] else "NOT PASSED")
        L.append(f"\n[{st['name']}] {status}   questions {st['written']}/{st['total_params']}   "
                 f"cumulative AUROC {f(cum['auroc'])}")
        if st.get("context"):
            continue
        for pn, m in st["params"].items():
            if not m["written"]:
                L.append(f"  - {pn}: no question")
                continue
            gate = "" if m["passed"] is None else ("ok" if m["passed"] else "FAIL (" + "; ".join(m["gate_reasons"]) + ")")
            cut = m.get("decision_threshold", 0.5)
            L.append(f"  - {pn} [{m['type']}, {m['scope']}{'' if cut == 0.5 else f', decision threshold {cut}'}] {gate}")
            L.append(f"      checks {m['checks_passed']}/{m['labeled']}   caught {m['caught']}/{m['errors']}   "
                     f"false alarms {m['false_alarms']}   AUROC {f(m['auroc'], 2)}   bal.acc {f(m['balanced_accuracy'], 2)}"
                     f"   value acc {f(m['value_accuracy'], 2)}   mean P {f(m['mean_p'], 2)}   max tokens {m['max_tokens']}"
                     + (f"   TRUNCATED {m['truncated']}" if m["truncated"] else ""))
            L.append(f"      Q: {m['instructions']}")
            L.append(f"      evidence: {', '.join(m['evidence'])}")
    fin = s["final"]
    a = fin["at_threshold"]
    L.append(f"\nconfidence after {s['last_stage']}: AUROC {f(fin['auroc'])}  Brier {f(fin['brier'])}  "
             f"(pipeline's own {f(fin['pipeline_baseline_auroc'])})  "
             f"threshold {s['threshold']}: coverage {f(a['coverage'])}, incorrect among accepted "
             f"{f(a['incorrect_among_accepted'])}  suggested threshold "
             f"{'-' if fin['suggested_threshold'] is None else format(fin['suggested_threshold'], '.4g')}")
    if s.get("blocked_at"):
        L.append(f"BLOCKED at {s['blocked_at']}")
    return "\n".join(L) + "\n\n"


def append_run_log(run_id, summary):
    os.makedirs(RUNS_DIR, exist_ok=True)
    entry = run_log_entry(run_id, summary)
    with open(RUN_LOG, "a") as f:
        f.write(entry)
    with open(os.path.join(RUNS_DIR, run_id, "report.txt"), "w") as f:
        f.write(entry)


def rebuild_run_log():
    """Regenerate run_log.txt from every run directory, oldest first."""
    runs = sorted(d for d in os.listdir(RUNS_DIR) if os.path.exists(os.path.join(RUNS_DIR, d, "summary.json")))
    with open(RUN_LOG, "w") as f:
        for d in runs:
            with open(os.path.join(RUNS_DIR, d, "summary.json")) as sf:
                f.write(run_log_entry(d, json.load(sf)))
    return len(runs)

# --------------------------------------------------------------------------- runner

def in_subset(case_id, subset):
    """Fixed half-split of a split: "tune" (tweak wording here) vs "holdout" (confirm here, never tune)."""
    if not subset:
        return True
    half = int(hashlib.sha256(("half" + case_id).encode()).hexdigest(), 16) % 2
    return half == (0 if subset == "tune" else 1)


def pick_cases(records, limit, case_id, seed=7, subset=None):
    if case_id:
        return [r for r in records if r["case_id"] == case_id]
    records = [r for r in records if in_subset(r["case_id"], subset)]
    if not limit or limit >= len(records):
        return records
    rng = random.Random(seed)
    good = [r for r in records if r["target"]["acceptable_for_auto_processing"]]
    bad = [r for r in records if not r["target"]["acceptable_for_auto_processing"]]
    rng.shuffle(good)
    rng.shuffle(bad)
    return good[: limit // 2] + bad[: limit - limit // 2]


def train_gate_passed(stage, h):
    """Did the latest FULL train run (no --case / --limit) with these exact questions pass this stage?"""
    if not os.path.isdir(RUNS_DIR):
        return False
    for d in sorted(os.listdir(RUNS_DIR), reverse=True):
        path = os.path.join(RUNS_DIR, d, "summary.json")
        if not d.endswith("-train") or not os.path.exists(path):
            continue
        with open(path) as f:
            s = json.load(f)
        if not s.get("gate_eligible", True):
            continue
        for st in s["stages"]:
            if st["name"] == stage and st["hash"] == h and not st.get("context"):
                return st["passed"]
    return False


def run_calls(sname, jobs, args, log, cache_only=False):
    def run(job):
        try:
            resp, cached = answer(job["payload"], use_cache=not args.no_cache, cache_only=cache_only)
            return job, resp, cached, None
        except Exception as e:  # noqa: BLE001 - every failure is surfaced in the report
            return job, None, False, str(e)

    out, n_cached, errors, t0 = [], 0, [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (job, resp, cached, err) in enumerate(ex.map(run, jobs), 1):
            n_cached += cached
            if err:
                errors.append({"case_id": job["case_id"], "param": job["param"], "line_id": job["line_id"], "error": err})
            elif resp is not None:  # None = not cached in cache-only mode
                out.append((job, resp))
            if i % 100 == 0 or i == len(jobs):
                log(f"PROGRESS {sname} {i}/{len(jobs)} cached={n_cached} errors={len(errors)} {time.time() - t0:.0f}s")
    for e in errors[:5]:
        log(f"  ERROR {e}")
    return out, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="train")
    ap.add_argument("--stage", help="run up to and including this stage (default: all, stopping at first failing gate)")
    ap.add_argument("--only", metavar="STAGE", help="run just this stage (earlier stages read from cache for the "
                    "cumulative score; locked until earlier stages passed on train, unless --force)")
    ap.add_argument("--params", help="with --only: run just these params of that stage (comma-separated; sample run)")
    ap.add_argument("--force", action="store_true", help="continue past failing gates")
    ap.add_argument("--limit", type=int, default=0, help="balanced sample size (0 = all cases)")
    ap.add_argument("--case", help="run a single case_id")
    ap.add_argument("--subset", choices=["tune", "holdout"], help="fixed half of the split (tune wording on one, confirm on the other)")
    ap.add_argument("--preview", metavar="STAGE.PARAM", help="print the payload for one param on the first case")
    ap.add_argument("--check-enums", action="store_true", help="verify enum options are exhaustive and exclusive")
    ap.add_argument("--parser-flags", action="store_true", help="list code-computed parser mistakes and exit")
    ap.add_argument("--rebuild-log", action="store_true", help="regenerate runs/run_log.txt from all runs")
    ap.add_argument("--inspect", metavar="STAGE", help="with --case: show that stage's input vs output vs expected")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    Q = load_questions()
    if args.rebuild_log:
        print(f"rebuilt {RUN_LOG} from {rebuild_run_log()} runs")
        return
    if args.check_enums:
        _, issues = check_enums(Q)
        sys.exit(1 if issues else 0)

    stage_names = list(Q.STAGES)
    if args.case and args.case.endswith(TAMPER_SUFFIX) and (args.preview or args.inspect):
        cases = [get_record(args.split, args.case)]
    else:
        cases = pick_cases(load_split(args.split), args.limit, args.case, subset=args.subset)
    if args.parser_flags:
        from collections import Counter
        c = Counter((f["type"], f["severity"]) for r in cases for f in parser_flags(r["input"]))
        print(f"{args.split}: {len(cases)} cases, "
              f"{sum(any(f['severity'] == 'error' for f in parser_flags(r['input'])) for r in cases)} with parser errors")
        for (t, sev), n in sorted(c.items()):
            print(f"  {sev:8} {t:24} {n}")
        for r in cases:
            for f in parser_flags(r["input"]):
                if f["severity"] == "error":
                    print(f"  {r['case_id']} line {f['line_id']}: {f['detail']}")
        return
    if not cases:
        sys.exit("no cases selected")

    if args.inspect:
        if args.case and args.case.endswith(TAMPER_SUFFIX):
            cases = [get_record(args.split, args.case)]
        io = stage_io(Q.STAGES[args.inspect]["document"], cases[0])
        print(f"# {cases[0]['case_id']}  stage {args.inspect}  "
              f"(acceptable={cases[0]['target']['acceptable_for_auto_processing']})\n")
        print("== INPUT ==")
        print(io["input_text"] or json.dumps(io["input"], indent=1)[:4000])
        print("\n== OUTPUT vs PREVIOUS STAGE vs EXPECTED ==")
        print(f"{'field':20} {'line':5} {'output':28} {'previous':28} {'expected':28} ok")
        for f in io["fields"]:
            ok = {True: "yes", False: "NO", None: "-"}[f["ok"]]
            print(f"{f['field']:20} {str(f['line_id'] or ''):5} {str(f['value'])[:28]:28} {str(f['previous'])[:28]:28} "
                  f"{str(f['expected'])[:28]:28} {ok}")
        for fl in io["flags"]:
            print(f"parser {fl['severity']}: {fl['type']} line {fl['line_id']}: {fl['detail']}")
        return

    if args.preview:
        sname, pname = args.preview.split(".", 1)
        scfg = Q.STAGES[sname]
        doc = stage_document(scfg["document"], cases[0]["input"])
        line_id, payload, value = build_payloads(pname, scfg["params"][pname], cases[0], doc)[0]
        if isinstance(payload, list):
            payload = {"one question per note": payload or "(no notes -> answered NO without a call)"}
        print(f"# case {cases[0]['case_id']}  line {line_id}  stage value {value}  (~{len(json.dumps(payload)) // 3} tokens est.)")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for name in (args.stage, args.only):
        if name and name not in Q.STAGES:
            sys.exit(f"unknown stage {name}; stages: {', '.join(stage_names)}")
    last = stage_names.index(args.only or args.stage) if (args.only or args.stage) else len(stage_names) - 1
    if args.only and not args.force:
        locked = [s for s in stage_names[:last] if not train_gate_passed(s, stage_hash(Q.STAGES[s], Q.GATE))]
        if locked:
            sys.exit(f"Stage '{args.only}' is locked: {', '.join(locked)} {'has' if len(locked) == 1 else 'have'} not "
                     f"passed a full train run with the current questions. Pass {'it' if len(locked) == 1 else 'them'} "
                     f"first, or use --force.")
    labeled_split = args.split == "train"

    def log(msg):
        print(msg, flush=True)

    enum_report, enum_issues = check_enums(Q, verbose=False)
    for i in enum_issues:
        log(f"ENUM ISSUE {i}")
    log(f"split={args.split} cases={len(cases)} stages={' -> '.join(stage_names[:last + 1])}"
        + (f"  (only running '{args.only}'; earlier stages read from cache)" if args.only else ""))

    by_case = {r["case_id"]: r for r in cases}
    calls, stages_out = [], []
    case_state = {r["case_id"]: {} for r in cases}
    history = []  # [(stage, {param: {case: prob}}, params, carries_forward)] for the cumulative score
    blocked = None

    for si, sname in enumerate(stage_names[: last + 1]):
        scfg = Q.STAGES[sname]
        params = scfg["params"]
        active = {k: v for k, v in params.items() if v.get("instructions", "").strip()}
        if args.params and sname == args.only:
            wanted = set(args.params.split(","))
            unknown = wanted - set(params)
            if unknown:
                sys.exit(f"unknown params for {sname}: {', '.join(sorted(unknown))}")
            active = {k: v for k, v in active.items() if k in wanted}
            params = {k: v for k, v in params.items() if k in wanted}
        synth = tampered_records(cases, scfg.get("synthetic_errors", 0)) if scfg["document"] == "parser" else []
        for r in synth:
            by_case[r["case_id"]] = r
        stage_cases = cases + synth  # synthetic copies count in this stage's metrics only, never in confidence
        docs = {r["case_id"]: stage_document(scfg["document"], r["input"]) for r in stage_cases}
        if synth:
            log(f"[{sname}] + {len(synth)} tampered copies with a planted table/text conflict")
        jobs = [{"case_id": r["case_id"], "param": pname, "line_id": lid, "payload": pl, "value": val}
                for r in stage_cases for pname, p in active.items()
                for lid, pl, val in build_payloads(pname, p, r, docs[r["case_id"]])]
        context = bool(args.only) and sname != args.only  # earlier stage: cached answers only, no API calls
        log(f"[{sname}] {len(active)}/{len(params)} questions written, {len(jobs)} calls"
            + (" (from cache, for the cumulative score)" if context else ""))
        results, errors = run_calls(sname, jobs, args, log, cache_only=context)

        rows = []
        for job, resp in results:
            p, rec, doc = active[job["param"]], by_case[job["case_id"]], docs[job["case_id"]]
            labels = param_labels(p, rec, doc)
            label = None if labels is None else labels.get(job["line_id"])
            pc, pick = read_answer(job["param"], p, resp, job["value"])
            rows.append({
                "stage": sname, "param": job["param"], "case_id": job["case_id"], "line_id": job["line_id"],
                "p": round(pc, 4), "label": label, "tokens": resp.get("usage", {}).get("input_tokens", 0),
                "value": job["value"][0] if isinstance(job["value"], tuple) else job["value"], "pick": pick,
                "expected": expected_value(p, rec, job["line_id"]),
                "diff": field_diff(p, rec, doc, job["line_id"]) if label is False else "",
            })
        calls += rows

        pprob = {}  # param -> case -> prob (line params: product over lines)
        for r in rows:
            pprob.setdefault(r["param"], {}).setdefault(r["case_id"], 1.0)
            pprob[r["param"]][r["case_id"]] *= r["p"]
        history.append((sname, pprob, params, scfg.get("carry_forward", True)))

        # ---- per-param metrics + gate
        pmetrics = {}
        for pname, p in params.items():
            prs = [r for r in rows if r["param"] == pname]
            toks = [r["tokens"] for r in prs]
            picks = [r for r in prs if r["pick"] is not None and r["expected"] is not None]
            cut = p.get("decision_threshold", 0.5)
            # "says correct": value choice -> the model's top pick is the stage's value (with several options
            # the winner is often < 0.5); everything else -> P(correct) >= cut
            argmax = bool(p.get("answer_field")) and "decision_threshold" not in p
            for r in prs:
                r["says_correct"] = r["p"] >= 0.5 if ("no_value" in p or "yes_value" in p or p.get("stage_check")) else \
                    ((r["pick"] == r["value"]) if argmax else (r["p"] >= cut))
            lab = [r for r in prs if r["label"] is not None]
            sc, lb = [r["p"] for r in lab], [r["label"] for r in lab]
            ok = [r["says_correct"] for r in lab]
            m = {
                "decision_threshold": cut,
                "written": pname in active, "instructions": p.get("instructions", ""), "type": p["type"],
                "scope": p["scope"], "evidence": p["evidence"], "fields": p["fields"],
                "mechanism": p.get("mechanism"), "answer_field": p.get("answer_field"),
                "options": p.get("criteria_from") or list((p.get("criteria") or {}).keys()),
                "calls": len(prs), "labeled": len(lab),
                "errors": sum(not y for y in lb), "caught": sum((not y) and not o for o, y in zip(ok, lb)),
                "false_alarms": sum(y and not o for o, y in zip(ok, lb)),
                "checks_passed": sum(o == y for o, y in zip(ok, lb)),
                "decision": "top pick = stage value" if argmax else f"P >= {cut}",
                "auroc": auroc(sc, lb), "brier": brier(sc, lb),
                "balanced_accuracy": balanced_accuracy([1.0 if o else 0.0 for o in ok], lb),
                "value_accuracy": None if p.get("no_value") == "statement" else
                (sum((r["pick"] == r["expected"]) if "no_value" not in p else
                                       ((r["pick"] == p["no_value"]) == (r["expected"] == p["no_value"]))
                                       for r in picks) / len(picks)) if picks else None,
                "mean_p": (sum(r["p"] for r in prs) / len(prs)) if prs else None,
                "max_tokens": max(toks) if toks else 0,
                "truncated": sum(t >= CONTEXT_LIMIT for t in toks),
            }
            if p.get("stage_check"):
                det = []
                for r in prs:
                    rec = by_case[r["case_id"]]
                    ann = rec.get("training_annotations")
                    el = next((x for x in ann["expected_final_output"]["lines"] if x["line_id"] == r["line_id"]),
                              None) if ann else None
                    if el is not None:
                        # rebuild P(yes) from P(consistent) and the stage rule result
                        p_yes = r["p"] if r["value"] else 1 - r["p"]
                        det.append((p_yes, _stage_check(p["stage_check"], rec["input"], el)))
                m["detection_auroc"] = auroc([x for x, _ in det], [y for _, y in det])
                m["situations"] = sum(y for _, y in det)
                m["detected"] = sum(y and x >= 0.5 for x, y in det)
                m["detection_false_alarms"] = sum((not y) and x >= 0.5 for x, y in det)
            m["passed"], m["gate_reasons"] = param_gate(m, Q.GATE) if labeled_split else (None, [])
            m["label_source"] = "parser flags (code)" if p.get("label_from") == "parser_flags" else \
                ("failure_mechanisms" if p.get("mechanism") else "expected_final_output")
            pmetrics[pname] = m

        # ---- stage and cumulative confidence per case
        for r in cases:
            cid = r["case_id"]
            keep, all_probs = [], []
            for j, (_, pp, sparams, carries) in enumerate(history):
                later = {f for _, pp2, sp2, _ in history[j + 1:] for pn2 in pp2 for f in sp2[pn2]["fields"]}
                for pn in pp:
                    if cid in pp[pn]:
                        all_probs.append(pp[pn][cid])
                        # carry forward unless re-judged by a later stage (or the stage opts out, e.g. parser:
                        # its mistakes can still be corrected downstream)
                        if carries and not set(sparams[pn]["fields"]) <= later:
                            keep.append(pp[pn][cid])
            case_state[cid][sname] = {
                "stage_conf": combine([pprob[p][cid] for p in pprob if cid in pprob[p]], Q.COMBINE),
                "cum": {"carry_forward": combine(keep, Q.COMBINE), "product": combine(all_probs, "product")},
                "stage_label": doc_correct(active, r, docs[cid]) if active else None,
                "params": {p: round(pprob[p][cid], 4) for p in pprob if cid in pprob[p]},
            }

        final_labels = [r["target"]["acceptable_for_auto_processing"] for r in cases]
        cs = [case_state[r["case_id"]][sname] for r in cases]

        def filled(xs):
            return [0.5 if x is None else x for x in xs]

        slabels = [c["stage_label"] for c in cs]
        if labeled_split and not context:
            passed = bool(active) and len(active) == len(params) and \
                all(m["passed"] for m in pmetrics.values()) and not errors
        else:
            passed = train_gate_passed(sname, stage_hash(scfg, Q.GATE))
        stages_out.append({
            "name": sname, "document": scfg["document"], "hash": stage_hash(scfg, Q.GATE), "passed": passed,
            "context": context, "answered": len(results), "expected_calls": len(jobs),
            "written": len(active), "total_params": len(params), "api_errors": len(errors), "params": pmetrics,
            "stage_docs_correct": sum(bool(x) for x in slabels) if labeled_split and active else None,
            "stage_auroc_vs_stage_label": auroc(filled([c["stage_conf"] for c in cs]), slabels)
            if labeled_split and active else None,
            "cumulative": {mode: ({"auroc": None, "brier": None} if all(c["cum"][mode] is None for c in cs) else
                                  {"auroc": auroc(filled([c["cum"][mode] for c in cs]), final_labels),
                                   "brier": brier(filled([c["cum"][mode] for c in cs]), final_labels)})
                           for mode in ("carry_forward", "product")},
        })
        n_pass = sum(1 for m in pmetrics.values() if m["passed"])
        log(f"[{sname}] gate {'PASSED' if passed else 'NOT PASSED'}  params passing {n_pass}/{len(pmetrics)}  "
            f"cumulative AUROC {fmt(stages_out[-1]['cumulative'][Q.CUMULATIVE]['auroc'])}")
        if not passed and not args.force and si < last and not context:
            blocked = sname
            log(f"Stopped: stage '{sname}' has not passed its gate. Fix its questions (or use --force).")
            break

    # ---- threshold sweep on the cumulative score after the last stage that ran
    last_stage = stages_out[-1]["name"]
    final_labels = [r["target"]["acceptable_for_auto_processing"] for r in cases]
    last_scores = [case_state[r["case_id"]][last_stage]["cum"][Q.CUMULATIVE] for r in cases]
    no_score = all(x is None for x in last_scores)
    last_scores = [0.5 if x is None else x for x in last_scores]
    sweep, suggested = threshold_sweep(last_scores, final_labels, Q.TARGET_RISK)
    summary = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": args.split, "n_cases": len(cases), "combine": Q.COMBINE, "cumulative_mode": Q.CUMULATIVE,
        "gate": Q.GATE, "threshold": Q.THRESHOLD, "target_risk": Q.TARGET_RISK,
        "stage_order": stage_names, "stages": stages_out, "only": args.only, "case": args.case,
        "gate_eligible": labeled_split and not args.case and not args.limit and not args.params and not args.subset,
        "subset": args.subset, "blocked_at": blocked, "last_stage": last_stage,
        "enum_issues": enum_issues, "enums": enum_report,
        "final": {
            "auroc": None if no_score else auroc(last_scores, final_labels),
            "brier": None if no_score else brier(last_scores, final_labels),
            "at_threshold": at_threshold(last_scores, final_labels, Q.THRESHOLD),
            "suggested_threshold": None if no_score else suggested,
            "pipeline_baseline_auroc": auroc([r["input"]["validation_output"].get("reported_confidence", 0.5)
                                              for r in cases], final_labels),
        },
        "sweep": sweep,
    }
    cases_out = [{"case_id": r["case_id"], "label": r["target"]["acceptable_for_auto_processing"],
                  "mechanisms": (r.get("training_annotations") or {}).get("failure_mechanisms", []),
                  "parser_flags": parser_flags(r["input"]),
                  "stages": case_state[r["case_id"]]} for r in cases]
    pf = [f for c in cases_out for f in c["parser_flags"]]
    summary["parser_flags"] = {
        "cases_with_errors": sum(any(f["severity"] == "error" for f in c["parser_flags"]) for c in cases_out),
        "cases_with_warnings": sum(any(f["severity"] == "warning" for f in c["parser_flags"]) for c in cases_out),
        "by_type": {t: {"severity": next(f["severity"] for f in pf if f["type"] == t),
                        "count": sum(f["type"] == t for f in pf)} for t in sorted({f["type"] for f in pf})},
    }

    run_dir = os.path.join(RUNS_DIR, f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{args.split}")
    os.makedirs(run_dir, exist_ok=True)
    for name, obj in (("summary.json", summary), ("cases.json", cases_out), ("calls.json", calls),
                      ("questions_snapshot.json", {"STAGES": Q.STAGES, "GATE": Q.GATE})):
        with open(os.path.join(run_dir, name), "w") as f:
            json.dump(obj, f, indent=1 if name == "summary.json" else None)
    append_run_log(os.path.basename(run_dir), summary)
    print_report(summary)
    log(f"RUN_DIR {run_dir}")
    log(f"logged to {RUN_LOG}")


def fmt(x, nd=3):
    return "  -  " if x is None else f"{x:.{nd}f}"


def print_report(s):
    print("\n================ STAGES ================")
    for st in s["stages"]:
        cum = st["cumulative"][s["cumulative_mode"]]
        print(f"\n[{st['name']}]  gate: {'PASSED' if st['passed'] else 'NOT PASSED'}   questions {st['written']}/{st['total_params']}"
              f"   cumulative AUROC {fmt(cum['auroc'])}  Brier {fmt(cum['brier'])}")
        print(f"  {'param':20} {'checks':>11} {'caught':>9} {'false_al':>8} {'auroc':>6} {'bal_acc':>7} {'val_acc':>7} {'tok':>5}  gate")
        for pn, m in st["params"].items():
            if not m["written"]:
                print(f"  {pn:20} {'-- no question --':>30}")
                continue
            chk = f"{m['checks_passed']}/{m['labeled']}" if m["labeled"] else f"{m['calls']} calls"
            gate = "" if m["passed"] is None else ("ok" if m["passed"] else "FAIL: " + "; ".join(m["gate_reasons"]))
            print(f"  {pn:20} {chk:>11} {str(m['caught']) + '/' + str(m['errors']):>9} {m['false_alarms']:>8} "
                  f"{fmt(m['auroc'], 2):>6} {fmt(m['balanced_accuracy'], 2):>7} {fmt(m['value_accuracy'], 2):>7} "
                  f"{m['max_tokens']:>5}  {gate}")
    f = s["final"]
    print(f"\n================ CONFIDENCE AFTER '{s['last_stage']}' ================")
    print(f"AUROC {fmt(f['auroc'])}  Brier {fmt(f['brier'])}  (pipeline's own score: AUROC {fmt(f['pipeline_baseline_auroc'])})")
    a = f["at_threshold"]
    print(f"threshold {s['threshold']}: coverage {fmt(a['coverage'])}, incorrect among accepted {fmt(a['incorrect_among_accepted'])}")
    st = f["suggested_threshold"]
    print(f"suggested threshold for <= {s['target_risk']:.0%} incorrect among accepted: "
          f"{'none reaches it' if st is None else f'{st:.4g}'}")
    pfl = s.get("parser_flags", {})
    if pfl:
        print(f"\nPARSER FLAGS: {pfl['cases_with_errors']} cases with errors, {pfl['cases_with_warnings']} with warnings  "
              + ", ".join(f"{t} ({v['severity']}) x{v['count']}" for t, v in pfl["by_type"].items()))
    if not s.get("gate_eligible"):
        print("\nNote: sample/single-case/val run -- gate results shown for info, they don't lock or unlock stages.")
    if s["blocked_at"]:
        print(f"\nBLOCKED at stage '{s['blocked_at']}' -- later stages not run.")


if __name__ == "__main__":
    main()
