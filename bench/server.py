"""
Local UI for the stage bench.   bin/python bench/server.py [--port 8765]   ->  http://localhost:8765

Reads bench/questions.py and bench/runs/*, and can launch bench.py runs in the background.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

JOB = {"proc": None, "log": [], "run_id": None, "returncode": None, "args": None}
LOCK = threading.Lock()
_split_cache = {}


def cases_for(split):
    if split not in _split_cache:
        _split_cache[split] = {r["case_id"]: r for r in bench.load_split(split)}
    return _split_cache[split]


def config():
    Q = bench.load_questions()
    stages = []
    for sname, scfg in Q.STAGES.items():
        params = []
        for pname, p in scfg["params"].items():
            params.append({
                "name": pname, "type": p["type"], "instructions": p.get("instructions", ""),
                "scope": p["scope"], "evidence": p["evidence"], "fields": p["fields"],
                "mechanism": p.get("mechanism"), "answer_field": p.get("answer_field"),
                "label_from": p.get("label_from"),
                "options": p.get("criteria_from") or p.get("criteria"),
                "hash": bench.stage_hash(p, {}),
            })
        h = bench.stage_hash(scfg, Q.GATE)
        stages.append({"name": sname, "document": scfg["document"], "params": params,
                       "carry_forward": scfg.get("carry_forward", True), "hash": h,
                       "passed_on_train": bench.train_gate_passed(sname, h)})
    for i, st in enumerate(stages):  # locked until every earlier stage passed on train with current questions
        st["locked_by"] = [x["name"] for x in stages[:i] if not x["passed_on_train"]]
    return {"stages": stages, "gate": Q.GATE, "combine": Q.COMBINE, "cumulative": Q.CUMULATIVE,
            "threshold": Q.THRESHOLD, "target_risk": Q.TARGET_RISK}


def list_runs():
    out = []
    if not os.path.isdir(bench.RUNS_DIR):
        return out
    for d in sorted(os.listdir(bench.RUNS_DIR), reverse=True):
        path = os.path.join(bench.RUNS_DIR, d, "summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            s = json.load(f)
        out.append({
            "id": d, "split": s["split"], "created": s["created"], "n_cases": s["n_cases"],
            "last_stage": s["last_stage"], "blocked_at": s["blocked_at"], "final_auroc": s["final"]["auroc"],
            "only": s.get("only"), "case": s.get("case"), "gate_eligible": s.get("gate_eligible", True),
            "stages": [{"name": st["name"], "passed": st["passed"], "written": st["written"],
                        "context": st.get("context", False),
                        "total": st["total_params"], "hash": st["hash"],
                        "cum_auroc": st["cumulative"][s["cumulative_mode"]]["auroc"]} for st in s["stages"]],
        })
    return out


def load_run(run_id):
    run_dir = os.path.join(bench.RUNS_DIR, os.path.basename(run_id))
    out = {}
    for name in ("summary", "cases", "calls"):
        with open(os.path.join(run_dir, name + ".json")) as f:
            out[name] = json.load(f)
    return out


def preview(stage, param, case_id, split, line_id=None):
    Q = bench.load_questions()
    scfg = Q.STAGES[stage]
    rec = bench.get_record(split, case_id) if case_id.endswith(bench.TAMPER_SUFFIX) else cases_for(split)[case_id]
    doc = bench.stage_document(scfg["document"], rec["input"])
    payloads = bench.build_payloads(param, scfg["params"][param], rec, doc)
    for lid, payload, value in payloads:
        if line_id in (None, "", "null") or lid == line_id:
            return {"line_id": lid, "stage_value": value, "payload": payload}
    return {"error": "line not found"}


def case_io(stage, case_id, split):
    Q = bench.load_questions()
    rec = bench.get_record(split, case_id) if case_id.endswith(bench.TAMPER_SUFFIX) else cases_for(split)[case_id]
    io = bench.stage_io(Q.STAGES[stage]["document"], rec)
    io.update(case_id=case_id, stage=stage, label=rec["target"]["acceptable_for_auto_processing"],
              mechanisms=(rec.get("training_annotations") or {}).get("failure_mechanisms", []),
              case_ids=list(cases_for(split)))
    return io


def start_run(body):
    with LOCK:
        if JOB["proc"] and JOB["proc"].poll() is None:
            return {"error": "a run is already in progress"}
        cmd = [sys.executable, "-u", os.path.join(HERE, "bench.py"), "--split", body.get("split", "train")]
        if body.get("only"):
            cmd += ["--only", body["only"]]
        elif body.get("stage"):
            cmd += ["--stage", body["stage"]]
        if body.get("case"):
            cmd += ["--case", body["case"]]
        if body.get("force"):
            cmd.append("--force")
        if body.get("limit"):
            cmd += ["--limit", str(int(body["limit"]))]
        JOB.update(log=[], run_id=None, returncode=None, args=body)
        JOB["proc"] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump(proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            with LOCK:
                JOB["log"].append(line)
                JOB["log"] = JOB["log"][-400:]
                if line.startswith("RUN_DIR "):
                    JOB["run_id"] = os.path.basename(line.split(" ", 1)[1])
        proc.wait()
        with LOCK:
            JOB["returncode"] = proc.returncode

    threading.Thread(target=pump, args=(JOB["proc"],), daemon=True).start()
    return {"ok": True}


def cancel_run():
    with LOCK:
        proc = JOB["proc"]
        if not proc or proc.poll() is not None:
            return {"error": "no run in progress"}
        proc.terminate()
        JOB["log"].append("CANCELLED by user -- answers fetched so far stay cached")
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"ok": True}


def job_status():
    with LOCK:
        running = bool(JOB["proc"] and JOB["proc"].poll() is None)
        return {"running": running, "log": JOB["log"][-60:], "run_id": JOB["run_id"],
                "returncode": JOB["returncode"], "args": JOB["args"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "ui", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/config":
                return self._send(200, config())
            if u.path == "/api/enums":
                report, issues = bench.check_enums(bench.load_questions(), verbose=False)
                return self._send(200, {"report": report, "issues": issues})
            if u.path == "/api/runs":
                return self._send(200, list_runs())
            if u.path == "/api/run":
                return self._send(200, load_run(q["id"]))
            if u.path == "/api/preview":
                return self._send(200, preview(q["stage"], q["param"], q["case"], q.get("split", "train"), q.get("line")))
            if u.path == "/api/case":
                return self._send(200, case_io(q["stage"], q["case"], q.get("split", "train")))
            if u.path == "/api/summary":
                path = os.path.join(HERE, "SUMMARY.md")
                text = open(path).read() if os.path.exists(path) else "# No summary yet\n\nWrite bench/SUMMARY.md."
                return self._send(200, {"markdown": text})
            if u.path == "/api/job":
                return self._send(200, job_status())
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 - show the error in the UI
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        try:
            if u.path == "/api/run":
                return self._send(200, start_run(body))
            if u.path == "/api/cancel":
                return self._send(200, cancel_run())
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Stage bench UI: http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
