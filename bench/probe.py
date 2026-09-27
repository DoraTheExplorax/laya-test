"""
Try question variants for one param without editing questions.py.  Results go to runs/run_log.txt.
Scores the tune half and the holdout half of the training split side by side: pick wording on tune,
keep it only if holdout agrees.

    bin/python bench/probe.py STAGE.PARAM --limit 20 --variant "name=instructions" [--variant ...]
                              [--evidence slice1,slice2]
"""
import argparse
import datetime
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def evaluate(stage, pname, p, cases, workers=12):
    scfg = bench.load_questions().STAGES[stage]
    jobs = []
    for r in cases:
        doc = bench.stage_document(scfg["document"], r["input"])
        labels = bench.param_labels(p, r, doc) or {}
        for lid, payload, value in bench.build_payloads(pname, p, r, doc):
            jobs.append((r, lid, payload, value, labels.get(lid)))

    def run(job):
        r, lid, payload, value, label = job
        try:
            resp, _ = bench.answer(payload)
        except Exception as e:  # noqa: BLE001 - one failed call must not sink the whole probe
            return {"case": r["case_id"], "line": lid, "error": str(e), "label": None, "pick": None,
                    "expected": None, "p": None, "value": value}
        pc, pick = bench.read_answer(pname, p, resp, value)
        det = None
        if p.get("stage_check") and r.get("training_annotations"):
            el = next((x for x in r["training_annotations"]["expected_final_output"]["lines"] if x["line_id"] == lid), None)
            if el is not None:
                a = resp["answers"].get(pname) or resp["answers"]["*"]
                det = (a["noul"], bench._stage_check(p["stage_check"], r["input"], el))
        return {"case": r["case_id"], "line": lid, "p": pc, "pick": pick, "value": value, "label": label,
                "expected": bench.expected_value(p, r, lid), "det": det}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(run, jobs))
    failed = [x for x in rows if x.get("error")]
    if failed:
        print(f"  ({len(failed)} calls failed after retries and were skipped: {failed[0]['error']})")
    rows = [x for x in rows if not x.get("error")]
    lab = [x for x in rows if x["label"] is not None]
    # same decision rule as bench.py
    binary = "no_value" in p or "yes_value" in p or bool(p.get("stage_check"))
    argmax = bool(p.get("answer_field")) and not binary and "decision_threshold" not in p
    says = [(x["pick"] == x["value"]) if argmax else (x["p"] >= p.get("decision_threshold", 0.5)) for x in lab]
    picks = [x for x in rows if x["pick"] is not None and x["expected"] is not None]
    det = [x["det"] for x in rows if x.get("det")]
    return {
        "det": (f"detects {sum(y and q >= 0.5 for q, y in det)}/{sum(y for _, y in det)} situations, "
                f"{sum((not y) and q >= 0.5 for q, y in det)} false detections, "
                f"detection AUROC {bench.fmt(bench.auroc([q for q, _ in det], [y for _, y in det]), 2)}") if det else "",
        "n": len(lab), "errors": sum(not x["label"] for x in lab),
        "checks": sum(s == x["label"] for s, x in zip(says, lab)),
        "caught": sum((not x["label"]) and not s for s, x in zip(says, lab)),
        "false_alarms": sum(x["label"] and not s for s, x in zip(says, lab)),
        "auroc": bench.auroc([x["p"] for x in lab], [x["label"] for x in lab]),
        "value_acc": (sum(x["pick"] == x["expected"] for x in picks) / len(picks)) if picks else None,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="STAGE.PARAM")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", choices=["tune", "holdout", "both"], default="both",
                    help="report the tune half, the holdout half, or both side by side (default)")
    ap.add_argument("--variant", action="append", default=[], help='"name=instructions"; omit to test the current one')
    ap.add_argument("--evidence", help="comma-separated evidence slices (default: the param's own)")
    ap.add_argument("--criteria", help='JSON {option: description} replacing the options (enum params)')
    ap.add_argument("--set", action="append", default=[], help='override a param key: key=JSON, e.g. per_note=true')
    args = ap.parse_args()
    stage, pname = args.target.split(".", 1)
    base = dict(bench.load_questions().STAGES[stage]["params"][pname])
    if args.evidence:
        base["evidence"] = args.evidence.split(",")
    for kv in args.set:
        k, v = kv.split("=", 1)
        import json as _j
        base[k] = _j.loads(v)
    if args.criteria:
        import json
        base["criteria"] = json.loads(args.criteria)
    halves = ["tune", "holdout"] if args.subset == "both" else [args.subset]
    cases = {h: bench.pick_cases(bench.load_split(args.split), args.limit, None, subset=h) for h in halves}
    variants = [v.split("=", 1) for v in args.variant] or [["current", base["instructions"]]]
    log = [f"PROBE {datetime.datetime.now().isoformat(timespec='seconds')}  {args.target}  {args.split} limit {args.limit}"
           f"  evidence {','.join(base['evidence'])}" + (f"  criteria {base['criteria']}" if args.criteria else "")
           + (f"  set {args.set}" if args.set else "")]
    for name, text in variants:
        p = dict(base, instructions=text)
        parts = []
        for h in halves:
            r = evaluate(stage, pname, p, cases[h])
            parts.append((f"{h}: {r['det']}" if r["det"] else
                          f"{h}: checks {r['checks']}/{r['n']} caught {r['caught']}/{r['errors']} false alarms "
                          f"{r['false_alarms']} AUROC {bench.fmt(r['auroc'], 2)}"))
        line = f"  {name:14} " + "   |   ".join(parts) + f"\n      Q: {text}"
        print(line)
        log.append(line)
    with open(bench.RUN_LOG, "a") as f:
        f.write("\n".join(log) + "\n\n")


if __name__ == "__main__":
    sys.exit(main())
