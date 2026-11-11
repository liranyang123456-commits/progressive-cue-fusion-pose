"""Start and end of the Adam baselines: is the proposal that Adam starts from inside the 10-degree basin,
and is its final pose? The start is the lowest-loss seed of the stage-I list written by the run; runs
whose seed list was not archived use the first logged iterate, i.e. the pose after one Adam step.

Reads run.log and the stage-I seed lists of local runs (benchmark_runs/<mesh>/<method>/), which the release does not
ship; its output is benchmark_results/adam_start_end_summary.json."""
import csv, json, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_geom import full_pose_metrics_row
R = ROOT / "benchmark_runs"
MESHES = ("bunny_100gt", "dragon_100gt", "armadillo_50gt", "happy_50gt", "lucy_50gt")
def rows(d):
    p = d / "pose_metrics_summary.csv"
    return {r["tag"]: r for r in csv.DictReader(p.open(encoding="utf-8-sig"))} if p.exists() else {}
def err(r, d, e, a):
    base = {"tag": r["tag"], "gt_dist": float(r["gt_dist"]), "gt_elev_deg": float(r["gt_elev_deg"]), "gt_azim_deg": float(r["gt_azim_deg"]),
            "est_dist": float(d), "est_elev_deg": float(e), "est_azim_deg": float(a)}
    return float(full_pose_metrics_row(base)["rot_err_deg"])
pat = re.compile(r"^\[Opt\] (gt_\d+)_refine iter 0/\d+ .*dist/elev/azim=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
out = {}
for m in ("rand_restart_adam", "grid_adam"):
    T = dict(n=0, start_ok=0, end_ok=0, lost=0, gained=0)
    gap = []
    for mesh in MESHES:
        d = R / mesh / m
        rr = rows(d)
        first = {}
        for line in (d / "run.log").open(encoding="utf-8", errors="replace"):
            mm = pat.match(line)
            if mm and mm.group(1) not in first: first[mm.group(1)] = tuple(float(mm.group(i)) for i in (2, 3, 4))
        for tag, r in rr.items():
            js = d / tag / "seed_00" / f"{tag}_rand_octant_best.json"
            e1 = float(r["rot_err_deg"])
            if js.exists():
                seeds = json.loads(js.read_text(encoding="utf-8"))["topk_seeds"]
                s = min(seeds, key=lambda x: float(x["loss_total"]))
                e0 = err(r, s["dist"], s["elev_deg"], s["azim_deg"])
                if tag in first: gap.append(abs(err(r, *first[tag]) - e0))
            elif tag in first:
                e0 = err(r, *first[tag]); T["from_first_iterate"] = T.get("from_first_iterate", 0) + 1
            else:
                continue
            T["n"] += 1; T["start_ok"] += e0 <= 10; T["end_ok"] += e1 <= 10; T["lost"] += (e0 <= 10 < e1); T["gained"] += (e1 <= 10 < e0)
    gap.sort()
    print(m, T, "| |err(first iterate)-err(start)| median %.2f p95 %.2f deg" % (gap[len(gap)//2], gap[int(0.95*len(gap))]))
    out[m] = T
topk = {mesh: rows(R / mesh / "rand_topk_only") for mesh in MESHES}
n = sum(len(v) for v in topk.values()); ok = sum(float(r["rot_err_deg"]) <= 10 for v in topk.values() for r in v.values())
out["rand_topk_only"] = {"n": n, "end_ok": ok}
print("rand_topk_only", out["rand_topk_only"])
(ROOT / "benchmark_results" / "adam_start_end_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
