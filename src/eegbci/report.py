"""Build the results page from eegbci/results/*. python -m eegbci.report"""
import base64
import html
import json
import re

import numpy as np

from . import config as C

R = C.RESULTS_DIR
e = html.escape


def log_lookup():
    rows = [json.loads(l) for l in open(R / "explore_log.jsonl")]
    def get(name, scheme, split, key=None):
        for d in reversed(rows):
            if d["name"] == name and d["scheme"] == scheme and d["split"] == split and (key is None or d["key"] == key):
                return d["summary"]
        raise KeyError((name, scheme, split))
    return get


def line_chart(curves, band):
    W, H, l, r, t, b = 720, 260, 48, 16, 16, 34
    a, bb = curves[f"{band}|rest->task"], curves[f"{band}|task->rest"]
    rel = np.array(a["rel"])
    allv = np.concatenate([np.array(c["mean"]) + s * np.array(c["sem"]) for c in (a, bb) for s in (-1, 1)])
    lo, hi = np.floor(allv.min() * 4) / 4, np.ceil(allv.max() * 4) / 4
    x = lambda v: l + (W - l - r) * (v - rel[0]) / (rel[-1] - rel[0])
    y = lambda v: t + (H - t - b) * (hi - v) / (hi - lo)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Parieto-occipital {band} power around block changes">']
    for v in np.arange(lo, hi + 1e-9, 0.25):
        s.append(f'<line x1="{l}" x2="{W-r}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"axis" if abs(v) < 1e-9 else "grid"}"/>'
                 f'<text x="{l-6}" y="{y(v)+4:.1f}" class="tick" text-anchor="end">{v:+.2f}</text>')
    for v in range(-12, 21, 4):
        s.append(f'<text x="{x(v):.1f}" y="{H-12}" class="tick" text-anchor="middle">{v:+d}s</text>')
    s.append(f'<rect x="{x(-12):.1f}" y="{t}" width="{x(-2)-x(-12):.1f}" height="{H-t-b}" class="base-band"/>')
    s.append(f'<rect x="{x(3):.1f}" y="{t}" width="{x(15)-x(3):.1f}" height="{H-t-b}" class="post-band"/>')
    s.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{t}" y2="{H-b}" class="chance"/>'
             f'<text x="{x(0)+4:.1f}" y="{t+12}" class="tick">block change</text>')
    for c, cls, nm in ((a, "s1", "rest → task"), (bb, "s2", "task → rest")):
        m, se = np.array(c["mean"]), np.array(c["sem"])
        up = " ".join(f"{x(rv):.1f},{y(v):.1f}" for rv, v in zip(rel, m + se))
        dn = " ".join(f"{x(rv):.1f},{y(v):.1f}" for rv, v in zip(rel[::-1], (m - se)[::-1]))
        s.append(f'<polygon points="{up} {dn}" class="{cls}-band"/>')
        s.append(f'<polyline points="{" ".join(f"{x(rv):.1f},{y(v):.1f}" for rv, v in zip(rel, m))}" class="{cls}-line"/>')
        s.append(f'<text x="{x(rel[-1])-2:.1f}" y="{y(m[-8:].mean())-8:.1f}" class="lab" text-anchor="end">{nm}</text>')
    s.append("</svg>")
    return "".join(s)


def bars(items, chance=0.5, null_band=None, W=720):
    """items: [(label, value, cls, note)] horizontal bars on 0.4..0.8 AUC axis."""
    rowh, l, r, t = 30, 250, 60, 26
    H = t + rowh * len(items) + 30
    lo, hi = 0.4, 0.8
    x = lambda v: l + (W - l - r) * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="AUC by model">']
    for v in np.arange(lo, hi + 1e-9, 0.1):
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{t-6}" y2="{H-26}" class="grid"/><text x="{x(v):.1f}" y="{H-10}" class="tick" text-anchor="middle">{v:.1f}</text>')
    if null_band:
        s.append(f'<rect x="{x(null_band[0]):.1f}" y="{t-6}" width="{x(null_band[1])-x(null_band[0]):.1f}" height="{H-t-20}" class="null-band"/>'
                 f'<text x="{x(null_band[1])+4:.1f}" y="{t-10}" class="tick">drift-null range</text>')
    s.append(f'<line x1="{x(chance):.1f}" x2="{x(chance):.1f}" y1="{t-6}" y2="{H-26}" class="chance"/>')
    for i, (lab, v, cls, note) in enumerate(items):
        yy = t + i * rowh
        s.append(f'<text x="{l-10}" y="{yy+18}" class="lab" text-anchor="end">{e(lab)}</text>')
        s.append(f'<g class="hit"><path d="M{x(lo):.1f},{yy+6} H{x(v)-4:.1f} q4,0 4,4 v10 q0,4 -4,4 H{x(lo):.1f} Z" class="{cls}"/><title>{e(lab)}: {v:.3f} {e(note)}</title></g>')
        s.append(f'<text x="{x(v)+6:.1f}" y="{yy+18}" class="val">{v:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


def subject_dots(ens):
    subs = sorted({k.split("-")[0] for k in ens})
    rows = {n: [np.mean([ens[r][n]["auc"] for r in ens if r.startswith(s)]) for s in subs] for n in ("riemann", "shallow", "ensemble")}
    W, l, r, t, rowh = 720, 60, 20, 24, 26
    H = t + rowh * len(subs) + 30
    lo, hi = 0.3, 1.0
    x = lambda v: l + (W - l - r) * (v - lo) / (hi - lo)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="AUC per participant">']
    for v in np.arange(lo, hi + 1e-9, 0.1):
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{t-8}" y2="{H-26}" class="grid"/><text x="{x(v):.1f}" y="{H-10}" class="tick" text-anchor="middle">{v:.1f}</text>')
    s.append(f'<line x1="{x(.5):.1f}" x2="{x(.5):.1f}" y1="{t-8}" y2="{H-26}" class="chance"/><text x="{x(.5)+4:.1f}" y="{t-12}" class="tick">chance</text>')
    for i, sub in enumerate(subs):
        yy = t + i * rowh + 8
        vals = [rows[n][i] for n in rows]
        s.append(f'<text x="{l-10}" y="{yy+4}" class="lab" text-anchor="end">{sub}</text>')
        s.append(f'<line x1="{x(min(vals)):.1f}" x2="{x(max(vals)):.1f}" y1="{yy}" y2="{yy}" class="link"/>')
        for n, cls in (("riemann", "m1"), ("shallow", "m3"), ("ensemble", "m2")):
            v = rows[n][i]
            s.append(f'<g class="hit"><circle cx="{x(v):.1f}" cy="{yy}" r="{6 if n == "ensemble" else 4.5}" class="{cls}"/><title>{sub} {n}: {v:.3f}</title></g>')
    s.append("</svg>")
    return "".join(s), rows, subs


def main():
    get = log_lookup()
    curves = json.load(open(R / "transition_curves.json"))
    ens = json.load(open(R / "ensemble_loso.json"))
    nested = json.load(open(R / "nested_loso.json"))
    ersp = json.load(open(R / "ersp.json"))
    topo = base64.b64encode(open(R / "fig_topomaps.png", "rb").read()).decode()
    nulls = []
    for f in sorted(R.glob("ensemble_loso_null*.json")):
        d = json.load(open(f)); nulls.append(np.mean([v["ensemble"]["auc"] for v in d.values()]))
    replay = json.load(open(R / "replay_S05.json")) if (R / "replay_S05.json").exists() else []
    calib = json.load(open(R / "calibration.json"))

    def agg(n, k="auc"):
        return float(np.mean([v[n][k] for v in ens.values()]))
    nested_auc = float(np.mean([a for d in nested.values() for a in d["test"].values()]))
    nested_subj = [float(np.mean(list(d["test"].values()))) for d in nested.values()]
    dots, subj_rows, subs = subject_dots(ens)

    r1 = {k: get(n, s, "within") for k, (n, s) in {
        "cmd3": ("fb-ts align=rec", "cmd3"), "tr": ("fb-ts align=rec", "task_rest"), "pw": ("fb-ts align=rec", "push_walk"),
        "t_cmd3": ("time-only", "cmd3"), "t_tr": ("time-only", "task_rest"), "t_pw": ("time-only", "push_walk")}.items()}
    conf = {k: get(*v) for k, v in {
        "within": ("CONFIRM logbp all fb causal60", "task_rest", "within"),
        "xsess": ("CONFIRM logbp all fb causal60 smooth=8s", "task_rest", "xsess"),
        "loso": ("CONFIRM ts PO a+b causal60 smooth=8s", "task_rest", "loso")}.items()}
    prog = [("Tangent space, all channels, offline align", get("ts mi all-ch smooth=0s", "task_rest", "loso")["mean"], "b0", "round 2"),
            ("+ causal re-centering (60 s)", get("logbp all-ch fb causal-align60 smooth=0s", "task_rest", "loso")["mean"], "b0", "log-power, round 2"),
            ("+ 8 s causal smoothing", get("logbp all-ch fb causal-align60 smooth=4s", "task_rest", "loso")["mean"], "b0", "4 s smoothing shown"),
            ("Parieto-occipital tangent space", conf["loso"]["mean"], "b1", "round 3, null-confirmed"),
            ("Fusion, 4 s windows", get("fusion logbp-all+ts-PO win4 lr0.01 smooth=8s", "task_rest", "loso")["mean"], "b1", "round 4"),
            ("Nested selection (unbiased)", nested_auc, "b1", "config chosen on 9 people"),
            ("EEGNet (fixed input)", 0.631, "b3", "deep"),
            ("ShallowConvNet", agg("shallow"), "b3", "deep"),
            ("Ensemble Riemann + ShallowConvNet", agg("ensemble"), "b2", "equal weights, fixed a priori")]
    null_band = (min(nulls), max(nulls)) if nulls else (conf["loso"]["null_mean"] - 2 * conf["loso"]["null_std"],
                                                      conf["loso"]["null_mean"] + 2 * conf["loso"]["null_std"])
    ctx = dict(
        topo=topo, curve_beta=line_chart(curves, "beta"), curve_alpha=line_chart(curves, "alpha"), prog=bars(prog, null_band=null_band),
        dots=dots, ens_auc=f"{agg('ensemble'):.3f}", ens_bacc=f"{100*agg('ensemble','bacc_adaptive'):.0f}%",
        ens_bacc_fixed=f"{100*agg('ensemble','bacc_fixed'):.0f}%",
        n_subj_ok=sum(np.mean([ens[r]['ensemble']['auc'] for r in ens if r.startswith(s)]) > .5 for s in subs),
        nested_auc=f"{nested_auc:.3f}", nested_subj=f"{np.mean(nested_subj):.3f}", nested_ok=sum(v > .5 for v in nested_subj),
        xsess=f"{get('fusion logbp-all+ts-PO win4 lr0.01 smooth=8s', 'task_rest', 'xsess')['mean']:.3f}",
        r1=r1, conf=conf, calib=calib, nulls=", ".join(f"{v:.3f}" for v in nulls) if nulls else "pending",
        null_mean=f"{np.mean(nulls):.3f}" if nulls else "n/a",
        beta_on=ersp["beta|rest->task"], beta_off=ersp["beta|task->rest"],
        replay_rows="".join(f"<tr><td class='mono'>{e(d['rec'])}</td><td class='num'>{d['auc']:.3f}</td><td class='num'>{d['state_acc']:.2f}</td>"
                            f"<td class='num'>{d['false_on']:.2f}</td><td class='num'>{d['detected']}/{d['onsets']}</td>"
                            f"<td class='num'>{d['latency']:.1f} s</td><td class='num'>{d['ms']:.1f} ms</td><td class='num'>{d['parity']:.0e}</td></tr>"
                            for d in replay),
        subj_table="".join(f"<tr><td class='mono'>{s}</td>" + "".join(f"<td class='num'>{subj_rows[n][i]:.3f}</td>" for n in ("riemann", "shallow", "ensemble"))
                           + f"<td class='num'>{nested[s]['selected']}</td></tr>" for i, s in enumerate(subs)),
    )
    style = re.search(r"<style>.*?</style>", open(C.ASSETS_DIR / "style.html", encoding="utf-8").read(), re.S).group(0)
    tpl = open(C.ASSETS_DIR / "report_template.html", encoding="utf-8").read().replace("{{STYLE}}", style)
    out = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1), {"np": np}, ctx)), tpl)
    (R / "brain_switch_report.html").write_text(out, encoding="utf-8")
    print("wrote", R / "brain_switch_report.html", len(out))


if __name__ == "__main__":
    main()
