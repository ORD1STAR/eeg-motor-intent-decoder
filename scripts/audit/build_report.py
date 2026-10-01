import json, sys, re, html, statistics as st
import numpy as np

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
inv, segs, res = R["inventory"], R["seg_spectra"], R["results"]

CODES = {f"S{i:02d}": f"S{i:02d}" for i in range(1, 11)}   # recordings are already pseudonymised
e = html.escape


def subj(fn):
    m = re.search(r"S(?:0[1-9]|10)", fn)
    return m.group(0) if m else "?"


def tag(fn):
    m = re.match(r"^(test\d*|ft\d|\d+d?)", fn)
    return m.group(1) if m else "-"


def rid(fn):
    return f"{CODES[subj(fn)]}·{tag(fn)}"


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ numbers
wr = {r["file"]: r["bacc"] for r in res["within_random"]}
wb = {r["file"]: r for r in res["within_block"]}
dr = {r["file"]: r["bacc_push1_vs_push2"] for r in res["drift"]}
files_ml = sorted(wr, key=lambda f: (CODES[subj(f)], tag(f)))
med_wr = st.median(wr.values()); med_wb = st.median(r["bacc"] for r in wb.values())
med_dr = st.median(dr.values()); min_dr = min(dr.values())
cs = res["cross_session"]; med_cs = st.median(r["bacc"] for r in cs) if cs else float("nan")
loso = sorted(res["loso"], key=lambda r: CODES[r["subject"]])
med_loso_ts = st.median(r["bacc_ts"] for r in loso); med_loso_bp = st.median(r["bacc_bp"] for r in loso)
proto = [r for r in inv if r["scenario"] in (1, 2)]
total_h = sum(r["duration"] for r in inv) / 3600

# EC / EO alpha check (occipital alpha, dB)
def bp(fn, seg, region, band):
    for s in segs:
        if s["file"] == fn and s["seg"] == seg:
            return s["bp"][region][band]
    return None

ec_rows = []
for f in sorted({s["file"] for s in segs if s["scenario"] == 1}, key=lambda f: (CODES[subj(f)], tag(f))):
    a, b = bp(f, "neutral-EC", "occipital", "alpha"), bp(f, "neutral-EO", "occipital", "alpha")
    ec_rows.append((f, 10 * (a - b)))
s2_rows = []
for f in sorted({s["file"] for s in segs if s["scenario"] == 2}, key=lambda f: (CODES[subj(f)], tag(f))):
    a = bp(f, "push#1", "occipital", "alpha"); b = np.mean([bp(f, x, "occipital", "alpha") for x in ("push#2", "push#3")])
    s2_rows.append((f, 10 * (a - b)))

# class contrasts (task block minus rest blocks), dB, per region/band
def contrast(scn, label, region, band):
    out = []
    for f in {s["file"] for s in segs if s["scenario"] == scn}:
        task = [s["bp"][region][band] for s in segs if s["file"] == f and s["label"] == label]
        rest = [s["bp"][region][band] for s in segs if s["file"] == f and s["label"] == 0 and s["seg"] != "neutral-EC"]
        if task and rest:
            out.append(10 * (np.mean(task) - np.mean(rest)))
    return out

contr = {}
for scn in (1, 2):
    for lbl, nm in ((1, "push"), (2, "walk")):
        for region in ("central", "occipital", "frontal-polar"):
            for band in ("alpha", "beta"):
                v = contrast(scn, lbl, region, band)
                contr[(scn, nm, region, band)] = (float(np.median(v)), float(np.mean(np.array(v) < 0)))
pw = {}
for scn in (1, 2):
    for region in ("central", "occipital"):
        for band in ("alpha", "beta"):
            v = []
            for f in {s["file"] for s in segs if s["scenario"] == scn}:
                p = [s["bp"][region][band] for s in segs if s["file"] == f and s["label"] == 1]
                w = [s["bp"][region][band] for s in segs if s["file"] == f and s["label"] == 2]
                v.append(10 * (np.mean(p) - np.mean(w)))
            pw[(scn, region, band)] = float(np.median(v))

# ------------------------------------------------------------------ SVG helpers
def dot_chart():
    rows = files_ml
    W, rowh, left, right, top = 760, 22, 110, 24, 34
    H = top + rowh * len(rows) + 30
    x = lambda v: left + (W - left - right) * v
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Balanced accuracy per recording, random split vs block split" class="chart">']
    for t in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-8}" y2="{H-26}" class="grid"/>'
                 f'<text x="{x(t):.1f}" y="{H-10}" class="tick" text-anchor="middle">{int(t*100)}%</text>')
    s.append(f'<line x1="{x(1/3):.1f}" x2="{x(1/3):.1f}" y1="{top-14}" y2="{H-26}" class="chance"/>'
             f'<text x="{x(1/3)+4:.1f}" y="{top-16}" class="tick">chance 33%</text>')
    for i, f in enumerate(rows):
        y = top + i * rowh + rowh / 2
        a, b = wb[f]["bacc"], wr[f]
        s.append(f'<text x="{left-10}" y="{y+4:.1f}" class="lab" text-anchor="end">{e(rid(f))} <tspan class="muted">S{inv_sc[f]}</tspan></text>')
        s.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="link"/>')
        s.append(f'<g class="hit"><circle cx="{x(a):.1f}" cy="{y:.1f}" r="5" class="m-honest"/><title>{e(rid(f))} block split: {pct(a)}</title></g>')
        s.append(f'<g class="hit"><circle cx="{x(b):.1f}" cy="{y:.1f}" r="5" class="m-leaky"/><title>{e(rid(f))} random split: {pct(b)}</title></g>')
    s.append("</svg>")
    return "".join(s)


def loso_chart():
    W, H, left, bottom, top = 760, 260, 44, 40, 18
    n = len(loso); gw = (W - left - 10) / n; bw = gw * 0.3
    y = lambda v: top + (H - top - bottom) * (1 - v)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Leave-one-subject-out balanced accuracy" class="chart">']
    for t in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{left}" x2="{W-10}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{left-6}" y="{y(t)+4:.1f}" class="tick" text-anchor="end">{int(t*100)}%</text>')
    for i, r in enumerate(loso):
        gx = left + i * gw + gw / 2
        for j, (k, cls, nm) in enumerate((("bacc_ts", "b-ts", "Riemann TS + LR"), ("bacc_bp", "b-bp", "log band power + sLDA"))):
            bx = gx - bw - 1 + j * (bw + 2); v = r[k]
            s.append(f'<g class="hit"><path d="M{bx:.1f},{y(0):.1f} V{y(v)+3:.1f} q0,-3 3,-3 h{bw-6:.1f} q3,0 3,3 V{y(0):.1f} Z" class="{cls}"/><title>{CODES[r["subject"]]} ({r["n_files"]} files) {nm}: {pct(v)}</title></g>')
        s.append(f'<text x="{gx:.1f}" y="{H-bottom+16}" class="tick" text-anchor="middle">{CODES[r["subject"]]}</text>')
    s.append(f'<line x1="{left}" x2="{W-10}" y1="{y(1/3):.1f}" y2="{y(1/3):.1f}" class="chance"/><text x="{W-12}" y="{y(1/3)-5:.1f}" class="tick" text-anchor="end">chance 33%</text>')
    s.append(f'<line x1="{left}" x2="{W-10}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="axis"/></svg>')
    return "".join(s)


def timeline(scn):
    tt = {1: [(0, 120, "ec", "Baseline, eyes closed"), (120, 240, "n", "Baseline, eyes open"), (240, 270, "p", "Push"), (270, 290, "n", "Rest"),
              (290, 320, "p", "Push"), (320, 340, "n", "Rest"), (340, 370, "w", "Walk"), (370, 390, "n", "Rest"), (390, 420, "w", "Walk"), (420, 480, "n", "Rest")],
          2: [(0, 80, "p", "Push"), (80, 160, "w", "Walk"), (160, 180, "n", "Rest"), (180, 260, "p", "Push"), (260, 340, "w", "Walk"),
              (340, 360, "n", "Rest"), (360, 430, "p", "Push"), (430, 500, "w", "Walk")]}[scn]
    marks = {1: [0, 120, 240, 270, 290, 320, 340, 370, 390, 420], 2: [0, 80, 160, 180, 260, 340, 360, 430]}[scn]
    mlab = {1: ["EC", "EO"] + ["EO"] * 8, 2: ["EC"] + ["EO"] * 7}[scn]
    W, left, right = 760, 10, 10
    x = lambda t: left + (W - left - right) * t / 500
    s = [f'<svg viewBox="0 0 {W} 92" role="img" aria-label="Scenario {scn} timeline" class="chart">']
    for t0, t1, c, nm in tt:
        s.append(f'<g class="hit"><rect x="{x(t0)+1:.1f}" y="26" width="{x(t1)-x(t0)-2:.1f}" height="26" rx="3" class="seg-{c}"/><title>{nm}: {t0}–{t1} s</title></g>')
        if t1 - t0 >= 30:
            s.append(f'<text x="{(x(t0)+x(t1))/2:.1f}" y="43.5" class="seglab seglab-{c}" text-anchor="middle">{nm.split(",")[0] if t1-t0 < 100 else nm}</text>')
    for m, l in zip(marks, mlab):
        s.append(f'<line x1="{x(m):.1f}" x2="{x(m):.1f}" y1="12" y2="58" class="mark"/><text x="{x(m)+2:.1f}" y="10" class="tick">{l}</text>')
    for t in range(0, 501, 60 if scn == 1 else 50):
        s.append(f'<text x="{x(t):.1f}" y="74" class="tick" text-anchor="middle">{t}s</text>')
    s.append("</svg>")
    return "".join(s)


inv_sc = {r["file"]: r["scenario"] for r in inv}

# ------------------------------------------------------------------ tables
def inventory_table():
    rows = []
    for r in sorted(inv, key=lambda r: (CODES.get(subj(r["file"]), "S99"), r["scenario"] if r["scenario"] >= 0 else 9, tag(r["file"]))):
        f = r["file"]; q = r["quality"]; sc = r["scenario"]
        sc_txt = {1: "Scenario 1", 2: "Scenario 2", 0: "4×120 s (unlabelled)", -1: "Short test"}[sc]
        used = "train+test" if (sc in (1, 2) and not re.match(r"^\dd_", f) and f not in ("1_S02.edf", "2_S07.edf")) else "not used"
        notes = []
        if re.match(r"^\dd_", f): notes.append(f"byte-identical copy of {tag(f)[0]}")
        if "PAUSE" in [m[1] for m in r["markers"]]: notes.append("PAUSE marker at 148.6 s")
        if r["duration"] == 440: notes.append("ends at 440 s, last rest block cut")
        if f.startswith("test11") or f.startswith("test21"): notes.append(f"first 27 s of {f[:5]}")
        if f in ("1_S02.edf",): notes.append("valid recording never used")
        if q["noisy_channels"]: notes.append("noisy: " + ", ".join(c.replace("-TP", "") for c in q["noisy_channels"]))
        bad = q["frac_seconds_ptp_gt150"]
        rows.append(f'<tr><td class="mono">{e(rid(f))}</td><td>{sc_txt}</td><td class="num">{r["duration"]:.0f}</td>'
                    f'<td class="num">{r["meas_date"][:10]}</td><td class="num">{q["std_uV_median"]:.1f}</td>'
                    f'<td class="num {"hi" if bad > .15 else ""}">{bad*100:.0f}%</td><td><span class="chip {"chip-ok" if used != "not used" else "chip-off"}">{used}</span></td><td class="small">{e("; ".join(notes))}</td></tr>')
    return "".join(rows)


def ml_table():
    out = []
    for f in files_ml:
        cm = wb[f]["cm"]
        out.append(f'<tr><td class="mono">{e(rid(f))}</td><td>S{inv_sc[f]}</td><td class="num">{pct(wr[f])}</td><td class="num">{pct(wb[f]["bacc"])}</td><td class="num">{pct(dr[f])}</td><td class="mono small">{cm}</td></tr>')
    return "".join(out)


def cs_table():
    g = {}
    for r in cs:
        g.setdefault((r["subject"], r["scenario"]), []).append(r)
    out = []
    for (s, scn), rs in sorted(g.items(), key=lambda kv: (CODES[kv[0][0]], kv[0][1])):
        out.append(f'<tr><td class="mono">{CODES[s]}</td><td>S{scn}</td><td class="num">{len(rs)}</td><td class="num">{pct(np.mean([r["bacc"] for r in rs]))}</td><td class="mono small">{" · ".join(f"{tag(r["test"])}: {pct(r["bacc"])}" for r in rs)}</td></tr>')
    return "".join(out)


ec_med = st.median(v for _, v in ec_rows); ec_pos = sum(v > 0 for _, v in ec_rows)
s2_med = st.median(v for _, v in s2_rows); s2_pos = sum(v > 0 for _, v in s2_rows)

def fmt_db(v):
    return f"{v:+.1f} dB"

ctx = dict(
    n_files=len(inv), n_proto=len(proto), hours=f"{total_h:.1f}", med_wr=pct(med_wr), med_wb=pct(med_wb), med_dr=pct(med_dr), min_dr=pct(min_dr),
    med_cs=pct(med_cs), med_loso_ts=pct(med_loso_ts), med_loso_bp=pct(med_loso_bp), n_ml=len(files_ml),
    dot=dot_chart(), loso=loso_chart(), t1=timeline(1), t2=timeline(2), inv=inventory_table(), mlt=ml_table(), cst=cs_table(),
    ec_med=fmt_db(ec_med), ec_pos=ec_pos, ec_n=len(ec_rows), s2_med=fmt_db(s2_med), s2_pos=s2_pos, s2_n=len(s2_rows),
    c=lambda *k: fmt_db(contr[k][0]), cf=lambda *k: f"{100*contr[k][1]:.0f}%", pw=lambda *k: fmt_db(pw[k]),
)
tpl = open(sys.argv[3], encoding="utf-8").read()
out = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1), {}, ctx)), tpl)
open(OUT, "w", encoding="utf-8").write(out)
print("wrote", OUT, len(out))
print({k: ctx[k] for k in ctx if isinstance(ctx[k], (str, int)) and len(str(ctx[k])) < 40})
print({k: (round(v[0], 2), round(v[1], 2)) for k, v in contr.items()})
print({k: round(v, 2) for k, v in pw.items()})
