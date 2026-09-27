"""Print the baseline comparison table from the saved result JSONs."""
import os
RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import json, os, numpy as np, collections

RES = os.path.join(RESULTS, 'results_baselines.json')
TRIV = os.path.join(RESULTS, 'results_trivial.json')
res = json.load(open(RES)) if os.path.exists(RES) else {}
triv = json.load(open(TRIV))

NAME = {"deepsets": "DeepSets", "density": "Gaze-density CNN",
        "settx": "Set Transformer", "temporal": "Multi-scale temporal (+PE)",
        "temporal_shuf": r"\quad $\hookrightarrow$ frames shuffled"}

agg = collections.defaultdict(list)
for k, v in res.items():
    scene, kind, seed = k.split("|")
    agg[(scene, kind)].append(v)


def cell(scene, kind, key="Corner-Dist"):
    v = agg.get((f"scene{scene}", kind))
    if not v:
        return "--"
    a = np.array([x[key] for x in v])
    if len(a) == 1:
        return f"{a[0]:.2f}"
    return f"{a.mean():.2f}" + r"{\tiny$\pm$" + f"{a.std(ddof=1):.2f}".lstrip("0") + "}"


EOL = r"\\"
rows = [
    f"Mean-Train (trivial)       & {triv['2']['Mean-Train']['Corner-Dist']:.2f} & "
    f"{triv['3']['Mean-Train']['Corner-Dist']:.2f} {EOL}",
    f"Oracle-Const (upper bd.)   & {triv['2']['Oracle-Const']['Corner-Dist']:.2f} & "
    f"{triv['3']['Oracle-Const']['Corner-Dist']:.2f} {EOL}",
    f"Robust geometric fit       & 64.41 & 55.62 {EOL}",
    r"\hline",
]
for kind in ["deepsets", "density", "settx", "temporal", "temporal_shuf"]:
    if any(k[1] == kind for k in agg):
        rows.append(f"{NAME[kind]:26s} & {cell('2', kind)} & {cell('3', kind)} {EOL}")
rows += [r"\hline",
         r"ScreenLocNet (ours)        & \bf 17.50 & \bf 14.91 " + EOL]

print(r"{\centering\scriptsize\setlength{\tabcolsep}{4.5pt}")
print(r"\begin{tabular}{lcc}")
print(r"\hline")
print(f"Corner-Dist (cm) & Scene 2 & Scene 3 {EOL}")
print(r"\hline")
print("\n".join(rows))
print(r"\hline")
print(r"\end{tabular}")
print(r"\par}")
