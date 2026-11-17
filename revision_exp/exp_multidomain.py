"""
Multi-domain validation (Table 5) + runtime comparison (E7) + rank agreement (E6).
6 datasets x 6 models; N_INST test instances each.

Outputs: results/multidomain_instances.csv (per dataset/model/instance/feature),
         results/multidomain_summary.csv   (per dataset/model)
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import split, fit_model, score, background, save, MODELS, SVM_MAX_TRAIN
import local_cbfi_v2 as V2
import xai_baselines as XB

DATASETS = ["adult", "bank", "diabetes", "california", "bike", "ames"]
N_INST, N_SAMPLES = 10, 100


def run(ds, mname):
    try:
        Xtr, Xte, ytr, yte, task = split(ds)
        t0 = time.perf_counter()
        model = fit_model(mname, task, Xtr, ytr)
        fit_s = time.perf_counter() - t0
        sc = score(model, task, Xte, yte)
        bg = background(Xtr)
        rows, t_cbfi, t_shap, t_lime = [], [], [], []
        for i in range(N_INST):
            x = Xte.iloc[i]
            t = time.perf_counter(); d = V2.decompose_instance(model, x, bg, task=task, n_samples=N_SAMPLES, rng=i).loc[x.index]; t_cbfi.append(time.perf_counter() - t)
            t = time.perf_counter(); s = XB.shap_permutation(model, task, x, bg, seed=i); t_shap.append(time.perf_counter() - t)
            t = time.perf_counter(); l = XB.lime_weights(model, task, x, Xtr, seed=i); t_lime.append(time.perf_counter() - t)
            d["SHAP"], d["LIME"], d["instance"] = s, l, i
            rows.append(d.reset_index(names="feature"))
        inst = pd.concat(rows, ignore_index=True)
        inst["dataset"], inst["model"] = ds, mname

        agr = {}
        for a, b in [("Imp", "SHAP"), ("Imp", "LIME"), ("SHAP", "LIME"), ("G1", "SHAP")]:
            vals = pd.DataFrame([XB.agreement(g[a], g[b]) for _, g in inst.groupby("instance")]).mean()
            agr.update({f"{a}~{b}_{k}": v for k, v in vals.items()})
        g = inst.groupby("instance")
        summ = {
            "dataset": ds, "model": mname, "task": task, "p": Xtr.shape[1], "n_train": min(len(Xtr), SVM_MAX_TRAIN) if mname == "SVM" else len(Xtr), "score": sc,
            "fit_s": fit_s,
            "mean_abs_G1": inst.G1.abs().mean(), "mean_abs_G4": inst.G4.abs().mean(), "mean_abs_Imp": inst.Imp.abs().mean(),
            # share of the explained effect that is interaction, per instance, then averaged
            "G4_share": (g.apply(lambda d: d.G4_abs.sum() / (d.G1.abs().sum() + d.G4_abs.sum() + 1e-12))).mean(),
            "interaction_activation": inst.share_amplified.add(inst.share_attenuated).mean(),
            "sec_cbfi": np.mean(t_cbfi), "sec_shap_perm": np.mean(t_shap), "sec_lime": np.mean(t_lime),
            "sec_treeshap": XB.shap_tree_runtime(mname, model, Xte.iloc[:N_INST], bg),
            **agr}
        print(f"done {ds:10s} {mname:6s} score={sc:.3f} cbfi={np.mean(t_cbfi):.2f}s", flush=True)
        return inst, summ
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FAILED {ds} {mname}: {e!r}", flush=True)
        return None, {"dataset": ds, "model": mname, "error": repr(e)}


if __name__ == "__main__":
    jobs = [(d, m) for d in DATASETS for m in MODELS]
    # TabNet uses the GPU: run those sequentially after the CPU jobs
    cpu = [j for j in jobs if j[1] != "TabNet"]
    gpu = [j for j in jobs if j[1] == "TabNet"]
    out = Parallel(n_jobs=10)(delayed(run)(*j) for j in cpu)
    out += [run(*j) for j in gpu]
    save(pd.concat([o[0] for o in out if o[0] is not None], ignore_index=True), "multidomain_instances.csv")
    save(pd.DataFrame([o[1] for o in out]), "multidomain_summary.csv")
