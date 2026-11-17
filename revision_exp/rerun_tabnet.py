"""Re-run only the TabNet jobs (after fixing TabNet training) and replace their rows in the result CSVs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from common import RESULTS, save
import exp_multidomain as MD
import exp_stability as ST

out = [MD.run(d, "TabNet") for d in MD.DATASETS]
inst = pd.read_csv(os.path.join(RESULTS, "multidomain_instances.csv"))
summ = pd.read_csv(os.path.join(RESULTS, "multidomain_summary.csv"))
inst = pd.concat([inst[inst.model != "TabNet"]] + [o[0] for o in out if o[0] is not None], ignore_index=True)
summ = pd.concat([summ[summ.model != "TabNet"], pd.DataFrame([o[1] for o in out])], ignore_index=True)
save(inst, "multidomain_instances.csv")
save(summ, "multidomain_summary.csv")

conv = pd.read_csv(os.path.join(RESULTS, "stability_convergence.csv"))
rows = [r for d in ST.DATASETS for r in ST.convergence(d, "TabNet")]
save(pd.concat([conv[conv.model != "TabNet"], pd.DataFrame(rows)], ignore_index=True), "stability_convergence.csv")
