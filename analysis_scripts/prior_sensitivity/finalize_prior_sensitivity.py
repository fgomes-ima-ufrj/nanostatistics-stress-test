#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    mpath=a.output/"tables/prior_sensitivity_metrics_raw.csv"; ppath=a.output/"tables/prior_parameter_summaries_raw.csv"
    if not mpath.exists() or not ppath.exists(): raise SystemExit("Arquivos brutos ainda não existem")
    m=pd.read_csv(mpath); expected={"nominal","narrow","broad","mildly_misspecified"}; found=set(m.prior_mode.dropna().astype(str))
    print("Prior modes encontrados:",sorted(found)); print("Ausentes:",sorted(expected-found)); print("\nContagem por prior:\n",m.groupby("prior_mode").simulation_id.nunique())
    failures=m[m.diagnostic_flag.astype(str).str.startswith("fit_failed")]; print("\nFalhas:",len(failures))
    comp=m.groupby("prior_mode").simulation_id.nunique().rename("n_simulations").reset_index(); comp.to_csv(a.output/"tables/prior_sensitivity_completeness.csv",index=False)
    print("\nArquivos finais em:",a.output/"tables")
if __name__=="__main__": main()
