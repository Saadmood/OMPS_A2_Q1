"""Reproduce Question 1 from the supplied data, with one command per experiment."""
from argparse import ArgumentParser
from pathlib import Path

from src.q1_analysis import base_case, sensitivities
from src.q1_plots import input_profiles, base_dispatch, settlement_prices, contract_sweeps, cap_merit_order
from src.q1_report_tables import base_tables, sensitivity_tables


def main():
    root=Path(__file__).resolve().parent
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("experiment",nargs="?",default="all",choices=("base","sensitivity","settlement","all"))
    parser.add_argument("--out",type=Path,default=root/"results"/"q1",help="Destination for generated results")
    args=parser.parse_args(); out=args.out
    figures=out/"figures"; tables=out/"tables"
    base=base_case(out)
    base_tables(base,tables)
    input_profiles(base,figures)
    base_dispatch(base,figures)
    settlement_prices(base,figures)
    print(base.metrics[["model","cost_DKK","gain_vs_R_DKK","peak_import_kW","exceedance_kWh","hours_over_cap","manager_budget_DKK"]].round(3).to_string(index=False))
    print("All base validation checks passed. Member IR passes vs R / vs I:",base.settlement.IR_vs_R.sum(),base.settlement.IR_vs_I.sum())
    if args.experiment in ("sensitivity","all"):
        discount, caps, penalties, frontier, thresholds=sensitivities(base,out)
        contract_sweeps(discount,caps,penalties,thresholds,figures)
        cap_merit_order(caps,figures)
        sensitivity_tables(discount,caps,penalties,frontier,thresholds,tables)
        print("Break-even thresholds:",thresholds)
    print("Saved reproducible CSVs, validation checks, figures and LaTeX tables to",out)


if __name__=="__main__":
    main()
