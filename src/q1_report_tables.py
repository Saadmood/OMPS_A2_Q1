"""Generate the LaTeX tables and numeric macros used by the Q1 report."""
from pathlib import Path
import numpy as np


def _num(x,n=2):
    # Suppress negative zero and solver roundoff in presentation only.
    if abs(x) < .5*10**(-n): x=0.
    return f"{x:.{n}f}"


def _write(path,content):
    Path(path).write_text(content+"\n",encoding="utf-8")


def _tex_table(headers,rows,alignment):
    lines=[r"\begin{tabular}{"+alignment+"}",r"\toprule", " & ".join(headers)+r" \\",r"\midrule"]
    for row in rows:
        if row is None: lines.append(r"\midrule")
        else: lines.append(" & ".join(row)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    return "\n".join(lines)


def _macros(path,values):
    _write(path,"\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name,value in values.items()))


def base_tables(base,out):
    """Base costs, device response, member settlement, hourly duals and macros."""
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    df=base.metrics.set_index("model");s=base.settlement;v=base.validation
    mnames=[r"$\mathcal R$",r"$\mathcal I$",r"$\mathcal C$"]
    rows=[]
    for label,field,n in [
        ("Day-ahead energy, net", "energy_DKK",2),
        ("Grid tariffs", "tariffs_DKK",2),
        ("Exceedance penalty", "penalty_DKK",2),
        ("PV generation cost", "pv_cost_DKK",2),
        ("Load disutility", "disutility_DKK",2),
        (r"\textbf{Total cost $J^{\mathcal X}$}", "cost_DKK",2),
        (r"Gain relative to $\mathcal R$", "gain_vs_R_DKK",2),
    ]:
        cells=[_num(df.loc[m,field],n) for m in ("R","I","C")]
        if field=="cost_DKK": cells=[r"\textbf{"+x+"}" for x in cells]
        rows.append([label,*cells])
    _write(out/"base_cost.tex",_tex_table(["Cost or gain [DKK/day]",*mnames],rows,"lrrr"))
    rows=[]
    for label,field,n in [
        ("Gross metered import [kWh]", "daily_import_kWh",2),
        ("Gross metered export [kWh]", "daily_export_kWh",2),
        ("Peak import [kW]", "peak_import_kW",2),
        (r"\textbf{Total feeder exceedance [kWh]}", "exceedance_kWh",2),
        ("Peak exceedance [kW]", "max_exceedance_kW",2),
        ("Hours above cap [h]", "hours_over_cap",0),
        ("Hours at cap [h]", "hours_at_cap",0),
        ("Billed exceedance [kWh]", "billed_exceedance_kWh",2),
        ("Import removed in R-congested hours [kWh]", "import_removed_from_R_congested_hours_kWh",2),
        ("Internal exchange [kWh]", "internal_exchange_kWh",2),
    ]:
        cells=[_num(df.loc[m,field],n) for m in ("R","I","C")]
        if field=="exceedance_kWh": cells=[r"\textbf{"+x+"}" for x in cells]
        if field=="billed_exceedance_kWh":cells[0]="--"
        rows.append([label,*cells])
    _write(out/"base_dso.tex",_tex_table(["DSO/dispatch metric (one day)",*mnames],rows,"lrrr"))

    rows=[]
    for k,slope in enumerate((.25,.5,1.,2.),1):
        shifts=[sum(d.deviation.sum()/2 for c,d in a.members.items() if base.com.member(c).slope_class==k) for a in (base.R,base.I,base.C)]
        rows.append([f"L{k} ({slope:g})",*[_num(x) for x in shifts]])
    _write(out/"base_shifts.tex",_tex_table(["Load class ($c_i^L$)",*mnames],rows,"lrrr"))

    rows=[]
    for j,row in s.iterrows():
        if j and j%4==0:rows.append(None)
        gain=lambda x: r"\textbf{"+f"+{x:.2f}"+"}" if x>=1e-7 else _num(x)
        rows.append([row.member.replace("_",r"\_"),_num(row.R_cost_DKK),_num(row.I_cost_DKK),
                     _num(row.C_settled_cost_DKK),gain(row.gain_vs_R_DKK),gain(row.gain_vs_I_DKK)])
    rows.append(None)
    rows.append(["Total",_num(s.R_cost_DKK.sum()),_num(s.I_cost_DKK.sum()),_num(s.C_settled_cost_DKK.sum()),
                 _num(s.gain_vs_R_DKK.sum()),_num(s.gain_vs_I_DKK.sum())])
    _write(out/"member_settlement.tex",_tex_table(["Member",r"$J_i^{\mathcal R}$",r"$J_i^{\mathcal I}$",
                          r"$J_i^{\mathcal C,\mathrm{set}}$",r"$B_{i|\mathcal R}$",r"$B_{i|\mathcal I}$"],rows,"lrrrrr"))

    rows=[]
    for t in range(24):
        c=base.C;rows.append([str(t),_num(base.com.energy_price[t]),_num(c.aggregate_import[t],3),
                             _num(c.aggregate_export[t],3),_num(c.price[0,t],4),_num(c.cap_dual[0,t],4),
                             _num(c.cap_dual[0,t]*base.com.contract.capacity_limit_kW[t])])
    _write(out/"hourly_duals.tex",_tex_table(["Hour",r"$p_t$",r"$P_t^{\rm imp}$",r"$P_t^{\rm exp}$",r"$\lambda_t$",r"$\mu_t$",r"$\mu_t\bar P_t^{\rm DSO}$"],rows,"rrrrrrr"))

    costs=[base.R.total_cost,base.I.total_cost,base.C.total_cost]
    B=base.C.manager_budget; outside=s[["R_cost_DKK","I_cost_DKK"]].min(axis=1).sum();rebate=s.rebate_to_best_outside_DKK.sum()
    vals={
        "Rcost":_num(costs[0]),"Icost":_num(costs[1]),"Ccost":_num(costs[2]),
        "Cgain":_num(costs[0]-costs[2]),"Iloss":_num(costs[1]-costs[0]),
        "Cgainpct":_num(100*(costs[0]-costs[2])/costs[0],1),
        "Ilosspct":_num(100*(costs[1]-costs[0])/costs[0],1),
        "Poolinggain":_num(costs[1]-costs[2]),
        "Rex":_num(df.loc["R","exceedance_kWh"]),"Iex":_num(df.loc["I","exceedance_kWh"]),
        "Ireliefpct":_num(df.loc["I","exceedance_reduction_pct"],1),
        "Rpeak":_num(df.loc["R","peak_import_kW"]),"Ipeak":_num(df.loc["I","peak_import_kW"]),
        "Cpeak":_num(df.loc["C","peak_import_kW"]),
        "Cpeakpct":_num(100*df.loc["C","peak_reduction_kW"]/df.loc["R","peak_import_kW"],1),
        "Cexchange":_num(df.loc["C","internal_exchange_kWh"]),
        "Cbill":_num(df.loc["C","meter_payment_DKK"]),"Crent":_num(B),"Ccollections":_num(B+df.loc["C","meter_payment_DKK"]),
        "CmembersTotal":_num(s.C_settled_cost_DKK.sum()),
        "IRR":str(int(s.IR_vs_R.sum())),"IRI":str(int(s.IR_vs_I.sum())),"IRBoth":str(int((s.IR_vs_R&s.IR_vs_I).sum())),
        "OutsideTotal":_num(outside),"RebateNeed":_num(rebate),"RebateRemainder":_num(B-rebate),
        "SurplusVsBest":_num(outside-base.C.total_cost),
        "DSOreceiptsLost":_num(df.loc["R","DSO_receipts_DKK"]-df.loc["C","DSO_receipts_DKK"]),
        "DSOreceiptPerRelief":_num((df.loc["R","DSO_receipts_DKK"]-df.loc["C","DSO_receipts_DKK"])/df.loc["R","exceedance_kWh"]),
        "RtieLo":_num(v["R_optimal_face_low_exceedance_kWh"]),"RtieHi":_num(v["R_optimal_face_high_exceedance_kWh"]),
        "Rimp":_num(df.loc["R","daily_import_kWh"],3),"Cimp":_num(df.loc["C","daily_import_kWh"]),
        "Rnet":_num(df.loc["R","net_import_kWh"]),"Cnet":_num(df.loc["C","net_import_kWh"]),
        "Cpv":_num(df.loc["C","pv_generation_kWh"]),"Rpv":_num(df.loc["R","pv_generation_kWh"]),
        "Rlosses":_num(df.loc["R","battery_charge_kWh"]-df.loc["R","battery_discharge_kWh"]),
        "Closses":_num(df.loc["C","battery_charge_kWh"]-df.loc["C","battery_discharge_kWh"]),
        "Cshift":_num(df.loc["C","shifted_load_kWh"]),"Ishift":_num(df.loc["I","shifted_load_kWh"]),
        "Rshift":_num(df.loc["R","shifted_load_kWh"]),
        "UnavoidableI":_num(v["unavoidable_I_exceedance_kWh"]),
        "MaxtieMu":_num(v["cap_multiplier_DKK_per_kWh"],3),
    }
    _macros(out/"base_values.tex",vals)


def sensitivity_tables(discount,caps,penalties,frontier,thresholds,out):
    """Contract break-even table and exact threshold macros."""
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for _,r in frontier.iterrows():
        fmt=lambda v: "--" if np.isnan(v) else _num(100*v,1)+r"\%"
        rows.append([_num(r.cap_factor,2),_num(r.cap_kW,2),fmt(r.minimum_discount_I),fmt(r.minimum_discount_C)])
    _write(out/"break_even.tex",_tex_table(["Cap factor $f$","Cap [kW]",r"$\beta^*_{\mathcal I}$",r"$\beta^*_{\mathcal C}$"],rows,"rrrr"))
    d=thresholds
    get=lambda frame,mode,col,value,field: float(frame[(frame.model==mode)&np.isclose(frame[col],value)][field].iloc[0])
    vals={
        "BetastarC":_num(100*d["discount_break_even"]["C"],2),
        "FstarC":_num(d["cap_factor_break_even"]["C"],3),"FstarI":_num(d["cap_factor_break_even"]["I"],3),
        "KstarC":_num(d["cap_kW_break_even"]["C"],3),"KstarI":_num(d["cap_kW_break_even"]["I"],3),
        "AlphastarI":_num(d["penalty_cost_break_even"]["I"],3),
        "AlphasafeC":_num(d["C_exact_penalty_sufficient_DKK_per_kWh"],2),
        "ClossNoDisc":_num(-get(discount,"C","beta",0.,"gain_vs_R_DKK")),
        "IlossFullDisc":_num(-get(discount,"I","beta",1.,"gain_vs_R_DKK")),
        "CgainFullDisc":_num(get(discount,"C","beta",1.,"gain_vs_R_DKK")),
        "CtightCapLoss":_num(-get(caps,"C","cap_factor",.9,"gain_vs_R_DKK")),
        "CtightCap":_num(get(caps,"C","cap_factor",.9,"cap_kW")),
        "IPenaltyHalfGain":_num(get(penalties,"I","alpha_DKK_per_kWh",.5,"gain_vs_R_DKK")),
        "CPenaltyHalfGain":_num(get(penalties,"C","alpha_DKK_per_kWh",.5,"gain_vs_R_DKK")),
        "IPenaltyHalfEx":_num(get(penalties,"I","alpha_DKK_per_kWh",.5,"exceedance_kWh")),
        "CPenaltyHalfEx":_num(get(penalties,"C","alpha_DKK_per_kWh",.5,"exceedance_kWh")),
        "CPenaltyLowEx":_num(get(penalties,"C","alpha_DKK_per_kWh",.15,"exceedance_kWh")),
        "CBeforeExactEx":_num(d["C_exceedance_just_below_threshold_kWh"]),
    }
    _macros(out/"sensitivity_values.tex",vals)
