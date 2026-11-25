"""Experiments and independent checks for Question 1 (all values from solves)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data_loader import CommunityData, load_community
from .q1_model import EnergyCommunityLP, Solution, TOL


MODES = ("R", "I", "C")


def shifted_energy(dispatch):
    """kWh moved (one half of absolute deviation, since daily demand is fixed)."""
    return float(dispatch.deviation.sum() / 2)


def dso_exceedance(solution, cap):
    """Physical/DSO profile: sum of metered gross imports, not sum of billed e."""
    return np.maximum(solution.aggregate_import - cap, 0.)


def summary(com: CommunityData, s: Solution, cap, reference: Solution) -> dict:
    """Cost and DSO-service metrics on a common day and the given comparison cap."""
    imp, ex, ref_ex = s.aggregate_import, dso_exceedance(s, cap), dso_exceedance(reference, cap)
    peak, peak_ref = imp.max(), reference.aggregate_import.max()
    congestion_hours = ref_ex > TOL
    relief = float(ref_ex.sum() - ex.sum())
    return {
        "model": s.mode,
        "cost_DKK": s.total_cost,
        **{k + "_DKK": v for k, v in s.components.items()},
        "gain_vs_R_DKK": reference.total_cost - s.total_cost,
        "daily_import_kWh": float(imp.sum()),
        "daily_export_kWh": float(s.aggregate_export.sum()),
        "net_import_kWh": float((imp - s.aggregate_export).sum()),
        "peak_import_kW": float(peak),
        "peak_reduction_kW": float(peak_ref - peak),
        "exceedance_kWh": float(ex.sum()),
        "exceedance_reduction_kWh": relief,
        "exceedance_reduction_pct": 100 * relief / ref_ex.sum() if ref_ex.sum() > TOL else np.nan,
        "max_exceedance_kW": float(ex.max()),
        "hours_over_cap": int(np.sum(ex > TOL)),
        "hours_at_cap": int(np.sum(np.abs(imp - cap) <= TOL)),
        "billed_exceedance_kWh": float(s.excess.sum()),
        "import_removed_from_R_congested_hours_kWh": float(np.sum((reference.aggregate_import - imp)[congestion_hours])),
        "daily_load_kWh": sum(float(d.load.sum()) for d in s.members.values()),
        "shifted_load_kWh": sum(shifted_energy(d) for d in s.members.values()),
        "pv_generation_kWh": sum(float(d.pv.sum()) for d in s.members.values()),
        "pv_curtailed_kWh": sum(float(com.member(k).pv_available.sum() - d.pv.sum()) for k, d in s.members.items()),
        "battery_charge_kWh": sum(float(d.charge.sum()) for d in s.members.values()),
        "battery_discharge_kWh": sum(float(d.discharge.sum()) for d in s.members.values()),
        "internal_exchange_kWh": float(s.internal_exchange.sum()),
        "DSO_receipts_DKK": s.components["tariffs"] + s.components["penalty"],
        "meter_payment_DKK": float(s.meter_bill.sum()),
        "manager_budget_DKK": s.manager_budget,
        "max_primal_dual_residual": max(s.checks.values()),
    }


def individual_lower_bound(com: CommunityData, individual: Solution) -> float:
    """Unavoidable excess from members without storage, regardless of penalty.

    Since their import is at least max(0, l_min - g_available), summing the
    positive differences from their own cap is a valid lower bound.  Storage
    owners are omitted, so the bound does not assume a feasible battery state.
    """
    total = 0.
    for j, cid in enumerate(individual.member_ids):
        m = com.member(cid)
        if not m.has_battery:
            total += np.maximum(m.load_lower_kWh - m.pv_available - individual.meter_cap[j], 0.).sum()
    return float(total)


def _member_table(com, R, I, C):
    costs = {m: s.individual_cost for m, s in zip(MODES, (R, I, C))}
    rows = []
    for j, cid in enumerate(com.member_ids):
        m = com.member(cid)
        r, i, c = (costs[k][cid] for k in MODES)
        rows.append(dict(member=cid, slope=m.slope_class,
                         asset=("PV+BAT" if m.has_pv and m.has_battery else "PV" if m.has_pv else
                                "BAT" if m.has_battery else "No assets"),
                         slope_DKK_per_kWh=m.linear_disutility,
                         individual_cap_kW=I.meter_cap[j, 0],
                         net_R_kWh=R.members[cid].q.sum(),
                         R_cost_DKK=r, I_cost_DKK=i, C_settled_cost_DKK=c,
                         gain_vs_R_DKK=r-c, gain_vs_I_DKK=i-c,
                         IR_vs_R=bool(r-c >= -TOL), IR_vs_I=bool(i-c >= -TOL),
                         C_payment_DKK=float(C.price[0] @ C.members[cid].q),
                         C_local_cost_DKK=C.members[cid].local_cost,
                         rebate_to_best_outside_DKK=max(c-min(r,i), 0.),
                         C_shift_kWh=shifted_energy(C.members[cid]),
                         C_daily_energy_dual_DKK_per_kWh=C.members[cid].energy_dual,
                         I_billed_exceedance_kWh=I.excess[j].sum()))
    return pd.DataFrame(rows)


def _hourly_table(com, sols):
    rows = []
    p, cap = com.energy_price, com.contract.capacity_limit_kW
    for s in sols:
        for t in range(len(p)):
            imp, exp = s.aggregate_import[t], s.aggregate_export[t]
            rows.append(dict(model=s.mode, hour=t, price_DKK_per_kWh=p[t], cap_kW=cap[t],
                             import_kW=imp, export_kW=exp, exceedance_kW=max(0.,imp-cap[t]),
                             billed_exceedance_kW=s.excess[:,t].sum(),
                             internal_price_DKK_per_kWh=s.price[0,t] if s.mode=="C" else np.nan,
                             cap_multiplier_DKK_per_kWh=s.cap_dual[0,t] if s.mode=="C" else np.nan,
                             load_kW=sum(d.load[t] for d in s.members.values()),
                             pv_kW=sum(d.pv[t] for d in s.members.values()),
                             battery_charge_kW=sum(d.charge[t] for d in s.members.values()),
                             battery_discharge_kW=sum(d.discharge[t] for d in s.members.values()),
                             internal_exchange_kW=s.internal_exchange[t]))
    return pd.DataFrame(rows)


def _dispatch_table(com, sols):
    rows = []
    for s in sols:
        for j, cid in enumerate(com.member_ids):
            d, m = s.members[cid], com.member(cid)
            k = 0 if s.mode == "C" else j
            for t in m.hours:
                rows.append(dict(model=s.mode, member=cid, slope=m.slope_class, archetype=m.archetype,
                                 hour=t, reference_load_kW=m.reference_load[t], load_kW=d.load[t],
                                 lower_load_kW=m.load_lower_kWh[t], upper_load_kW=m.load_upper_kWh[t],
                                 pv_available_kW=m.pv_available[t], pv_kW=d.pv[t],
                                 charge_kW=d.charge[t], discharge_kW=d.discharge[t], soc_kWh=d.soc[t],
                                 net_demand_kW=d.q[t], effective_price_DKK_per_kWh=s.price[k,t],
                                 cap_multiplier_DKK_per_kWh=s.cap_dual[k,t],
                                 daily_energy_dual_DKK_per_kWh=d.energy_dual))
    return pd.DataFrame(rows)


def _shifts(com, sol):
    return {f"L{k}_shift_kWh": sum(shifted_energy(sol.members[m.consumer_id]) for m in com.members
                                 if m.slope_class == k) for k in (1,2,3,4)}


def validate(com, model, R, I, C):
    """Check official references, decomposability, KKT/rent and price response."""
    cap = com.contract.capacity_limit_kW
    ref = com.contract.reference_model_R
    r = summary(com, R, cap, R)
    # All rounded input references except exceedance are reproduced to their
    # displayed precision. Exceedance can vary across equal-cost dispatches.
    assert abs(r["cost_DKK"] - ref["cost_DKK"]) < 0.005 + TOL
    assert abs(r["daily_import_kWh"] - ref["daily_import_kWh"]) < 0.05 + TOL
    assert abs(r["peak_import_kW"] - ref["peak_import_kW"]) < 0.01
    assert r["hours_over_cap"] == 9 and summary(com, C, cap, R)["hours_at_cap"] == 18
    lo, hi = [model.solve("R", reference_tie=d) for d in ("min", "max")]
    tie_low, tie_high = (dso_exceedance(s, cap).sum() for s in (lo, hi))
    assert tie_low - TOL <= ref["exceedance_kWh"] <= tie_high + TOL
    net_shares = np.maximum(R.individual_net.sum(axis=1), 0.)
    net_file = np.array([com.contract.net_import_in_R_kWh[cid] for cid in com.member_ids])
    assert np.max(np.abs(net_shares - net_file)) < 0.005 + TOL
    assert abs(I.meter_cap.sum(axis=0) - cap).max() < TOL

    # The requested C limit case: no contract, full tariffs, each member's
    # own flows are charged (not a pooled meter at full tariffs).
    limit = model.solve("C", use_contract=False, individual_metering=True)
    assert abs(limit.total_cost - R.total_cost) < TOL
    separate = {}
    for name, sol in (("R", R), ("I", I)):
        # This is an implementation check of separability, not a new model.
        cost = sum(model.solve(name, member_ids=[cid]).total_cost for cid in com.member_ids)
        separate[name] = cost
        assert abs(cost - sol.total_cost) < TOL
    # Large cap/no discount also gives R for individual contracts.
    slack = model.solve("I", beta=0., cap_scale=100.)
    assert abs(slack.total_cost - R.total_cost) < TOL and slack.excess.sum() < TOL
    assert C.total_cost <= I.total_cost + TOL
    assert abs(sum(C.individual_cost.values()) - C.total_cost - C.manager_budget) < TOL
    # Independent LP algorithm: an interior-point solve with crossover must
    # give the same primary optimum (primal schedules can differ at ties).
    alternative = EnergyCommunityLP(com, method="highs-ipm")
    algorithm_cost_gaps = {s.mode: abs(alternative.solve(s.mode).total_cost-s.total_cost)
                           for s in (R,I,C)}
    assert max(algorithm_cost_gaps.values()) < TOL

    lower = individual_lower_bound(com, I)
    assert abs(lower - I.excess.sum()) < TOL

    # Supporting prices: solve each member at lambda, without its old
    # tariffs or a private cap. The central schedule must already minimize
    # this local objective, although the exact primal response can be tied.
    at_price = deepcopy(com)
    for m in at_price.members:
        m.energy_price = C.price[0].copy()
        m.import_tariff = m.export_tariff = 0.
    local = EnergyCommunityLP(at_price, method=model.method).solve("R")
    price_support_gap = max(abs(local.individual_cost[cid] - C.individual_cost[cid]) for cid in com.member_ids)
    assert price_support_gap < TOL

    # Validate the sign and economic interpretation of the cap multiplier
    # using a one-sided 0.001 kW relaxation of one hourly constraint.
    t = int(np.argmax(C.cap_dual[0]))
    eps = 0.001
    perturbed = cap.copy(); perturbed[t] += eps
    relaxed = model.solve("C", cap_profile=perturbed)
    fd = (C.total_cost - relaxed.total_cost) / eps
    assert abs(fd - C.cap_dual[0,t]) < 1e-5

    return dict(official_reference=ref, computed_reference=r,
                R_optimal_face_low_exceedance_kWh=float(tie_low), R_optimal_face_high_exceedance_kWh=float(tie_high),
                R_optimal_face_costs_DKK=[lo.total_cost, hi.total_cost],
                max_reference_net_rounding_kWh=float(np.max(np.abs(net_shares - net_file))),
                raw_share_sum=sum(com.contract.share.values()), C_limit_cost_DKK=limit.total_cost,
                I_slack_no_discount_cost_DKK=slack.total_cost, independent_costs_DKK=separate,
                alternative_algorithm_cost_gaps_DKK=algorithm_cost_gaps,
                unavoidable_I_exceedance_kWh=lower, price_support_max_cost_gap_DKK=price_support_gap,
                cap_relaxation_hour=t, cap_relaxation_kW=eps, cap_relaxation_saving_per_kW_DKK=fd,
                cap_multiplier_DKK_per_kWh=float(C.cap_dual[0,t]),
                KKT_checks={s.mode: s.checks for s in (R,I,C)})


@dataclass
class BaseCase:
    com: CommunityData
    model: EnergyCommunityLP
    R: Solution
    I: Solution
    C: Solution
    metrics: pd.DataFrame
    settlement: pd.DataFrame
    hourly: pd.DataFrame
    dispatch: pd.DataFrame
    validation: dict


def base_case(out: Path) -> BaseCase:
    """Run the supplied d1 case, validate and save all primal/dual outputs."""
    out.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")
    model = EnergyCommunityLP(com)
    R, I, C = [model.solve(mode) for mode in MODES]
    sols = (R, I, C)
    check = validate(com, model, R, I, C)
    metrics = pd.DataFrame([summary(com,s,com.contract.capacity_limit_kW,R) for s in sols])
    members = _member_table(com,R,I,C)
    hourly = _hourly_table(com,sols)
    dispatch = _dispatch_table(com,sols)
    metrics.to_csv(out / "base_metrics.csv", index=False)
    members.to_csv(out / "member_settlement.csv", index=False)
    hourly.to_csv(out / "base_hourly.csv", index=False)
    dispatch.to_csv(out / "member_dispatch.csv", index=False)
    pd.DataFrame([dict(model=s.mode,**_shifts(com,s)) for s in sols]).to_csv(out/"base_shift_by_slope.csv",index=False)
    members.groupby("asset",sort=False)[["gain_vs_R_DKK","gain_vs_I_DKK","IR_vs_R","IR_vs_I"]].agg(
        {"gain_vs_R_DKK":"mean","gain_vs_I_DKK":"mean","IR_vs_R":"sum","IR_vs_I":"sum"}).to_csv(out/"settlement_by_asset.csv")
    members.groupby("slope",sort=True)[["gain_vs_R_DKK","gain_vs_I_DKK","IR_vs_R","IR_vs_I"]].agg(
        {"gain_vs_R_DKK":"mean","gain_vs_I_DKK":"mean","IR_vs_R":"sum","IR_vs_I":"sum"}).to_csv(out/"settlement_by_slope.csv")
    m = com.members[0]
    pv = next(m for m in com.members if m.has_pv)
    pd.DataFrame(dict(hour=m.hours,price_DKK_per_kWh=m.energy_price,
                      reference_load_kW=m.reference_load,pv_available_kW=pv.pv_available,
                      cap_kW=com.contract.capacity_limit_kW)).to_csv(out/"input_profiles.csv",index=False)
    (out/"validation.json").write_text(json.dumps(check,indent=2,allow_nan=False),encoding="utf-8")
    return BaseCase(com,model,R,I,C,metrics,members,hourly,dispatch,check)


# Predefined one-factor sweeps, anchored in the official base-case numbers.
# Discount: full allowed interval. Cap: below daily energy to beyond R peak.
# Penalty: from slightly above the discounted tariff (0.125) to above base 10.
DISCOUNTS = np.linspace(0.,1.,21)
CAP_FACTORS = np.unique(np.round(np.r_[np.arange(.60,3.851,.05),1.3,3.85],8))
PENALTIES = np.array([.15,.20,.25,.30,.35,.40,.45,.5,.6,.75,1.,1.25,1.5,1.75,2.,2.5,
                      3.,3.5,4.,4.2,4.31,4.5,5.,6.,8.,10.,12.])


def sensitivities(base: BaseCase, out: Path):
    """Compare welfare and DSO service while varying one contract term at a time.

    Hypotheses (specified before the sweeps): greater discount lowers cost;
    greater cap lowers cost at the expense of a weaker constraint; greater
    penalty lowers billed excess but can destroy the value of individual
    contracts. Coupled metering weakly dominates separate metering in cost.
    """
    com, mo, R = base.com, base.model, base.R
    contract = com.contract
    b0, a0, f0 = contract.tariff_discount_ratio, contract.penalty_DKK_per_kWh, contract.capacity_limit_daily_energy_factor
    memo = {}

    def run(mode, beta=b0, alpha=a0, f=f0):
        if mode == "R":
            return R
        key = (mode,round(float(beta),12),round(float(alpha),12),round(float(f),12))
        if key not in memo:
            memo[key] = mo.solve(mode,beta=key[1],alpha=key[2],cap_scale=key[3]/f0)
        return memo[key]

    def point(study, mode, beta, alpha, f):
        s = run(mode,beta,alpha,f)
        cap = contract.capacity_limit_kW*f/f0
        return dict(study=study,beta=beta,alpha_DKK_per_kWh=alpha,cap_factor=f,cap_kW=cap[0],
                    **summary(com,s,cap,R),**_shifts(com,s))

    def sweep(study, values):
        rows = []
        for val in values:
            beta, alpha, f = (val,a0,f0) if study=="discount" else (b0,val,f0) if study=="penalty" else (b0,a0,val)
            local = [point(study,mode,beta,alpha,f) for mode in MODES]
            assert local[2]["cost_DKK"] <= local[1]["cost_DKK"] + TOL
            rows.extend(local)
        return pd.DataFrame(rows)

    # The sign change of cost gain is continuous; bracketed root solves use
    # the original LP rather than interpolating a plotted, coarse grid.
    def gain(m,beta=b0,alpha=a0,f=f0):
        return R.total_cost-run(m,beta,alpha,f).total_cost

    def first_beneficial(m,axis,lo,hi,beta=b0,alpha=a0,f=f0):
        def fun(x):
            return gain(m,beta=x if axis=="discount" else beta,
                        alpha=x if axis=="penalty" else alpha,f=x if axis=="cap" else f)
        l,h=fun(lo),fun(hi)
        if l >= -TOL and h >= -TOL:
            return float(lo) if axis != "penalty" else None
        if l < -TOL and h < -TOL:
            return None
        return float(brentq(fun,lo,hi,xtol=1e-7))

    beta_star={m:first_beneficial(m,"discount",0.,1.) for m in ("I","C")}
    f_star={m:first_beneficial(m,"cap",float(CAP_FACTORS.min()),float(CAP_FACTORS.max())) for m in ("I","C")}
    alpha_star={m:first_beneficial(m,"penalty",float(PENALTIES.min()),float(PENALTIES.max())) for m in ("I","C")}

    # Include the precise crossings in the saved sweeps and the plots.
    grid = lambda arr,roots: np.unique(np.r_[arr,[v for v in roots.values() if v is not None]])
    discount = sweep("discount",grid(DISCOUNTS,beta_star))
    caps = sweep("cap",grid(CAP_FACTORS,f_star))
    penalties = sweep("penalty",grid(PENALTIES,alpha_star))
    discount.to_csv(out/"discount_sweep.csv",index=False)
    caps.to_csv(out/"cap_sweep.csv",index=False)
    penalties.to_csv(out/"penalty_sweep.csv",index=False)

    # Check monotonicity of optimal values and (billed) excess as required by
    # parametric LP optimality. Physical exceedance for I is not the bill.
    for m in ("I","C"):
        v=discount[discount.model==m].sort_values("beta")
        assert np.max(np.diff(v.cost_DKK)) <= 1e-5
        v=caps[caps.model==m].sort_values("cap_factor")
        assert np.max(np.diff(v.cost_DKK)) <= 1e-5
        v=penalties[penalties.model==m].sort_values("alpha_DKK_per_kWh")
        assert np.min(np.diff(v.cost_DKK)) >= -1e-5
        assert np.max(np.diff(v.billed_exceedance_kWh)) <= 1e-5

    # Discount x cap interaction is intentional: a break-even curve/table
    # answers what combination suffices; alpha stays fixed at its base value.
    frontier=[]
    for f in (1.,1.15,1.3,1.5,1.75,2.,2.5,3.):
        row=dict(cap_factor=f,cap_kW=float(contract.capacity_limit_kW[0]*f/f0))
        for m in ("I","C"):
            row[f"minimum_discount_{m}"]=first_beneficial(m,"discount",0.,1.,f=f)
        frontier.append(row)
    frontier=pd.DataFrame(frontier)
    frontier.to_csv(out/"break_even_discount_by_cap.csv",index=False)

    # Exact-penalty sufficient threshold from the base C cap dual (see proof).
    alpha_safe = (1-b0)*com.import_tariff+base.C.cap_dual.max()
    before = run("C",alpha=alpha_safe-1e-4)
    after = run("C",alpha=alpha_safe+1e-4)
    assert after.excess.sum()<TOL
    assert abs(after.total_cost-base.C.total_cost)<TOL
    # At the equality threshold a violation can be cost-neutral; do not
    # call the boundary itself a guarantee for any LP optimum.
    plateau_I=run("I",alpha=6.)
    assert abs(plateau_I.excess.sum()-base.validation["unavoidable_I_exceedance_kWh"])<TOL

    # The DSO relief as a function of alpha can jump at a merit-order tie.
    # Locate the first observed beneficial interval, not a false continuous root.
    relief_brackets={}
    for m in ("I","C"):
        table=penalties[penalties.model==m].sort_values("alpha_DKK_per_kWh")
        relief=table.exceedance_reduction_kWh.to_numpy()
        positive=np.flatnonzero(relief>1e-5)
        j=int(positive[0]) if len(positive) else None
        relief_brackets[m]=None if j is None else [float(table.alpha_DKK_per_kWh.iloc[max(0,j-1)]),
                                                   float(table.alpha_DKK_per_kWh.iloc[j])]

    thresholds=dict(discount_break_even=beta_star,cap_factor_break_even=f_star,
                    cap_kW_break_even={m:None if v is None else float(contract.capacity_limit_kW[0]*v/f0) for m,v in f_star.items()},
                    penalty_cost_break_even=alpha_star,penalty_DSO_relief_grid_bracket=relief_brackets,
                    C_exact_penalty_sufficient_DKK_per_kWh=float(alpha_safe),
                    C_exceedance_just_below_threshold_kWh=float(before.excess.sum()),
                    C_exceedance_just_above_threshold_kWh=float(after.excess.sum()))
    (out/"thresholds.json").write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
    return discount,caps,penalties,frontier,thresholds

