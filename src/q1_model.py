"""Parametrised, continuous LP for Assignment 2 Question 1.

The member block is independent of the meter architecture.  R and I have
one meter per member; C has one shared meter.  All powers are hourly kWh
for the one-hour step (numerically kW).  Positive member q is net demand.

The assignment text charges the discounted import tariff *below* the cap
and alpha above it.  Writing e = max(0, import-cap) gives the equivalent
linear cost (p+a)*import + (b-p)*export + (alpha-a)*e.  A strictly
positive alpha-a makes e exact without binary variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .data_loader import CommunityData


TOL = 1e-7


class _LinearProgram:
    """Small named sparse-LP builder; bounds are part of the LP, not postprocessing."""

    def __init__(self):
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.var_names: list[str] = []
        self.eq_names: list[str] = []
        self.ub_names: list[str] = []
        self.eq_rows: list[dict[int, float]] = []
        self.ub_rows: list[dict[int, float]] = []
        self.beq: list[float] = []
        self.bub: list[float] = []

    def variables(self, name, size, cost=0.0, lb=0.0, ub=np.inf):
        """Create a block of continuous variables and return its column indices."""
        idx = np.arange(len(self.c), len(self.c) + size, dtype=int)
        for dest, values in ((self.c, cost), (self.lb, lb), (self.ub, ub)):
            dest.extend(np.broadcast_to(np.asarray(values, float), (size,)).tolist())
        self.var_names.extend(f"{name}[{t}]" for t in range(size))
        return idx

    def equality(self, name, terms: dict[int, float], rhs=0.0):
        self.eq_names.append(name)
        self.eq_rows.append(terms)
        self.beq.append(float(rhs))
        return len(self.beq) - 1

    def inequality(self, name, terms: dict[int, float], rhs):
        """Add terms @ x <= rhs and return the row index."""
        self.ub_names.append(name)
        self.ub_rows.append(terms)
        self.bub.append(float(rhs))
        return len(self.bub) - 1

    def _matrix(self, rows: list[dict[int, float]]) -> csr_matrix:
        ri, ci, val = [], [], []
        for row, terms in enumerate(rows):
            for col, coef in terms.items():
                ri.append(row)
                ci.append(col)
                val.append(coef)
        return coo_matrix((val, (ri, ci)), shape=(len(rows), len(self.c))).tocsr()

    def solve(self, method="highs-ds", *, price_only_objective=None):
        """Solve the LP, including bounds, and return primal/dual validation data."""
        c = np.asarray(self.c)
        lb, ub = np.asarray(self.lb), np.asarray(self.ub)
        aeq, aub = self._matrix(self.eq_rows), self._matrix(self.ub_rows)
        beq, bub = np.asarray(self.beq), np.asarray(self.bub)
        ans = linprog(c if price_only_objective is None else price_only_objective,
                      A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq,
                      bounds=list(zip(lb, ub)), method=method,
                      options={"primal_feasibility_tolerance": 1e-8,
                               "dual_feasibility_tolerance": 1e-8})
        if not ans.success:
            raise RuntimeError(f"LP was not solved to optimality: {ans.message}")
        # HiGHS gives the derivative with respect to the RHS of each row.
        # Equality marginals are free; <=-constraint marginals are nonpositive.
        y, z = ans.eqlin.marginals, ans.ineqlin.marginals
        lo, hi = ans.lower.marginals, ans.upper.marginals
        finite_l, finite_u = np.isfinite(lb), np.isfinite(ub)
        dual_value = (beq @ y + bub @ z + lb[finite_l] @ lo[finite_l]
                      + ub[finite_u] @ hi[finite_u])
        objective_coef = c if price_only_objective is None else price_only_objective
        stationarity = objective_coef - aeq.T @ y - aub.T @ z - lo - hi
        x = ans.x

        def maximum(a):
            return float(np.max(np.abs(a), initial=0.0))

        checks = {
            "dual_gap_DKK": abs(float(ans.fun - dual_value)),
            "stationarity_residual": maximum(stationarity),
            "equality_residual": maximum(aeq @ x - beq),
            "inequality_violation": maximum(np.maximum(aub @ x - bub, 0.0)),
            "bound_violation": max(maximum(np.maximum(lb[finite_l] - x[finite_l], 0.0)),
                                   maximum(np.maximum(x[finite_u] - ub[finite_u], 0.0))),
            "dual_sign_violation": max(maximum(np.maximum(z, 0.0)),
                                       maximum(np.maximum(-lo, 0.0)),
                                       maximum(np.maximum(hi, 0.0))),
            "complementarity_residual": max(maximum((bub - aub @ x) * z),
                                            maximum((x[finite_l] - lb[finite_l]) * lo[finite_l]),
                                            maximum((ub[finite_u] - x[finite_u]) * hi[finite_u])),
        }
        if max(checks.values()) > TOL * 10:
            raise AssertionError(f"LP primal/dual check failed: {checks}")
        return ans, checks


@dataclass
class MemberDispatch:
    """Primal schedules and daily-energy multiplier for one consumer."""
    load: np.ndarray
    deviation: np.ndarray
    pv: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    q: np.ndarray
    energy_dual: float
    disutility: float
    pv_cost: float

    @property
    def local_cost(self) -> float:
        return self.disutility + self.pv_cost


@dataclass
class Solution:
    """Meter and member primal schedules, marginal prices, costs and checks."""
    mode: str
    beta: float
    alpha: float
    cap: np.ndarray | None
    member_ids: list[str]
    groups: list[list[int]]
    members: dict[str, MemberDispatch]
    imp: np.ndarray
    exp: np.ndarray
    excess: np.ndarray
    meter_cap: np.ndarray | None
    price: np.ndarray
    cap_dual: np.ndarray
    meter_bill: np.ndarray
    total_cost: float
    components: dict[str, float]
    checks: dict[str, float]

    @property
    def aggregate_import(self):
        return self.imp.sum(axis=0)

    @property
    def aggregate_export(self):
        return self.exp.sum(axis=0)

    @property
    def individual_net(self):
        return np.array([self.members[i].q for i in self.member_ids])

    @property
    def individual_cost(self):
        """Settled private cost: own grid bill (R/I), or lambda*q (C)."""
        costs = {}
        for k, group in enumerate(self.groups):
            for i in group:
                mid = self.member_ids[i]
                m = self.members[mid]
                payment = (self.meter_bill[k] if len(group) == 1 else
                           float(self.price[k] @ m.q))
                costs[mid] = m.local_cost + float(payment)
        return costs

    @property
    def manager_budget(self):
        """Uniform-price collections minus external meter bill (shared meter only)."""
        if len(self.groups) != 1 or len(self.groups[0]) == 1:
            return 0.0
        return float(self.price[0] @ self.individual_net.sum(axis=0) - self.meter_bill[0])

    @property
    def internal_exchange(self):
        """Physical behind-meter exchange, not counterflows across separate meters."""
        q = self.individual_net
        if len(self.groups) != 1 or len(self.groups[0]) == 1:
            return np.zeros(q.shape[1])
        return np.minimum(np.maximum(q, 0).sum(axis=0), np.maximum(-q, 0).sum(axis=0))


class EnergyCommunityLP:
    """One member/meter model for R, I, C and metering/contract limit cases.

    All contract parameters are changed in memory; the starter data are not
    modified.  The terminal SOC is cyclic unless explicitly set to 'free'.
    """

    def __init__(self, community: CommunityData, *, terminal="cyclic", method="highs-ds"):
        self.com = community
        self.terminal = terminal
        self.method = method
        if terminal not in ("cyclic", "free"):
            raise ValueError("terminal must be 'cyclic' or 'free'")
        if community.contract.export_capacity_limit_kW is not None:
            raise ValueError("Q1 assumes the supplied case with no export cap")

    def solve(self, mode: str, *, beta: float | None = None, alpha: float | None = None,
              cap_scale: float = 1.0, cap_profile: np.ndarray | None = None,
              use_contract: bool | None = None, individual_metering: bool | None = None,
              member_ids: Iterable[str] | None = None,
              reference_tie: str | None = None) -> Solution:
        """Optimize the given mode; R and I are separable when solved together.

        Metering and contract switches allow the required no-cap, full-tariff,
        own-meter limit case of C.  cap_scale multiplies the provided hourly
        cap; the fixed official net-energy shares also scale with it.
        For R only, reference_tie='min'/'max' produces an optimal-face witness
        with lower/higher imports in the original congested-hour set. This
        diagnoses nonunique reference schedules without altering primary cost.
        """
        if mode not in ("R", "I", "C"):
            raise ValueError("mode must be R, I or C")
        com, contract = self.com, self.com.contract
        selected = set(member_ids) if member_ids is not None else None
        ms = [m for m in com.members if selected is None or m.consumer_id in selected]
        if not ms or (selected is not None and len(ms) != len(selected)):
            raise ValueError("Unknown/empty member selection")
        ids = [m.consumer_id for m in ms]
        n, nt = len(ms), len(com.energy_price)
        contracted = mode != "R" if use_contract is None else use_contract
        own = mode != "C" if individual_metering is None else individual_metering
        groups = [[i] for i in range(n)] if own else [list(range(n))]
        beta = float(contract.tariff_discount_ratio if beta is None else beta) if contracted else 0.0
        alpha = float(contract.penalty_DKK_per_kWh if alpha is None else alpha) if contracted else 0.0
        if not 0 <= beta <= 1:
            raise ValueError("discount must be in [0, 1]")
        a, b = (1 - beta) * com.import_tariff, (1 - beta) * com.export_tariff
        gamma = alpha - a if contracted else 0.0
        if contracted and gamma <= 0:
            raise ValueError("For the below-cap tariff rule this LP requires alpha > discounted import tariff")
        p = np.asarray(com.energy_price)
        if np.min(p - b) <= 0:
            raise ValueError("Q1 LP assumes positive effective export prices (no dissipative cycling)")

        cap = None
        meter_cap = None
        if contracted:
            cap = np.asarray(contract.capacity_limit_kW if cap_profile is None else cap_profile,
                             float) * cap_scale
            if cap.shape != (nt,) or np.min(cap) < 0:
                raise ValueError("cap must be a nonnegative hourly vector")
            # The supplied shares are rounded to six decimals. Normalization
            # enforces sum_i kappa_i = 1 without changing the split rule.
            share_sum = sum(contract.share.values())
            meter_cap = np.array([cap * contract.share[ids[g[0]]] / share_sum
                                  for g in groups]) if own else cap[None, :]

        lp = _LinearProgram()
        cols = []
        daily_rows = []
        # Common appliance constraints, written once for all three models.
        for m in ms:
            cid = m.consumer_id
            l = lp.variables(f"l_{cid}", nt, lb=m.load_lower_kWh, ub=m.load_upper_kWh)
            d = lp.variables(f"d_{cid}", nt, cost=m.linear_disutility)
            g = lp.variables(f"g_{cid}", nt, cost=m.pv_marginal_cost,
                             ub=m.pv_available) if m.has_pv else None
            ch = dis = s = None
            for t in range(nt):
                lp.inequality(f"abs+_{cid}[{t}]", {l[t]: 1., d[t]: -1.}, m.reference_load[t])
                lp.inequality(f"abs-_{cid}[{t}]", {l[t]: -1., d[t]: -1.}, -m.reference_load[t])
            daily_rows.append(lp.inequality(f"daily_{cid}", {j: -1. for j in l}, -m.min_daily_energy_kWh))
            if m.has_battery:
                ch = lp.variables(f"ch_{cid}", nt, ub=m.battery_max_charge_kW)
                dis = lp.variables(f"dis_{cid}", nt, ub=m.battery_max_discharge_kW)
                s = lp.variables(f"soc_{cid}", nt, ub=m.battery_capacity_kWh)
                for t in range(nt):
                    row = {s[t]: 1., ch[t]: -m.battery_charging_efficiency,
                           dis[t]: 1. / m.battery_discharging_efficiency}
                    if t:
                        row[s[t - 1]] = -1.
                    lp.equality(f"soc_{cid}[{t}]", row, m.battery_initial_soc_kWh if t == 0 else 0.)
                if self.terminal == "cyclic":
                    lp.equality(f"terminal_{cid}", {s[-1]: 1.}, m.battery_initial_soc_kWh)
            cols.append((l, d, g, ch, dis, s))

        meter_cols, balance_rows, cap_rows = [], [], []
        for k, group in enumerate(groups):
            imp = lp.variables(f"import_{k}", nt, cost=p + a)
            exp = lp.variables(f"export_{k}", nt, cost=b - p)
            e = lp.variables(f"excess_{k}", nt, cost=gamma) if contracted else None
            balances, caps = [], []
            for t in range(nt):
                # Import - export - sum(net member demand) = 0.
                # Its RHS marginal is lambda in the course sign convention.
                row = {imp[t]: 1., exp[t]: -1.}
                for i in group:
                    l, d, g, ch, dis, s = cols[i]
                    row[l[t]] = -1.
                    if g is not None:
                        row[g[t]] = 1.
                    if ch is not None:
                        row[ch[t]], row[dis[t]] = -1., 1.
                balances.append(lp.equality(f"balance_{k}[{t}]", row))
                if contracted:
                    caps.append(lp.inequality(f"cap_{k}[{t}]", {imp[t]: 1., e[t]: -1.}, meter_cap[k, t]))
            meter_cols.append((imp, exp, e))
            balance_rows.append(balances)
            cap_rows.append(caps)

        ans, checks = lp.solve(method=self.method)
        x = ans.x
        if reference_tie is not None:
            if reference_tie not in ("min", "max") or contracted or not own:
                raise ValueError("reference_tie is an own-meter, no-contract diagnostic")
            original_import = np.array([x[col[0]] for col in meter_cols]).sum(axis=0)
            congested = original_import > contract.capacity_limit_kW + TOL
            tie_cost = np.zeros(len(lp.c))
            for imp, _, _ in meter_cols:
                tie_cost[imp[congested]] = 1. if reference_tie == "min" else -1.
            lp.equality("primary_optimal_face", {j: c for j, c in enumerate(lp.c) if c}, ans.fun)
            witness, witness_checks = lp.solve(method=self.method, price_only_objective=tie_cost)
            x = witness.x
            checks.update({"tie_" + k: v for k, v in witness_checks.items()})
            checks["tie_primary_cost_error"] = abs(float(np.asarray(lp.c) @ x - ans.fun))
        zero = np.zeros(nt)
        dispatch = {}
        for j, m in enumerate(ms):
            l, d, g, ch, dis, s = cols[j]
            gl, cl, dl, sl = [x[idx].copy() if idx is not None else zero.copy() for idx in (g, ch, dis, s)]
            q = x[l] + cl - dl - gl
            dispatch[ids[j]] = MemberDispatch(
                load=x[l].copy(), deviation=x[d].copy(), pv=gl, charge=cl, discharge=dl,
                soc=sl, q=q, energy_dual=float(-ans.ineqlin.marginals[daily_rows[j]]),
                disutility=float(m.linear_disutility * x[d].sum()),
                pv_cost=float(m.pv_marginal_cost * gl.sum()),
            )

        # Canonical gross flows are the positive/negative parts of net flow.
        # For beta=1 and slack cap, an optimal LP may contain cost-neutral
        # simultaneous metering. Removing it preserves net flow and cost.
        nets = np.array([sum((dispatch[ids[j]].q for j in group), start=zero.copy()) for group in groups])
        imp, exp = np.maximum(nets, 0), np.maximum(-nets, 0)
        excess = np.maximum(imp - meter_cap, 0) if contracted else np.zeros_like(imp)
        prices = np.array([ans.eqlin.marginals[row] for row in balance_rows])
        cap_dual = np.array([-ans.ineqlin.marginals[row] for row in cap_rows]) if contracted else np.zeros_like(imp)
        energy = (imp - exp) @ p
        tariffs = a * (imp - excess).sum(axis=1) + b * exp.sum(axis=1)
        penalty = alpha * excess.sum(axis=1)
        bills = energy + tariffs + penalty
        components = dict(energy=float(energy.sum()), tariffs=float(tariffs.sum()),
                          penalty=float(penalty.sum()),
                          disutility=sum(m.disutility for m in dispatch.values()),
                          pv_cost=sum(m.pv_cost for m in dispatch.values()))
        cost = float(sum(components.values()))
        checks["objective_recomposition"] = abs(cost - ans.fun)
        checks["daily_energy_error"] = max(abs(dispatch[m.consumer_id].load.sum() - m.min_daily_energy_kWh) for m in ms)
        checks["absolute_deviation_error"] = max(float(np.max(np.abs(dispatch[m.consumer_id].deviation -
                                                np.abs(dispatch[m.consumer_id].load - m.reference_load)))) for m in ms)
        checks["simultaneous_charge_discharge"] = max(float(np.max(np.minimum(d.charge, d.discharge))) for d in dispatch.values())
        checks["meter_cap_rent_identity"] = (abs(float(np.sum(prices * (imp - exp)) - bills.sum() -
                                                       np.sum(cap_dual * meter_cap))) if contracted else
                                              abs(float(np.sum(prices * (imp - exp)) - bills.sum())))
        # KKT-based meter price bounds, including the zero-flow interval.
        upper = p[None, :] + a + cap_dual
        checks["meter_price_bound_error"] = max(float(np.max(np.maximum(p[None, :] - b - prices, 0))),
                                                float(np.max(np.maximum(prices - upper, 0))))
        if max(checks.values()) > 1e-6:
            raise AssertionError(f"Model/settlement validation failed: {checks}")
        return Solution(mode, beta, alpha, cap, ids, groups, dispatch, imp, exp, excess,
                        meter_cap, prices, cap_dual, bills, cost, components, checks)
