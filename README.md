# 46750 — Assignment 2, Question 1

## Reproduce the report

The original starter package supplied for Assignment 2 is retained: all files under
data/ and the original loader and scenario-generation module are unmodified.
See STARTER_README.md for the course's detailed data/API documentation.
Question 1 uses the supplied community "base" and day "d1".

### Environment (Windows, macOS or Linux)

Tested with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and
Matplotlib 3.10.8. The LP backend is HiGHS dual simplex through scipy.optimize.linprog;
it reports primal and dual solutions and does not require a Gurobi license.
The course starter allows other solver implementations; its Gurobi requirements
remain available for later questions.

Create a virtual environment and install the pinned Q1 dependencies:

    python -m venv .venv
    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    # macOS/Linux: source .venv/bin/activate
    python -m pip install -r requirements-q1.txt

### Commands (run from this directory)

| Experiment | One command | Outputs |
|---|---|---|
| Base R, I, C and validation | python run_q1.py base | base_metrics.csv, base_hourly.csv, member_dispatch.csv, input_profiles.csv, base_shift_by_slope.csv, validation.json, base tables and figures |
| Dual settlement / IR / budget | python run_q1.py settlement | member_settlement.csv, settlement_by_asset.csv, settlement_by_slope.csv, settlement_prices.pdf, hourly_duals.tex |
| Contract terms | python run_q1.py sensitivity | discount_sweep.csv, cap_sweep.csv, penalty_sweep.csv, break_even_discount_by_cap.csv, thresholds.json, sensitivity tables and figures (also re-runs base validation) |
| Full Q1 | python run_q1.py all | All the above |

All generated output is in results/q1/ by default. To write to a different
directory, use --out PATH. The report in report/q1_report.tex expects the
default output path. CSVs retain full precision; values in the report are
formatted, with solver-scale values rounded to zero for display only.

After running the full Q1 command, build the editable report using a standard
LaTeX distribution:

    cd report
    latexmk -pdf -interaction=nonstopmode -halt-on-error q1_report.tex

Alternatively, run pdflatex -interaction=nonstopmode -halt-on-error q1_report.tex
twice (for cross-references), then open the resulting q1_report.pdf.

## Structure

    run_q1.py               command-line experiment entry point
    src/
      data_loader.py        supplied, unchanged Assignment 1/2 InputData loader
      scenario_generation.py supplied, unchanged (for Question 3)
      q1_model.py           one sparse LP and shared appliance block for R/I/C
      q1_analysis.py        metrics, experiments, exact crossings, validation
      q1_plots.py           automatically generated, labeled PDF/PNG figures
      q1_report_tables.py   automatically generated LaTeX tables and numeric macros
    data/                   original course data, unchanged
    report/q1_report.tex    editable Q1 report, referencing the generated figures/tables
    results/q1/             solved CSVs, validation, plot files, report tables
    STARTER_README.md       original documentation of the starter package

### Modeling conventions

- CommunityData and InputData from the original starter loader are passed
  directly to EnergyCommunityLP. R and I have separate member meters and
  separable objectives; C shares one meter. This is the same parametric LP
  with different metering and contract switches, not three independent copies.
- An hourly interval is one hour. A numeric flow in kW also denotes the number
  of kWh in that interval. SOC is in kWh; costs are DKK/day.
- Battery trajectories obey the supplied efficiency and initial SOC, with an
  explicit cyclic terminal condition (s_23 = s_initial). This reproduces the
  official R benchmark and avoids one-off gains from depleting initial energy.
- Following the assignment handout's explicit "below the cap" tariff rule,
  above-cap import pays p + alpha, while below-cap import pays p + a with
  a = (1-beta)*tau_imp. Thus the coefficient on the excess variable is
  gamma = alpha-a, not alpha. All sweeps satisfy gamma>0. If a different
  interpretation charged the discounted tariff also on the above-cap tranche,
  its excess coefficient would instead be alpha; the submitted report and
  code consistently use the handout's stated tariff rule.
- The supplied cap is 9.39 kW and the six-decimal individual contract shares
  are normalized to sum to exactly one. Only the cap profile is scaled in
  cap sweeps; the base reference shares are kept fixed. No data file is edited.
- Solver equality balances use import minus export minus total net demand
  equals zero. The RHS equality marginal is the positive economic internal
  price. Cap constraints are import minus excess <= cap; the negative of
  the solver's RHS inequality marginal is the nonnegative cap multiplier.
- For a discount of one, simultaneous metered import/export can be an
  optimizer tie. The reported gross flows are the positive/negative parts
  of net demand, preserving feasibility and economic cost. With the given
  positive buying/selling price regimes and lossy batteries, simultaneous
  charging/discharging is dominated.
- This LP is designed for the supplied positive-price, no-export-cap,
  linear-disutility case of Question 1. Later questions can reuse this
  loader, member parameters, meter-cost expression, and reference results.

### Validation (automatically executed)

Every solve must be optimal and passes primal feasibility, bound feasibility,
dual signs, stationarity, dual objective / primal objective agreement,
complementarity, objective recomposition, exact load deviation, daily energy,
absence of simultaneous charge/discharge, meter price bounds, and the
cap-rent identity. Feasibility tolerances are 1e-8; assertions use 1e-6
for numerical residual checks.

The base case additionally checks the published cost, daily import, peak,
and numbers of congested/binding hours, distinct optimal-face witnesses for
the reference's nonunique exceedance, reference net imports, the cap split,
independently solved R/I members, and two no-contract limit cases.
All three base optimal objective values are independently confirmed with
HiGHS interior-point plus crossover.
The unavoidable individual excess is checked against an analytical lower
bound. Each member is re-optimized at the community's solved internal prices
to verify price support; the largest cap dual is checked against a finite
difference. The sweeps check cost dominance and the monotonic properties
of optimal values and billed excess. The exact-penalty threshold is tested
immediately on either side. Any failed assertion stops the run.

The given rounded reference exceedance (66.5 kWh) is not a unique scalar
among the primary LP's equal-cost schedules. The code evaluates two optimal-face
witnesses without changing the cost objective; the value in the starter file
lies between their exceedances. The primary dual-simplex solution is used
consistently across the report.

### Report integration

The LaTeX document is a self-contained Question 1 section with its own
nomenclature and appendix so that it can be integrated into the group's
final report. The full group report should retain the course-required
participation table with the real contributions (not estimated here),
the AI-use disclosure, and the reproducibility/repository link. The source
includes a disclosure for the assistance used to produce this section;
the group should verify and adapt it to accurately describe the final work.
