# gum-calc

> 🇫🇷 [Version française](README.md)

GUM uncertainty calculation engine for experimental physics lab reports, with LaTeX export.

---

## Author

**Victorio BONNEVILLE DIAZ** — L3 General Physics student, UPEC (Université Paris-Est Créteil)

Code generated with [Claude (Anthropic)](https://www.anthropic.com) from an architecture and logic defined by the author, based on GUM metrology courses from L2 Physics at Université Paris-Est Créteil.

---

## Overview

This project aims to simplify and automate uncertainty calculations for physics lab reports, through a calculation engine and LaTeX export.

The engine complies with the **GUM** standard (*Guide to the Expression of Uncertainty in Measurement*). It covers type A and B uncertainties, propagation by symbolic differentiation via SymPy, the coverage factor via the Welch-Satterthwaite formula, and linear regression by least squares.

---

## Project structure

```
gum-calc/
├── GUM_-_Incertitudes_de_Mesures.ipynb   # Template notebook
├── gum_calc.py                           # Calculation engine and LaTeX export
└── README.md
```

---

## How it works

The template notebook serves as a reference that can be adapted for each problem.
For each lab session, two options:
- duplicate the template notebook and fill in one cell per measurand
- use [Claude (Anthropic)](https://www.anthropic.com) through the skill: "Skill gum-notebook.skill"

Each cell follows the same structure:

**Input uncertainties:** for each measured quantity, choose the appropriate uncertainty type from:
- `uncertainty_type_A` (repeated measurements)
- `uncertainty_type_B_from_resolution` (instrument resolution)
- `uncertainty_type_B_uniform` (known uniform distribution)
- `uncertainty_type_B_relative` (already known uncertainty, from calibration or propagation)
- `uncertainty_type_exact` (constant).

**GUM calculation:** `full_gum_analysis` takes the measurand formula (in SymPy), nominal values and input uncertainties, and returns the propagated result, the combined uncertainty `uc`, the coverage factor `k` (via Welch-Satterthwaite if needed), the expanded uncertainty `U`, and the uncertainty budget.

**Report generation:** `generate_bilan` produces the corresponding LaTeX block (measurement model, nominal values, source-by-source uncertainties, sensitivity coefficients, budget, boxed result), ready to use in the report.

**Final export:** `generate_annexe` assembles all bilans generated in the notebook into a single LaTeX section `\section{Annexe : Bilans d'incertitudes}`.

**Any modification to the calculation engine (formulas, rounding, LaTeX format) must be made exclusively in `gum_calc.py`, never in the notebook.**

---

## Installation and setup

Libraries used:

- `sympy` — symbolic differentiation of sensitivity coefficients
- `scipy` — Student's t-table for the coverage factor (Welch-Satterthwaite)
- `math` and `re` — numerical calculations and unit parsing (Python standard library)
- `jupyter` — execution environment for the template notebook

Install dependencies:

```bash
pip install sympy scipy jupyter
```

**Path configuration in the notebook:**

At the top of the notebook, the following line points to the folder containing `gum_calc.py`. Adapt it to your machine:

```python
import sys
sys.path.insert(0, r"C:\path\to\gum-calc")
```

**Tested versions:** Python 3.11.9 · SymPy 1.14.0 · SciPy 1.17.1

---

## Step 1 — Declare uncertainties

Before calculating anything, each input quantity is characterised by its standard uncertainty. `gum_calc.py` offers five functions depending on the situation. Valid types recognised by the LaTeX export are `"exact"`, `"A"` and `"B"` — any other type raises an explicit `ValueError`.

---

### Type A uncertainty

Type A uncertainty applies when a series of repeated measurements of the same quantity is available. Statistical dispersion is exploited: the standard uncertainty is the standard deviation of the mean,

```
u_A = s / √N
```

where `s` is the empirical standard deviation of the series and `N` the number of measurements.

```python
u_x = uncertainty_type_A(values)
```

| Parameter | Type | Description |
|---|---|---|
| `values` | `list[float]` | List of N measured values (N ≥ 2) |

The useful value going forward is `u_x["u"]`. The key `nu` (= N−1) is retained for Welch-Satterthwaite if N < 30.

---

### Type B uncertainty — instrument resolution

Type B uncertainty applies when measurements are not repeated. Ignorance about the true value is modelled by a uniform distribution of width δ (the instrument resolution). The associated standard uncertainty is

```
u_B = (δ/2) / √3
```

```python
u_x = uncertainty_type_B_from_resolution(resolution)
```

| Parameter | Type | Description |
|---|---|---|
| `resolution` | `float` | Instrument resolution δ (last digit or graduation) |

---

### Type B uncertainty — half-width known directly

Same principle as above, but when the manufacturer's datasheet gives a tolerance ±a directly rather than a resolution δ.

```python
u_x = uncertainty_type_B_uniform(half_width)
```

| Parameter | Type | Description |
|---|---|---|
| `half_width` | `float` | Half-width a of the interval (same unit as the quantity) |

> Note: `uncertainty_type_B_from_resolution(δ)` is a shortcut that calls `uncertainty_type_B_uniform(δ/2)` internally.

---

### Type B uncertainty — value provided directly

When the standard uncertainty is known directly — from calibration, manufacturer's datasheet, or propagated from an already-calculated intermediate measurand.

```python
u_x = uncertainty_type_B_relative(u_standard=value)
```

| Parameter | Type | Description |
|---|---|---|
| `u_standard` | `float` | Standard uncertainty u (not the expanded uncertainty U = k·u) |
| `relative_knowledge` | `float` or `None` | Relative knowledge on u — if provided, degrees of freedom estimated by ν = 1/(2r²) |

> In a measurand chain, always pass `res["uc"]`, never `res["U"]`. Passing U = k·u_c would double the coverage factor — this is a GUM error.

---

### Exact constant

For a fundamental physical constant (c, e, h…) or any quantity whose uncertainty is negligible compared to other sources.

```python
u_x = uncertainty_type_exact()
```

The standard uncertainty is zero. The variable is excluded from the uncertainty budget.

---

## Step 2 — Calculate the measurand

`full_gum_analysis` is the main function. It takes the measurand formula, nominal values and uncertainties declared in step 1, and returns the complete result: GUM propagation by symbolic differentiation, Welch-Satterthwaite if needed, and consistent rounding.

```python
res = full_gum_analysis(
    formula_str        = "...",
    variable_names     = [...],
    nominal_values     = {...},
    uncertainty_inputs = {...},
)
```

| Parameter | Description |
|---|---|
| `formula_str` | SymPy expression of the measurand — `*` for multiplication, `**` for power, `sqrt()`, `log()`, `sin()`, etc. |
| `variable_names` | Ordered list of variable names — must be identical to the keys of `nominal_values` and `uncertainty_inputs`. Cannot be empty. |
| `nominal_values` | Dict `{name: nominal value}` |
| `uncertainty_inputs` | Dict `{name: dict uncertainty_type_*}` |
| `k_override` | Forced k factor (optional) — if absent, Welch-Satterthwaite determines k automatically |

Useful keys in the result:

| Key | Description |
|---|---|
| `res["result_rounded"]` | Rounded nominal value, consistent with U |
| `res["U_rounded"]` | Expanded uncertainty U = k·u_c, rounded to 2 significant figures |
| `res["uc"]` | Combined standard uncertainty u_c (to pass in a chain — see below) |
| `res["k"]` | Coverage factor retained |
| `res["nu_eff"]` | Effective degrees of freedom |
| `res["budget"]` | Dict `{name: %}` — contribution of each source to the combined variance |

---

## Step 3 — Generate the LaTeX report

`generate_bilan` takes the same inputs as `full_gum_analysis` and produces a complete LaTeX block: measurement model, nominal values, source-by-source uncertainties, sensitivity coefficients, Welch-Satterthwaite if needed, budget, and boxed result.

```python
bilan = generate_bilan(
    measurand_name     = "...",
    measurand_symbol   = "...",
    formula_str        = "...",
    variable_names     = [...],
    variable_symbols   = {...},
    variable_units     = {...},
    nominal_values     = {...},
    uncertainty_inputs = {...},
    measurand_unit     = r"...",
)
```

| Parameter | Description |
|---|---|
| `measurand_name` | Full name of the measurand — displayed in parentheses after the symbol in the introduction sentence (e.g. "The measurand $R$ (Resistance) is related to…"). Ignored if identical to `measurand_symbol`. |
| `measurand_symbol` | LaTeX symbol of the measurand |
| `formula_str` | Same expression as in `full_gum_analysis` |
| `variable_names` | Same list as in `full_gum_analysis` |
| `variable_symbols` | Dict `{python_name: LaTeX_symbol}` |
| `variable_units` | Dict `{python_name: siunitx_unit}` |
| `nominal_values` | Same dict as in `full_gum_analysis` |
| `uncertainty_inputs` | Same dict as in `full_gum_analysis` |
| `measurand_unit` | siunitx unit of the measurand |
| `k_override` | Forced k factor (optional) |
| `subsection` | `True` by default — generates a `\subsection{}` at the top of the report |
| `global_sig_figs` | Significant figures for intermediate values (default: 3) |

Units follow the siunitx package syntax: `r"\ohm"`, `r"\volt"`, `r"\ampere"`, `r"\second"`, `r"\meter"`, `r"\kilo\gram"`, `r"\meter\per\second"`, etc. Empty string `""` if dimensionless.

---

## Step 4 — Assemble the LaTeX appendix

`generate_annexe` takes the list of all bilans in the order of the report and produces the complete LaTeX section.

```python
print(generate_annexe([bilan_1, bilan_2, ...]))
```

---

## Special cases

### Linear regression only

When characterising an experimental line without deriving another measurand from it. The model is y = θ₀ + θ₁·x. The report includes the estimators, their standard uncertainties derived from the residual variance, and the r² coefficient.

The slope unit is automatically deduced by algebra from `y_unit / x_unit` — for example `\meter\per\second` divided by `\second` gives `\meter\per\square\second`. It can be explicitly overridden via `slope_unit` if needed.

```python
bilan_reg = generate_bilan_regression(
    x_symbol         = "...",
    y_symbol         = "...",
    x_unit           = r"...",
    y_unit           = r"...",
    x_data           = [...],
    y_data           = [...],
    subsection_title = "...",
    slope_unit       = r"...",    # optional — deduced from y_unit/x_unit if absent
    intercept_unit   = r"...",    # optional — equals y_unit if absent
    slope_symbol     = r"\theta_1",
    intercept_symbol = r"\theta_0",
    global_sig_figs  = 3,         # optional — significant figures for intermediate values
)
```

---

### Regression + derived measurand in one call

When the final measurand depends on the slope or intercept of a regression. The names `"theta1"` (slope) and `"theta0"` (intercept) in `variable_names` are automatically recognised and fed by the regression — no need to provide them in `nominal_values_helpers` or `uncertainty_inputs_helpers`.

```python
latex = full_pipeline_regression_to_measurand(
    x_data                     = [...],
    y_data                     = [...],
    x_symbol                   = "...",
    y_symbol                   = "...",
    x_unit                     = r"...",
    y_unit                     = r"...",
    formula_str                = "...",
    variable_names             = [...],
    variable_symbols           = {...},
    variable_units             = {...},
    nominal_values_helpers     = {...},
    uncertainty_inputs_helpers = {...},
    measurand_symbol           = "...",
    measurand_name             = "...",   # optional — displayed in parentheses in the GUM report
    measurand_unit             = r"...",
    slope_symbol               = r"\theta_1",
    intercept_symbol           = r"\theta_0",
    global_sig_figs            = 3,       # optional — sig. fig. consistency regression + GUM
)
print(latex)
```

The output contains the regression report followed by the GUM report of the measurand, separated by a `\clearpage`.

---

### Measurand chain

When a measurand Y depends on a previously calculated measurand X, pass `res_X["uc"]` via `uncertainty_type_B_relative`.

```python
res_X = full_gum_analysis(...)

uncertainty_inputs_Y = {
    "X":     uncertainty_type_B_relative(u_standard=res_X["uc"]),  # never res_X["U"]
    "other": uncertainty_type_B_from_resolution(...),
}
res_Y = full_gum_analysis(...)
```

> Never pass `res_X["U"]` — that would apply k twice.