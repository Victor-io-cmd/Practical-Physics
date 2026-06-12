"""
gum_calc.py — Moteur de calcul GUM + Export LaTeX
"""

import math
import sympy as sp
from scipy import stats as scipy_stats


# ============================================================
# PARTIE 1 — MOTEUR DE CALCUL GUM
# ============================================================

def uncertainty_type_A(values: list[float]) -> dict:
    """
    Incertitude de type A par répétition sur N mesures indépendantes.

    Paramètres
    ----------
    values : liste des valeurs mesurées (au moins 2).

    Retourne un dict avec :
      mean  — moyenne empirique
      s     — écart-type empirique (estimateur de σ)
      u     — incertitude-type u_A = s / sqrt(N)
      N     — nombre de mesures
      nu    — degrés de liberté (N - 1)
      type  — "A"
    """
    N = len(values)
    if N < 2:
        raise ValueError("Type A requiert au moins 2 mesures.")
    mean = sum(values) / N
    variance = sum((x - mean) ** 2 for x in values) / (N - 1)
    s = math.sqrt(variance)
    u_A = s / math.sqrt(N)
    return {
        "mean": mean,
        "s": s,
        "u": u_A,
        "N": N,
        "nu": N - 1,
        "type": "A",
    }


def uncertainty_type_B_uniform(half_width: float) -> dict:
    """
    Incertitude de type B — distribution uniforme de demi-largeur a.

    u_B = a / sqrt(3)

    Paramètres
    ----------
    half_width : demi-largeur a de la distribution (même unité que la grandeur).
    """
    u_B = half_width / math.sqrt(3)
    return {
        "u": u_B,
        "a": half_width,
        "type": "B",
        "distribution": "uniform",
        "nu": float("inf"),
    }


def uncertainty_type_B_from_resolution(resolution: float) -> dict:
    """
    Incertitude de type B à partir de la résolution δ d'un instrument numérique.

    La demi-largeur vaut δ/2, donc u_B = (δ/2) / sqrt(3).

    Paramètres
    ----------
    resolution : résolution δ de l'instrument (graduation ou dernier digit).
    """
    return uncertainty_type_B_uniform(half_width=resolution / 2)


def uncertainty_type_exact() -> dict:
    """
    Constante exacte — incertitude-type nulle par définition.

    À utiliser pour les constantes physiques fondamentales (c, e, h...),
    les masses étalons certifiées, ou toute grandeur dont l'incertitude
    est négligeable devant les autres sources.

    Retourne un dict avec u = 0.0 et nu = inf.
    """
    return {
        "u": 0.0,
        "type": "exact",
        "distribution": "constant",
        "nu": float("inf"),
    }


def uncertainty_type_B_relative(u_standard: float, relative_knowledge: float = None) -> dict:
    """
    Incertitude de type B — valeur de l'incertitude-type fournie directement.

    À utiliser quand u est connue par calibration, notice constructeur,
    ou propagée depuis une étape précédente (ex : paramètre de régression).

    Paramètres
    ----------
    u_standard        : incertitude-type u (pas l'incertitude élargie U).
                        Attention : entrer u = U/k, pas U directement.
    relative_knowledge: connaissance relative sur u (0 < r ≤ 1).
                        Si fourni, les degrés de liberté effectifs sont
                        estimés par nu = 1 / (2 * r²) (GUM annexe G).
                        Laisser None si inconnu → nu = inf (type B classique).
    """
    if u_standard < 0:
        raise ValueError("u_standard doit être positif ou nul.")
    nu = float("inf")
    if relative_knowledge is not None and relative_knowledge > 0:
        nu = 1.0 / (2.0 * relative_knowledge ** 2)
    return {
        "u": u_standard,
        "type": "B",
        "distribution": "general",
        "nu": nu,
    }


def calculate_uncertainty(
    formula_str: str,
    variable_names: list[str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, dict],
) -> dict:
    """
    Propagation GUM par dérivation symbolique (SymPy).

    Calcule la valeur nominale du mesurande, les coefficients de sensibilité,
    la variance composée et le budget d'incertitudes.

    Paramètres
    ----------
    formula_str       : expression SymPy du mesurande (ex: "U / I").
    variable_names    : liste ordonnée des noms de variables.
    nominal_values    : dict {nom: valeur nominale}.
    uncertainty_inputs: dict {nom: dict renvoyé par uncertainty_type_*}.

    Retourne un dict avec :
      result          — valeur nominale du mesurande
      uc              — incertitude-type composée
      uc_squared      — variance composée
      sensitivities   — dict {nom: c_i} coefficients de sensibilité
      contributions   — dict {nom: c_i² * u_i²} contributions à la variance
      budget          — dict {nom: % de la variance composée}
      partial_derivs  — dict {nom: expression SymPy de ∂f/∂x_i}
    """
    symbols = {name: sp.Symbol(name) for name in variable_names}
    formula = sp.sympify(formula_str, locals=symbols)
    subs = [(symbols[name], nominal_values[name]) for name in variable_names]

    result = float(formula.subs(subs))

    partial_derivs = {}
    sensitivities = {}
    for name in variable_names:
        dp = sp.diff(formula, symbols[name])
        partial_derivs[name] = dp
        sensitivities[name] = float(dp.subs(subs))

    contributions = {}
    uc_squared = 0.0
    for name in variable_names:
        ci = sensitivities[name]
        ui = uncertainty_inputs[name]["u"]
        contrib = ci ** 2 * ui ** 2
        contributions[name] = contrib
        uc_squared += contrib

    uc = math.sqrt(uc_squared)

    budget = {}
    for name in variable_names:
        budget[name] = 100.0 * contributions[name] / uc_squared if uc_squared > 0 else 0.0

    return {
        "result": result,
        "uc": uc,
        "uc_squared": uc_squared,
        "sensitivities": sensitivities,
        "contributions": contributions,
        "budget": budget,
        "partial_derivs": partial_derivs,
    }


def welch_satterthwaite(
    variable_names: list[str],
    sensitivities: dict[str, float],
    uncertainty_inputs: dict[str, dict],
    uc_squared: float,
) -> dict:
    """
    Formule de Welch-Satterthwaite — degrés de liberté effectifs et facteur k.

    ν_eff = u_c⁴(y) / Σ [ (c_i · u_i)⁴ / ν_i ]

    Ne somme que les contributions dont ν_i est fini (types A ou B avec
    connaissance relative fournie). Si aucune telle contribution n'existe,
    retourne ν_eff = inf et k = 2.0.
    """
    denominator = 0.0
    has_finite_nu = False
    for name in variable_names:
        ci   = sensitivities[name]
        ui   = uncertainty_inputs[name]["u"]
        nu_i = uncertainty_inputs[name].get("nu", float("inf"))
        ui_y = ci * ui
        if nu_i != float("inf") and abs(ui_y) > 0:
            denominator  += (ui_y ** 4) / nu_i
            has_finite_nu = True

    if not has_finite_nu or denominator == 0.0:
        nu_eff = float("inf")
        k      = 2.0
    else:
        nu_eff = (uc_squared ** 2) / denominator
        k      = scipy_stats.t.ppf(0.975, df=nu_eff)
        k      = round(k, 3)

    return {"nu_eff": nu_eff, "k": k}


def expanded_uncertainty(uc: float, k: float = 2.0) -> float:
    """Incertitude élargie U = k · u_c."""
    return k * uc


def linear_regression(x_data: list[float], y_data: list[float]) -> dict:
    """
    Régression linéaire par moindres carrés ordinaires.

    Modèle : y = θ₀ + θ₁ · x

    Les incertitudes u(θ₀) et u(θ₁) sont dérivées de la variance résiduelle
    s²_res = Σ(y_i - ŷ_i)² / (N - 2).

    Paramètres
    ----------
    x_data, y_data : listes de même longueur (N ≥ 3).

    Retourne un dict avec :
      theta0, theta1     — estimateurs OLS
      u_theta0, u_theta1 — incertitudes-types associées
      s_res              — écart-type résiduel
      r2                 — coefficient de détermination R²
      N, nu              — nombre de points et degrés de liberté (N - 2)
      y_pred, residuals  — prédictions et résidus
    """
    N = len(x_data)
    if N != len(y_data):
        raise ValueError("x_data et y_data doivent avoir la même longueur.")
    if N < 3:
        raise ValueError("La régression linéaire requiert au moins 3 points.")

    sum_x  = sum(x_data)
    sum_y  = sum(y_data)
    sum_x2 = sum(x ** 2 for x in x_data)
    sum_xy = sum(x * y for x, y in zip(x_data, y_data))

    det = N * sum_x2 - sum_x ** 2
    if det == 0:
        raise ValueError("Déterminant nul : toutes les valeurs xi sont identiques.")

    theta1 = (N * sum_xy - sum_x * sum_y) / det
    theta0 = (sum_x2 * sum_y - sum_x * sum_xy) / det

    y_pred    = [theta0 + theta1 * x for x in x_data]
    residuals = [y - yp for y, yp in zip(y_data, y_pred)]
    s2_res    = sum(r ** 2 for r in residuals) / (N - 2)

    u_theta1 = math.sqrt(s2_res * N / det)
    u_theta0 = math.sqrt(s2_res * sum_x2 / det)

    mean_y = sum_y / N
    ss_tot = sum((y - mean_y) ** 2 for y in y_data)
    ss_res = sum(r ** 2 for r in residuals)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "theta0": theta0,
        "theta1": theta1,
        "u_theta0": u_theta0,
        "u_theta1": u_theta1,
        "s_res": math.sqrt(s2_res),
        "r2": r2,
        "N": N,
        "y_pred": y_pred,
        "residuals": residuals,
        "nu": N - 2,
    }


def round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Arrondit value à sig_figs chiffres significatifs."""
    if value == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    factor    = 10 ** (sig_figs - 1 - magnitude)
    return round(value * factor) / factor


def format_result(result: float, U: float) -> dict:
    """
    Arrondit le résultat et l'incertitude élargie de façon cohérente.

    L'incertitude élargie U est arrondie à 2 chiffres significatifs.
    Le résultat est arrondi au même ordre de grandeur que U.

    Retourne un dict avec :
      result   — valeur nominale arrondie (float)
      U        — incertitude élargie arrondie à 2 chiffres sig
      decimals — nombre de décimales utiles (pour affichage cohérent)

    FIX : decimals est désormais calculé uniquement à partir de mag_U,
    ce qui évite les valeurs aberrantes quand mag_U > 0.
    """
    U_rounded = round_to_sig_figs(U, 2)
    if U_rounded == 0:
        return {"result": result, "U": 0.0, "decimals": 2}

    mag_U          = math.floor(math.log10(abs(U_rounded)))
    decimals       = max(0, -mag_U + 1)
    factor         = 10 ** mag_U
    result_rounded = math.floor(result / factor + 0.5) * factor

    return {
        "result": result_rounded,
        "U": U_rounded,
        "decimals": decimals,
    }


def full_gum_analysis(
    formula_str: str,
    variable_names: list[str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, dict],
    k_override: float = None,
) -> dict:
    """
    Pipeline GUM complet : propagation + Welch-Satterthwaite + formatage.

    Paramètres
    ----------
    formula_str       : expression SymPy du mesurande.
    variable_names    : liste ordonnée des noms de variables.
    nominal_values    : dict {nom: valeur nominale}.
    uncertainty_inputs: dict {nom: dict renvoyé par uncertainty_type_*}.
    k_override        : facteur d'élargissement forcé (optionnel).
                        Si None, k est déterminé par Welch-Satterthwaite.

    Retourne le dict de calculate_uncertainty enrichi de :
      nu_eff         — degrés de liberté effectifs
      k              — facteur d'élargissement retenu
      U              — incertitude élargie U = k · u_c
      result_rounded — valeur nominale arrondie
      U_rounded      — U arrondie à 2 chiffres significatifs
      decimals       — nombre de décimales pour affichage cohérent
    """
    calc = calculate_uncertainty(
        formula_str, variable_names, nominal_values, uncertainty_inputs
    )
    ws = welch_satterthwaite(
        variable_names,
        calc["sensitivities"],
        uncertainty_inputs,
        calc["uc_squared"],
    )
    k         = k_override if k_override is not None else ws["k"]
    U         = expanded_uncertainty(calc["uc"], k)
    formatted = format_result(calc["result"], U)

    return {
        **calc,
        "nu_eff": ws["nu_eff"],
        "k": k,
        "U": U,
        "result_rounded": formatted["result"],
        "U_rounded": formatted["U"],
        "decimals": formatted["decimals"],
    }


# ============================================================
# PARTIE 2 — EXPORT LATEX
# ============================================================

def _sci(value: float, sig: int = 2) -> str:
    """
    Formate value en notation scientifique LaTeX (×10^n).
    Pour les valeurs dans [-2, 3] en ordre de grandeur, utilise la notation
    décimale standard. En dehors, utilise a × 10^n.

    FIX : gestion correcte des valeurs négatives en notation scientifique.
    """
    if value == 0:
        return "0"
    sign     = "-" if value < 0 else ""
    mag      = math.floor(math.log10(abs(value)))
    if -2 <= mag <= 3:
        decimals = max(0, sig - 1 - mag)
        return f"{value:.{decimals}f}"
    mantissa = abs(value) / (10 ** mag)
    mantissa = round(mantissa, sig - 1)
    if mantissa == int(mantissa):
        mantissa_str = str(int(mantissa))
    else:
        mantissa_str = f"{mantissa:.{sig-1}f}"
    return rf"{sign}{mantissa_str} \times 10^{{{mag}}}"


def _si(value: float, unit: str, sig: int = 3) -> str:
    """
    Formate value avec son unité pour le package siunitx : \\SI{val}{unit}.
    Pour les valeurs hors de [-2, 3] en ordre de grandeur, utilise la
    notation scientifique compatible siunitx (ex: 1.50e-5).

    FIX : gestion correcte des valeurs négatives en notation scientifique.
    """
    if value == 0:
        return rf"\SI{{0}}{{{unit}}}" if unit else "0"
    sign = "-" if value < 0 else ""
    mag  = math.floor(math.log10(abs(value)))
    if -2 <= mag <= 3:
        decimals = max(0, sig - 1 - mag)
        val_str  = f"{value:.{decimals}f}"
    else:
        mantissa = abs(value) / (10 ** mag)
        mantissa = round(mantissa, sig - 1)
        val_str  = f"{sign}{mantissa:.{sig-1}f}e{mag}"
    if unit:
        return rf"\SI{{{val_str}}}{{{unit}}}"
    else:
        return val_str


def _unit_tex(unit_str: str) -> str:
    """Formate une unité pour affichage inline en mode texte LaTeX."""
    if unit_str:
        return rf"\,\text{{{unit_str}}}"
    return ""


def generate_bilan(
    measurand_name: str,
    measurand_symbol: str,
    formula_str: str,
    variable_names: list[str],
    variable_symbols: dict[str, str],
    variable_units: dict[str, str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, dict],
    measurand_unit: str = "",
    k_override: float = None,
    subsection: bool = True,
    global_sig_figs: int = 3,
) -> str:
    """
    Génère le bloc LaTeX complet du bilan d'incertitudes pour un mesurande.

    Le bloc inclut : modèle de mesure, valeurs nominales, incertitudes-types
    source par source, coefficients de sensibilité, Welch-Satterthwaite si
    nécessaire, budget d'incertitudes, et résultat encadré.

    Paramètres
    ----------
    measurand_name   : nom en toutes lettres du mesurande (ex: "Résistance").
    measurand_symbol : symbole LaTeX du mesurande (ex: "R").
    formula_str      : expression SymPy du mesurande.
    variable_names   : liste ordonnée des noms de variables Python.
    variable_symbols : dict {nom_python: symbole_LaTeX}.
    variable_units   : dict {nom_python: unité_siunitx}.
    nominal_values   : dict {nom_python: valeur nominale}.
    uncertainty_inputs: dict {nom_python: dict uncertainty_type_*}.
    measurand_unit   : unité siunitx du mesurande (ex: r"\\ohm").
    k_override       : facteur k forcé (None = Welch-Satterthwaite auto).
    subsection       : si True, génère un \\subsection{} en tête.
    global_sig_figs  : nombre de chiffres significatifs pour l'affichage
                       des valeurs intermédiaires (défaut 3).
    """
    res      = full_gum_analysis(
        formula_str, variable_names, nominal_values, uncertainty_inputs, k_override
    )
    sym_map  = {n: variable_symbols.get(n, n) for n in variable_names}
    unit_map = {n: variable_units.get(n, "")  for n in variable_names}
    lines    = []

    if subsection:
        lines.append(rf"\subsection{{Bilan --- Grandeur ${measurand_symbol}$}}")
    lines.append("")

    formula_sympy = sp.sympify(formula_str, locals={n: sp.Symbol(n) for n in variable_names})
    for n in variable_names:
        formula_sympy = formula_sympy.subs(sp.Symbol(n), sp.Symbol(sym_map[n]))
    formula_latex = sp.latex(formula_sympy)

    lines.append(rf"Le mesurande ${measurand_symbol}$ est lié aux grandeurs d'entrée par le modèle :")
    lines.append(r"\[")
    lines.append(rf"    {measurand_symbol} = {formula_latex}")
    lines.append(r"\]")

    defs = []
    for n in variable_names:
        unit = unit_map[n]
        val  = nominal_values[n]
        defs.append(rf"${sym_map[n]} = {_si(val, unit, sig=global_sig_figs)}$")
    nom_val = res["result"]
    lines.append(
        r"avec " + " et ".join(defs)
        + rf", ce qui donne la valeur nominale "
        rf"${measurand_symbol}_{{\mathrm{{nom}}}} = {_si(nom_val, measurand_unit, sig=global_sig_figs)}$."
    )
    lines.append(r"\newline")
    lines.append("")

    for n in variable_names:
        inp   = uncertainty_inputs[n]
        sym   = sym_map[n]
        unit  = unit_map[n]
        u_val = inp["u"]

        if inp["type"] == "exact":
            lines.append(
                rf"La grandeur ${sym}$ est une constante exacte déterminée par définition. "
                rf"Son incertitude-type est nulle : $u({sym}) = 0$."
            )
        elif inp["type"] == "A":
            N_mes = inp["N"]
            s     = inp["s"]
            lines.append(
                rf"La répétition de $N = {N_mes}$ mesures sur ${sym}$ fournit "
                rf"une incertitude de type~A :"
            )
            lines.append(r"\[")
            lines.append(
                rf"    u_A(\bar{{{sym}}}) = \frac{{s_{{{sym}}}}}{{\sqrt{{{N_mes}}}}}"
                rf" = \frac{{{_si(s, unit, sig=global_sig_figs)}}}{{\sqrt{{{N_mes}}}}}"
                rf" = {_si(u_val, unit, sig=global_sig_figs)}"
            )
            lines.append(r"\]")
        else:
            dist = inp.get("distribution", "uniform")
            if dist == "uniform":
                a     = inp.get("a", u_val * math.sqrt(3))
                delta = a * 2
                lines.append(
                    rf"La grandeur ${sym}$ est issue d'une lecture unique sur un instrument "
                    rf"de résolution $\delta = {_si(delta, unit, sig=global_sig_figs)}$, "
                    rf"ce qui conduit à une incertitude de type~B :"
                )
                lines.append(r"\[")
                lines.append(
                    rf"    u_B({sym}) = \frac{{\delta/2}}{{\sqrt{{3}}}}"
                    rf" = \frac{{{_si(a, unit, sig=global_sig_figs)}}}{{\sqrt{{3}}}}"
                    rf" = {_si(u_val, unit, sig=global_sig_figs)}"
                )
                lines.append(r"\]")
            else:
                lines.append(
                    rf"L'incertitude-type de type~B sur ${sym}$ est estimée directement à :"
                )
                lines.append(r"\[")
                lines.append(rf"    u_B({sym}) = {_si(u_val, unit, sig=global_sig_figs)}")
                lines.append(r"\]")
        lines.append("")

    lines.append(r"\newline")
    lines.append(r"Les coefficients de sensibilité, évalués aux valeurs nominales, sont :")
    lines.append(r"\[")
    ci_terms = []
    for n in variable_names:
        ci     = res["sensitivities"][n]
        dp_sym = res["partial_derivs"][n]
        for nn in variable_names:
            dp_sym = dp_sym.subs(sp.Symbol(nn), sp.Symbol(sym_map[nn]))
        dp_latex = sp.latex(dp_sym)
        ci_terms.append(
            rf"c_{{{sym_map[n]}}} = \left.\frac{{\partial {measurand_symbol}}}"
            rf"{{\partial {sym_map[n]}}}\right|_{{\mathrm{{nom}}}}"
            rf" = {dp_latex} = {_sci(ci, sig=global_sig_figs)}"
        )
    lines.append(r"    " + r"\qquad ".join(ci_terms))
    lines.append(r"\]")
    lines.append("")

    uc = res["uc"]
    lines.append(r"La propagation des incertitudes pour des grandeurs indépendantes donne :")
    lines.append(r"\[")
    inner_terms = []
    for n in variable_names:
        if uncertainty_inputs[n]["type"] == "exact":
            continue
        ci = res["sensitivities"][n]
        ui = uncertainty_inputs[n]["u"]
        inner_terms.append(
            rf"({_sci(ci, sig=global_sig_figs)})^2 \, ({_si(ui, unit_map[n], sig=global_sig_figs)})^2"
        )
    lines.append(
        rf"    u_c({measurand_symbol}) = \sqrt{{{' + '.join(inner_terms)}}}"
        rf" = {_si(uc, measurand_unit, sig=global_sig_figs)}"
    )
    lines.append(r"\]")
    lines.append("")

    lines.append(r"\newline")
    nu_eff         = res.get("nu_eff", float("inf"))
    k              = res["k"]
    type_A_sources = [n for n in variable_names if uncertainty_inputs[n]["type"] == "A"]
    has_small_A    = any(uncertainty_inputs[n]["N"] < 30 for n in type_A_sources)

    if nu_eff != float("inf") and type_A_sources:
        nu_lines = []
        for n in variable_names:
            if uncertainty_inputs[n]["type"] == "exact":
                continue
            ci   = res["sensitivities"][n]
            ui   = uncertainty_inputs[n]["u"]
            nu_i = uncertainty_inputs[n].get("nu", float("inf"))
            if nu_i == float("inf"):
                nu_lines.append(
                    rf"\dfrac{{({_sci(ci)} \cdot {_si(ui, unit_map[n])})^4}}{{\infty}}"
                )
            else:
                nu_lines.append(
                    rf"\dfrac{{({_sci(ci)} \cdot {_si(ui, unit_map[n])})^4}}{{{int(nu_i)}}}"
                )

        a_descriptions = []
        for n in type_A_sources:
            N_mes = uncertainty_inputs[n]["N"]
            a_descriptions.append(
                rf"${sym_map[n]}$ porte sur $N = {N_mes}$ mesures "
                rf"($\nu_{{{sym_map[n]}}} = {N_mes - 1}$)"
            )
        lines.append(
            "La source " + " et ".join(a_descriptions)
            + r". On détermine le facteur d'élargissement par Welch-Satterthwaite :"
        )
        lines.append(r"\[")
        lines.append(
            rf"    \nu_{{\mathrm{{eff}}}} = \frac{{u_c^4({measurand_symbol})}}"
            rf"{{{' + '.join(nu_lines)}}} \approx {nu_eff:.0f}"
        )
        lines.append(r"\]")
        lines.append(
            rf"Pour $\nu_{{\mathrm{{eff}}}} = {nu_eff:.0f}$ (95\,\%), "
            rf"la table de Student donne $k = {k:.2f}$."
        )
    else:
        lines.append(
            rf"Toutes les composantes significatives sont de type~B ou historiques ; "
            rf"on retient $k = {k:.2f}$ (95\,\%)."
        )
    lines.append("")

    U = res["U"]
    lines.append(r"\[")
    lines.append(
        rf"    U({measurand_symbol}) = k\,u_c({measurand_symbol}) = "
        rf"{k:.2f} \times {_si(uc, measurand_unit, sig=global_sig_figs)} = "
        rf"{_si(U, measurand_unit, sig=2)}"
    )
    lines.append(r"\]")
    lines.append("")

    lines.append(r"\newline")
    lines.append(r"Le budget d'incertitudes :")
    lines.append(r"\[")
    budget_terms = []
    for n in variable_names:
        if uncertainty_inputs[n]["type"] == "exact":
            continue
        pct = res["budget"][n]
        budget_terms.append(
            rf"\frac{{c_{{{sym_map[n]}}}^2\,u^2({sym_map[n]})}}{{u_c^2({measurand_symbol})}} = {pct:.1f}\,\%"
        )
    lines.append(r"    " + r"\qquad ".join(budget_terms))
    lines.append(r"\]")

    non_exact_budget = {
        kk: vv for kk, vv in res["budget"].items()
        if uncertainty_inputs[kk]["type"] != "exact"
    }
    if non_exact_budget:
        dominant = max(non_exact_budget, key=non_exact_budget.get)
        lines.append(
            rf"La mesure de ${sym_map[dominant]}$ contribue à "
            rf"{non_exact_budget[dominant]:.0f}\,\% de la variance composée."
        )
    lines.append("")

    lines.append(r"Le résultat final s'écrit :")
    lines.append(r"\[")
    lines.append(
        rf"    \boxed{{{measurand_symbol} = "
        rf"{_si(res['result_rounded'], measurand_unit, sig=global_sig_figs)} \pm "
        rf"{_si(res['U_rounded'], measurand_unit, sig=2)}}}"
    )
    lines.append(r"\]")

    return "\n".join(lines)


def generate_bilan_regression(
    x_symbol: str,
    y_symbol: str,
    x_unit: str,
    y_unit: str,
    x_data: list[float],
    y_data: list[float],
    subsection_title: str = "Régression linéaire",
    slope_unit: str = "",
    intercept_unit: str = "",
    slope_symbol: str = r"\theta_1",
    intercept_symbol: str = r"\theta_0",
    _reg_precomputed: dict = None,
) -> str:
    """
    Génère le bloc LaTeX du bilan de régression linéaire.

    Paramètres
    ----------
    x_symbol, y_symbol     : symboles LaTeX des axes.
    x_unit, y_unit         : unités siunitx des axes.
    x_data, y_data         : données expérimentales.
    subsection_title       : titre de la sous-section.
    slope_unit             : unité siunitx de la pente (défaut : y_unit/x_unit).
    intercept_unit         : unité siunitx de l'ordonnée à l'origine (défaut : y_unit).
    slope_symbol           : symbole LaTeX de la pente (défaut : r"\\theta_1").
    intercept_symbol       : symbole LaTeX de l'ordonnée à l'origine (défaut : r"\\theta_0").
    _reg_precomputed       : dict issu de linear_regression() déjà calculé (usage interne,
                             évite le double calcul dans full_pipeline_regression_to_measurand).
    """
    reg   = _reg_precomputed if _reg_precomputed is not None else linear_regression(x_data, y_data)
    lines = []

    s_unit = slope_unit if slope_unit else (
        rf"{y_unit}/{x_unit}" if x_unit and y_unit else ""
    )
    i_unit = intercept_unit if intercept_unit else y_unit

    lines.append(rf"\subsection{{Bilan --- {subsection_title}}}")
    lines.append("")
    lines.append(
        rf"On cherche les paramètres de la droite "
        rf"${y_symbol} = {intercept_symbol} + {slope_symbol} \cdot {x_symbol}$ :"
    )
    lines.append(r"\[")
    lines.append(
        rf"    Q({intercept_symbol}, {slope_symbol}) = "
        rf"\sum_{{i=1}}^{{N}} \left(y_i - ({intercept_symbol} + {slope_symbol} x_i)\right)^2"
    )
    lines.append(r"\]")
    lines.append("")

    N = reg["N"]
    lines.append(rf"Avec $N = {N}$ points de mesure, les estimateurs sont :")
    lines.append(r"\[")
    lines.append(rf"    {slope_symbol} = {_si(reg['theta1'], s_unit, sig=3)}")
    lines.append(r"\]")
    lines.append(r"\[")
    lines.append(rf"    {intercept_symbol} = {_si(reg['theta0'], i_unit, sig=3)}")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        rf"Les incertitudes-types associées "
        rf"($s^2_{{\mathrm{{res}}}} = {_sci(reg['s_res']**2, sig=3)}$) sont :"
    )
    lines.append(r"\[")
    lines.append(
        rf"    u({intercept_symbol}) = {_si(reg['u_theta0'], i_unit, sig=2)} "
        rf"\qquad u({slope_symbol}) = {_si(reg['u_theta1'], s_unit, sig=2)}"
    )
    lines.append(r"\]")
    lines.append("")
    lines.append(
        rf"Le coefficient de corrélation $r^2 = {reg['r2']:.4f}$ "
        rf"confirme la qualité de l'ajustement."
    )
    lines.append(r"\newline")
    lines.append("")
    lines.append(r"Le résultat de la régression s'écrit :")
    lines.append(r"\[")
    lines.append(
        rf"    \boxed{{{y_symbol} = "
        rf"\left({_si(reg['theta0'], i_unit, sig=3)} \pm {_si(reg['u_theta0'], i_unit, sig=2)}\right)"
        rf" + \left({_si(reg['theta1'], s_unit, sig=3)} \pm {_si(reg['u_theta1'], s_unit, sig=2)}\right)"
        rf" \cdot {x_symbol}}}"
    )
    lines.append(r"\]")

    return "\n".join(lines)


# ============================================================
# PARTIE 3 — PIPELINE INTÉGRÉ (régression → mesurande)
# ============================================================

def full_pipeline_regression_to_measurand(
    x_data: list[float],
    y_data: list[float],
    x_symbol: str,
    y_symbol: str,
    x_unit: str,
    y_unit: str,
    formula_str: str,
    variable_names: list[str],
    variable_symbols: dict[str, str],
    variable_units: dict[str, str],
    nominal_values_helpers: dict[str, float],
    uncertainty_inputs_helpers: dict[str, dict],
    measurand_symbol: str,
    measurand_unit: str = "",
    slope_unit: str = "",
    intercept_unit: str = "",
    slope_symbol: str = r"\theta_1",
    intercept_symbol: str = r"\theta_0",
) -> str:
    """
    Génère d'un seul appel le LaTeX complet : régression + propagation GUM.

    À utiliser quand le mesurande final dépend d'un paramètre de régression
    (pente ou ordonnée à l'origine) et d'autres grandeurs mesurées séparément.

    Les variables 'theta1' et/ou 'theta0' dans variable_names sont reconnues
    automatiquement et alimentées par la régression. Les autres variables sont
    fournies via nominal_values_helpers et uncertainty_inputs_helpers.

    Paramètres clés
    ---------------
    variable_names            : doit inclure 'theta1' et/ou 'theta0'.
    nominal_values_helpers    : valeurs nominales des variables hors régression.
    uncertainty_inputs_helpers: incertitudes des variables hors régression.
    slope_symbol, intercept_symbol : notation LaTeX (transmise aux deux blocs).

    FIX : la régression n'est calculée qu'une seule fois et transmise à
    generate_bilan_regression via _reg_precomputed.
    """
    reg      = linear_regression(x_data, y_data)
    nominals = {**nominal_values_helpers}
    inputs   = {**uncertainty_inputs_helpers}

    if "theta1" in variable_names:
        nominals["theta1"]      = reg["theta1"]
        inputs["theta1"]        = uncertainty_type_B_relative(reg["u_theta1"])
        inputs["theta1"]["nu"]  = reg["nu"]
    if "theta0" in variable_names:
        nominals["theta0"]      = reg["theta0"]
        inputs["theta0"]        = uncertainty_type_B_relative(reg["u_theta0"])
        inputs["theta0"]["nu"]  = reg["nu"]

    tex_reg = generate_bilan_regression(
        x_symbol, y_symbol, x_unit, y_unit, x_data, y_data,
        slope_unit=slope_unit,
        intercept_unit=intercept_unit,
        slope_symbol=slope_symbol,
        intercept_symbol=intercept_symbol,
        _reg_precomputed=reg,
    )

    tex_gum = generate_bilan(
        measurand_name=measurand_symbol,
        measurand_symbol=measurand_symbol,
        formula_str=formula_str,
        variable_names=variable_names,
        variable_symbols=variable_symbols,
        variable_units=variable_units,
        nominal_values=nominals,
        uncertainty_inputs=inputs,
        measurand_unit=measurand_unit,
        subsection=True,
    )

    return tex_reg + "\n\n\\clearpage\n\n" + tex_gum


def generate_annexe(bilans: list[str]) -> str:
    """
    Assemble la section LaTeX d'annexe à partir d'une liste de bilans.

    Chaque élément de bilans est la chaîne retournée par generate_bilan()
    ou generate_bilan_regression(). L'ordre de la liste détermine l'ordre
    dans le PDF final.

    Exemple d'usage :
        print(generate_annexe([bilan_R, bilan_C, bilan_reg]))
    """
    body = "\n\n".join(bilans)
    return r"\section{Annexe : Bilans d'incertitudes}" + "\n\n" + body