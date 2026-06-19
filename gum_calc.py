"""
gum_calc.py — Moteur de calcul GUM + Export LaTeX
"""

import math
import re
import warnings
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
    if not variable_names:
        raise ValueError("variable_names ne peut pas être vide : un mesurande dépend d'au moins une grandeur.")
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

    # Garde-fou GUM : en grande dynamique (x ou y de plusieurs ordres de
    # grandeur), le calcul de `det` et des résidus peut souffrir d'une
    # cancellation catastrophique (différence de grands nombres proches),
    # produisant un résidu purement numérique sans qu'aucune exception ne
    # soit levée. Une telle valeur n'a aucune signification physique et
    # conduirait à une incertitude u(θ₁)/u(θ₀) artificiellement quasi nulle.
    scale_y = max((y ** 2 for y in y_data), default=0.0)
    if scale_y > 0 and s2_res < 1e-12 * scale_y:
        warnings.warn(
            "linear_regression : la variance résiduelle s2_res est du même "
            "ordre que le bruit de troncature flottante (cancellation "
            "catastrophique probable sur des données de grande dynamique). "
            "L'incertitude sur theta0/theta1 qui en découle n'a "
            "probablement aucune signification physique — vérifier "
            "l'échelle des données en entrée.",
            RuntimeWarning,
        )

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


def format_result(result: float, U: float, sig_figs_exact: int = 4) -> dict:
    """
    Arrondit le résultat et l'incertitude élargie de façon cohérente.

    Si U_rounded == 0 (toutes les grandeurs d'entrée sont exactes), le
    résultat est tout de même arrondi à `sig_figs_exact` chiffres
    significatifs : sans cela, tout code consultant directement
    `result_rounded` (notebook, print de débogage...) sans repasser par
    `_format_result_uncertainty` recevrait la précision flottante brute.
    """
    U_rounded = round_to_sig_figs(U, 2)
    if U_rounded == 0:
        result_rounded = round_to_sig_figs(result, sig_figs_exact) if result != 0 else 0.0
        return {"result": result_rounded, "U": 0.0, "decimals": 0}

    mag_U    = math.floor(math.log10(abs(U_rounded)))
    decimals = -mag_U + 1
    factor   = 10 ** (mag_U - 1)

    if result >= 0:
        result_rounded = math.floor(result / factor + 0.5) * factor
    else:
        result_rounded = -math.floor(-result / factor + 0.5) * factor

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

def _mantissa_exp(value: float, sig: int):
    sign     = "-" if value < 0 else ""
    mag      = math.floor(math.log10(abs(value)))
    mantissa = round(abs(value) / (10 ** mag), sig - 1)
    if mantissa >= 10:
        mantissa /= 10
        mag      += 1
    return sign, mantissa, mag


def _rename_symbols(expr, variable_names: list[str], sym_map: dict[str, str]):
    """
    Renomme les symboles Python d'une expression SymPy par leurs symboles
    LaTeX (sym_map), en une seule substitution simultanée.

    Une boucle de `.subs()` appliqués un par un serait incorrecte dès que
    deux variables ont des symboles LaTeX croisés (ex : sym_map = {'x':
    'y', 'y': 'x'}) : la substitution déjà appliquée pour 'x' serait
    recapturée par la substitution suivante pour 'y'. `simultaneous=True`
    applique les deux remplacements en une seule passe et évite la
    collision.
    """
    subs_pairs = [(sp.Symbol(n), sp.Symbol(sym_map[n])) for n in variable_names]
    return expr.subs(subs_pairs, simultaneous=True)


def _latex_ln(expr) -> str:
    """
    Rend une expression SymPy en LaTeX en forçant la notation `\\ln` pour
    le logarithme népérien.

    `sympy.log(x)` est le logarithme népérien, mais son rendu LaTeX par
    défaut est `\\log` (sans base explicite), ce qui peut induire en erreur
    un lecteur habitué à la convention « log = base 10, ln = base e ».
    Vérifié empiriquement : même `sympy.log(x, 10)` (base explicite) est
    décomposé en interne par SymPy en `log(x)/log(10)` — il n'existe pas
    de forme `\\log_{10}{...}` produite par le printer LaTeX de SymPy.
    La substitution globale `\\log{` -> `\\ln{` est donc sans risque de
    faux positif sur une base explicite.
    """
    return sp.latex(expr).replace(r"\log{", r"\ln{")


def _sci(value: float, sig: int = 2) -> str:
    """
    Alias historique conservé pour compatibilité d'appel : délègue
    entièrement à `_num`. Auparavant `_sci` retournait une chaîne brute
    (jamais encapsulée dans `\\num{}`), ce qui produisait un séparateur
    décimal point au lieu de virgule quel que soit le réglage de locale —
    notamment pour tous les coefficients de sensibilité. La délégation à
    `_num` corrige ce point sans dupliquer la logique de formatage.
    """
    return _num(value, sig=sig)


def _format_magnitude(value: float, sig: int) -> str:
    """
    Formate la magnitude numérique d'une valeur (sans \\num{} ni \\SI{}
    autour) : notation décimale pour -2 <= magnitude <= 3, notation
    scientifique mantisse*10^exposant sinon.

    Helper commun à `_num` et `_si`, qui ne diffèrent que par l'habillage
    final (\\num{} ou \\SI{}{unité}). Avant cette extraction, les deux
    fonctions dupliquaient le même algorithme d'arrondi/magnitude — la
    correction de `_sci` (voir commentaire ci-dessus) avait déjà montré
    qu'une telle duplication finit par se désynchroniser silencieusement.
    """
    value = round_to_sig_figs(value, sig)
    sign  = "-" if value < 0 else ""
    mag   = math.floor(math.log10(abs(value)))
    if -2 <= mag <= 3:
        decimals = max(0, sig - 1 - mag)
        return f"{value:.{decimals}f}"
    _, mantissa, mag = _mantissa_exp(value, sig)
    return f"{sign}{mantissa:.{sig-1}f}e{mag}"


def _num(value: float, sig: int = 3) -> str:
    if value == 0:
        return r"\num{0}"
    return rf"\num{{{_format_magnitude(value, sig)}}}"


def _si(value: float, unit: str, sig: int = 3) -> str:
    if value == 0:
        return rf"\SI{{0}}{{{unit}}}" if unit else r"\num{0}"
    if not unit:
        return _num(value, sig=sig)
    return rf"\SI{{{_format_magnitude(value, sig)}}}{{{unit}}}"


def _format_result_uncertainty(
    result_rounded: float,
    U_rounded: float,
    decimals: int,
    unit: str,
    sig_figs_exact: int = 4,
) -> str:
    if U_rounded == 0:
        val = round_to_sig_figs(result_rounded, sig_figs_exact) if result_rounded != 0 else 0.0
        return _si(val, unit, sig=sig_figs_exact) if unit else _num(val, sig=sig_figs_exact)

    mag_result = math.floor(math.log10(abs(result_rounded))) if result_rounded != 0 else 0

    if -2 <= mag_result <= 3:
        d       = max(0, decimals)
        val_str = f"{result_rounded:.{d}f}"
        u_str   = f"{U_rounded:.{d}f}"
        body    = rf"{val_str} +- {u_str}"
    else:
        d       = max(0, decimals + mag_result)
        scale   = 10 ** mag_result
        val_str = f"{result_rounded / scale:.{d}f}"
        u_str   = f"{U_rounded / scale:.{d}f}"
        body    = rf"{val_str} +- {u_str} e{mag_result}"

    if unit:
        return rf"\SI{{{body}}}{{{unit}}}"
    return rf"\num{{{body}}}"


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
    formula_sympy = _rename_symbols(formula_sympy, variable_names, sym_map)
    formula_latex = _latex_ln(formula_sympy)

    # measurand_name est rappelé entre parenthèses plutôt qu'inséré comme
    # sujet de la phrase ("La résistance R est liée..."), pour éviter de
    # devoir déduire automatiquement le genre grammatical du nom français
    # (masculin/féminin) et l'accord du participe qui en découlerait. Si
    # measurand_name n'est pas fourni, ou si l'appelant interne réutilise
    # le symbole comme nom (cf. full_pipeline_regression_to_measurand), la
    # parenthèse est omise pour ne pas afficher une redondance du type
    # "$R$ ($R$)".
    name_clause = ""
    if measurand_name and measurand_name.strip().lower() != measurand_symbol.strip().lower():
        name_clause = f" ({measurand_name})"

    lines.append(rf"\noindent Le mesurande ${measurand_symbol}${name_clause} est lié aux grandeurs d'entrée par le modèle :")
    lines.append(r"\[")
    lines.append(rf"    {measurand_symbol} = {formula_latex}")
    lines.append(r"\]")

    defs = []
    for n in variable_names:
        unit = unit_map[n]
        val  = nominal_values[n]
        defs.append(rf"${sym_map[n]} = {_si(val, unit, sig=global_sig_figs)}$")
    nom_val = res["result"]
    if len(defs) > 1:
        defs_str = ", ".join(defs[:-1]) + " et " + defs[-1]
    else:
        defs_str = defs[0]
    
    lines.append(
        r"\noindent avec " + defs_str
        + rf", ce qui donne la valeur nominale "
        rf"$\left.{measurand_symbol}\right|_{{\mathrm{{nom}}}} = {_si(nom_val, measurand_unit, sig=global_sig_figs)}$."
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
                rf"\noindent La grandeur ${sym}$ est une constante exacte déterminée par définition. "
                rf"Son incertitude-type est nulle : $u({sym}) = 0$."
            )
        elif inp["type"] == "A":
            N_mes     = inp["N"]
            s         = inp["s"]
            sym_mean  = sym if sym.strip().startswith(r"\bar{") else rf"\bar{{{sym}}}"
            lines.append(
                rf"\noindent La répétition de $N = {N_mes}$ mesures sur ${sym}$ fournit "
                rf"une incertitude de type~A :"
            )
            lines.append(r"\[")
            lines.append(
                rf"    u_A({sym_mean}) = \frac{{s_{{{sym}}}}}{{\sqrt{{{N_mes}}}}}"
                rf" = \frac{{{_num(s, sig=global_sig_figs)}}}{{\sqrt{{{N_mes}}}}}"
                rf" = {_si(u_val, unit, sig=global_sig_figs)}"
            )
            lines.append(r"\]")
        elif inp["type"] == "B":
            dist = inp.get("distribution", "uniform")
            if dist == "uniform":
                a     = inp.get("a", u_val * math.sqrt(3))
                delta = a * 2
                lines.append(
                    rf"\noindent La grandeur ${sym}$ est issue d'une lecture unique sur un instrument "
                    rf"de résolution $\delta = {_si(delta, unit, sig=global_sig_figs)}$, "
                    rf"ce qui conduit à une incertitude de type~B :"
                )
                lines.append(r"\[")
                lines.append(
                    rf"    u_B({sym}) = \frac{{\delta/2}}{{\sqrt{{3}}}}"
                    rf" = \frac{{{_num(a, sig=global_sig_figs)}}}{{\sqrt{{3}}}}"
                    rf" = {_si(u_val, unit, sig=global_sig_figs)}"
                )
                lines.append(r"\]")
            else:
                lines.append(
                    rf"\noindent L'incertitude-type de type~B sur ${sym}$ est estimée directement à :"
                )
                lines.append(r"\[")
                lines.append(rf"    u_B({sym}) = {_si(u_val, unit, sig=global_sig_figs)}")
                lines.append(r"\]")
        else:
            raise ValueError(
                f"Type d'incertitude inconnu pour la variable '{n}' : {inp['type']!r}. "
                "Attendu 'exact', 'A' ou 'B' (voir uncertainty_type_* dans la Partie 1)."
            )
        lines.append("")

    lines.append(r"\noindent Les coefficients de sensibilité, évalués aux valeurs nominales, sont :")
    ci_terms = []
    for n in variable_names:
        ci     = res["sensitivities"][n]
        dp_sym = res["partial_derivs"][n]
        dp_sym = _rename_symbols(dp_sym, variable_names, sym_map)
        dp_latex = _latex_ln(dp_sym)
        ci_terms.append(
            rf"c_{{{sym_map[n]}}} = \left.\frac{{\partial {measurand_symbol}}}"
            rf"{{\partial {sym_map[n]}}}\right|_{{\mathrm{{nom}}}}"
            rf" = {dp_latex} = {_sci(ci, sig=global_sig_figs)}"
        )

    ci_line = r"\qquad ".join(ci_terms)
    if len(ci_terms) > 3 or len(ci_line) > 90:
        lines.append(r"\begin{align*}")
        for i, term in enumerate(ci_terms):
            suffix = r" \\" if i < len(ci_terms) - 1 else ""
            term_aligned = term.replace(" = ", " &= ", 1)
            lines.append(rf"    {term_aligned}{suffix}")
        lines.append(r"\end{align*}")
    else:
        lines.append(r"\[")
        lines.append(r"    " + ci_line)
        lines.append(r"\]")
    lines.append("")

    uc = res["uc"]
    inner_terms = []
    for n in variable_names:
        if uncertainty_inputs[n]["type"] == "exact":
            continue
        ci = res["sensitivities"][n]
        ui = uncertainty_inputs[n]["u"]
        inner_terms.append(
            rf"({_sci(ci, sig=global_sig_figs)})^2 \, ({_num(ui, sig=global_sig_figs)})^2"
        )

    if not inner_terms:
        lines.append(
            rf"\noindent Toutes les grandeurs d'entrée étant des constantes exactes, "
            rf"l'incertitude-type composée est nulle : $u_c({measurand_symbol}) = 0$."
        )
        lines.append(r"\newline")
    else:
        lines.append(r"\noindent La propagation des incertitudes pour des grandeurs indépendantes donne :")
        radicand   = " + ".join(inner_terms)
        result_str = _si(uc, measurand_unit, sig=global_sig_figs)

        if len(inner_terms) > 1 and (len(radicand) > 70 or len(inner_terms) >= 3):
            S_val = uc ** 2
            n_terms = len(inner_terms)
            n_lines = math.ceil(n_terms / 3)
            terms_per_line = math.ceil(n_terms / n_lines)

            lines.append(r"\begin{align*}")
            for i in range(0, n_terms, terms_per_line):
                chunk  = inner_terms[i:i+terms_per_line]
                joined = " + ".join(chunk)
                prefix = r"    S &= " if i == 0 else r"    &\quad + "
                lines.append(rf"{prefix}{joined} \\")
            lines.append(rf"    &= {_num(S_val, sig=global_sig_figs)}")
            lines.append(r"\end{align*}")
            lines.append(r"\[")
            lines.append(
                rf"    u_c({measurand_symbol}) = \sqrt{{S}} = {result_str}"
            )
            lines.append(r"\]")
        else:
            lines.append(r"\[")
            lines.append(
                rf"    u_c({measurand_symbol}) = \sqrt{{{radicand}}}"
                rf" = {result_str}"
            )
            lines.append(r"\]")
        lines.append(r"\newline")

    nu_eff = res.get("nu_eff", float("inf"))
    k      = res["k"]

    if nu_eff != float("inf"):
        nu_lines = []
        descriptions = []
        for n in variable_names:
            if uncertainty_inputs[n]["type"] == "exact":
                continue
            ci   = res["sensitivities"][n]
            ui   = uncertainty_inputs[n]["u"]
            nu_i = uncertainty_inputs[n].get("nu", float("inf"))
            # nu_i (typiquement 1/(2r²) pour une connaissance relative) n'est
            # généralement pas entier. On tronque par int() plutôt que
            # d'arrondir au plus proche, par choix délibéré et conservateur :
            # un nu_i tronqué (donc plus petit) ne peut que réduire nu_eff et
            # augmenter k, jamais sous-estimer l'incertitude finale. Le
            # nu_eff global affiché juste au-dessus, lui, est arrondi au plus
            # proche ({nu_eff:.0f}) car il n'entre dans aucun calcul ultérieur.
            
            if nu_i == float("inf"):
                nu_lines.append(rf"\dfrac{{({_sci(ci, sig=global_sig_figs)} \cdot {_num(ui, sig=global_sig_figs)})^4}}{{\infty}}")
            else:
                nu_lines.append(rf"\dfrac{{({_sci(ci, sig=global_sig_figs)} \cdot {_num(ui, sig=global_sig_figs)})^4}}{{{int(nu_i)}}}")
                type_str = "type~A" if uncertainty_inputs[n]["type"] == "A" else "type~B"
                descriptions.append(rf"${sym_map[n]}$ ({type_str}, $\nu = {int(nu_i)}$)")

        desc_str = ", ".join(descriptions[:-1]) + " et " + descriptions[-1] if len(descriptions) > 1 else descriptions[0]
        lines.append(
            rf"\noindent Les degrés de liberté étant finis pour {desc_str}, "
            r"on détermine le facteur d'élargissement par la formule de Welch-Satterthwaite :"
        )
        lines.append(r"\[")
        lines.append(
            rf"    \nu_{{\mathrm{{eff}}}} = \frac{{u_c^4({measurand_symbol})}}"
            rf"{{{' + '.join(nu_lines)}}} \approx {nu_eff:.0f}"
        )
        lines.append(r"\]")
        lines.append(
            rf"\noindent Pour $\nu_{{\mathrm{{eff}}}} = {nu_eff:.0f}$ (95\,\%), "
            rf"la table de Student donne $k = \num{{{k:.2f}}}$."
        )
    else:
        lines.append(
            rf"\noindent Toutes les composantes configurées possèdent des degrés de liberté infinis, "
            rf"on retient la standardisation classique $k = \num{{{k:.2f}}}$ (95\,\%)."
        )
    lines.append("")

    U = res["U"]
    lines.append(r"\[")
    lines.append(
        rf"    U({measurand_symbol}) = k\,u_c({measurand_symbol}) = "
        rf"\num{{{k:.2f}}} \times {_num(uc, sig=global_sig_figs)} = "
        rf"{_si(U, measurand_unit, sig=2)}"
    )
    lines.append(r"\]")
    lines.append(r"\newline")

    non_exact_budget = {
        kk: vv for kk, vv in res["budget"].items()
        if uncertainty_inputs[kk]["type"] != "exact"
    }
    
    if non_exact_budget:
        lines.append(r"\noindent Le budget d'incertitudes :")
        budget_terms = []
        for n in variable_names:
            if uncertainty_inputs[n]["type"] == "exact":
                continue
            pct = res["budget"][n]
            budget_terms.append(
                rf"\frac{{c_{{{sym_map[n]}}}^2\,u^2({sym_map[n]})}}{{u_c^2({measurand_symbol})}} = \num{{{pct:.1f}}}\,\%"
            )
        budget_line = r"\qquad ".join(budget_terms)
        if len(budget_terms) > 3 or len(budget_line) > 90:
            lines.append(r"\begin{align*}")
            for i, term in enumerate(budget_terms):
                suffix = r" \\" if i < len(budget_terms) - 1 else ""
                term_aligned = term.replace(" = ", " &= ", 1)
                lines.append(rf"    {term_aligned}{suffix}")
            lines.append(r"\end{align*}")
        else:
            lines.append(r"\[")
            lines.append(r"    " + budget_line)
            lines.append(r"\]")

        dominant = max(non_exact_budget, key=non_exact_budget.get)
        lines.append(
            rf"\noindent La mesure de ${sym_map[dominant]}$ contribue à "
            rf"{non_exact_budget[dominant]:.0f}\,\% de la variance composée."
        )
    lines.append("")

    lines.append(r"\noindent Le résultat final s'écrit :")
    lines.append(r"\[")
    final_expr = _format_result_uncertainty(
        result_rounded=res['result_rounded'],
        U_rounded=res['U_rounded'],
        decimals=res['decimals'],
        unit=measurand_unit,
        sig_figs_exact=global_sig_figs
    )
    lines.append(rf"    \boxed{{{measurand_symbol} = {final_expr}}}")
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
    measurand_name: str = "",
    measurand_unit: str = "",
    slope_unit: str = "",
    intercept_unit: str = "",
    slope_symbol: str = r"\theta_1",
    intercept_symbol: str = r"\theta_0",
) -> str:
    """
    Génère d'un seul appel le LaTeX complet : régression + propagation GUM.
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
        measurand_name=measurand_name or measurand_symbol,
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


# --- Algèbre d'unités siunitx (numérateur/dénominateur), pour diviser
# proprement une unité y par une unité x sans concaténation textuelle ---

_SI_PREFIX_TOKENS = {
    r"\quecto", r"\ronto", r"\yocto", r"\zepto", r"\atto", r"\femto",
    r"\pico", r"\nano", r"\micro", r"\milli", r"\centi", r"\deci",
    r"\deca", r"\hecto", r"\kilo", r"\mega", r"\giga", r"\tera",
    r"\peta", r"\exa", r"\zetta", r"\yotta",
}
_SI_POWER_TOKENS = {r"\square": 2, r"\cubic": 3}
_SI_PER = r"\per"


def _parse_unit(unit_str: str) -> tuple[list, list]:
    """
    Découpe une unité siunitx en deux listes [nom, puissance] : numérateur
    (avant \\per) et dénominateur (après \\per). Les préfixes (\\kilo,
    \\milli...) sont rattachés au token d'unité qui suit pour former un
    seul nom (ex: \\kilo\\gram -> "\\kilo\\gram"), et les modificateurs de
    puissance (\\square, \\cubic) sont convertis en exposant entier.
    """
    tokens = re.findall(r"\\[A-Za-z]+", unit_str or "")
    numerator, denominator = [], []
    current = numerator
    pending_prefix, pending_power = "", 1
    for t in tokens:
        if t == _SI_PER:
            current = denominator
            continue
        if t in _SI_PREFIX_TOKENS:
            pending_prefix += t
            continue
        if t in _SI_POWER_TOKENS:
            pending_power = _SI_POWER_TOKENS[t]
            continue
        current.append([pending_prefix + t, pending_power])
        pending_prefix, pending_power = "", 1
    return numerator, denominator


def _merge_same_units(unit_list: list) -> list:
    """Additionne les puissances des occurrences répétées d'un même nom."""
    powers, order = {}, []
    for name, power in unit_list:
        if name not in powers:
            order.append(name)
            powers[name] = 0
        powers[name] += power
    return [[name, powers[name]] for name in order if powers[name] != 0]


def _cancel_units(numerator: list, denominator: list) -> tuple[list, list]:
    """Simplifie les unités communes au numérateur et au dénominateur."""
    num, den = dict(numerator), dict(denominator)
    for name in list(num):
        if name in den:
            shared      = min(num[name], den[name])
            num[name]  -= shared
            den[name]  -= shared
    return (
        [[n, p] for n, p in num.items() if p != 0],
        [[n, p] for n, p in den.items() if p != 0],
    )


def _render_unit_list(unit_list: list) -> str:
    parts = []
    for name, power in unit_list:
        if power == 1:
            parts.append(name)
        elif power == 2:
            parts.append(r"\square" + name)
        elif power == 3:
            parts.append(r"\cubic" + name)
        elif power != 0:
            parts.append(name + rf"\tothe{{{power}}}")
    return "".join(parts)


def _divide_units(numerator_unit: str, denominator_unit: str) -> str:
    """
    Construit l'unité siunitx du quotient numerator_unit / denominator_unit
    par une véritable algèbre numérateur/dénominateur, au lieu d'une simple
    concaténation textuelle "y_unit\\per{x_unit}".

    La concaténation textuelle échoue dès que numerator_unit contient déjà
    un \\per : diviser \\meter\\per\\second (une vitesse) par \\second (un
    temps) donnait auparavant \\meter\\per\\second\\per\\second, qui compile
    sans erreur mais s'affiche "m/(s s)" au lieu de "m/s²". Cette fonction
    traite chaque unité comme une paire (numérateur, dénominateur), combine
    les deux quotients, fusionne les puissances d'un même nom et simplifie
    les unités communes, avant de reconstruire la chaîne siunitx finale
    (ex: \\meter\\per\\square\\second).
    """
    num_y, den_y = _parse_unit(numerator_unit)
    num_x, den_x = _parse_unit(denominator_unit)
    combined_num = _merge_same_units(num_y + den_x)
    combined_den = _merge_same_units(den_y + num_x)
    combined_num, combined_den = _cancel_units(combined_num, combined_den)
    num_str = _render_unit_list(combined_num)
    den_str = _render_unit_list(combined_den)
    if den_str:
        return (num_str + _SI_PER + den_str) if num_str else (_SI_PER + den_str)
    return num_str


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
    reg   = _reg_precomputed if _reg_precomputed is not None else linear_regression(x_data, y_data)
    lines = []

    s_unit = slope_unit if slope_unit else _divide_units(y_unit, x_unit)
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
        rf"Le coefficient de corrélation $r^2 = \num{{{reg['r2']:.4f}}}$ "
        rf"confirme la qualité de l'ajustement."
    )
    lines.append(r"\newline")
    lines.append("")
    lines.append(r"Le résultat de la régression s'écrit :")

    intercept_term = (
        rf"\left({_si(reg['theta0'], i_unit, sig=3)} \pm {_si(reg['u_theta0'], i_unit, sig=2)}\right)"
    )
    slope_term = (
        rf"\left({_si(reg['theta1'], s_unit, sig=3)} \pm {_si(reg['u_theta1'], s_unit, sig=2)}\right)"
    )
    box_inline = rf"{y_symbol} = {intercept_term} + {slope_term} \cdot {x_symbol}"

    lines.append(r"\[")
    if len(box_inline) > 70:
        # Fallback multi-ligne : un \boxed{} ne peut contenir qu'une seule
        # ligne de display math nativement ; on imbrique un environnement
        # `array` pour conserver l'encadrement sur un résultat trop long
        # (cas fréquent dès que les unités composées entrent en jeu).
        lines.append(r"    \boxed{")
        lines.append(r"    \begin{array}{c}")
        lines.append(rf"    {y_symbol} = {intercept_term} \\[4pt]")
        lines.append(rf"    {{}} + {slope_term} \cdot {x_symbol}")
        lines.append(r"    \end{array}")
        lines.append(r"    }")
    else:
        lines.append(rf"    \boxed{{{box_inline}}}")
    lines.append(r"\]")

    return "\n".join(lines)


def generate_annexe(bilans: list[str]) -> str:
    """
    Assemble la section LaTeX d'annexe à partir d'une liste de bilans.
    """
    body = "\n\n".join(bilans)
    return r"\section{Annexe : Bilans d'incertitudes}" + "\n\n" + body