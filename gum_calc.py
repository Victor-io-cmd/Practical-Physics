"""
gum_calc.py — Moteur de calcul GUM + Export LaTeX

LIMITATIONS CONNUES
--------------------
- Les grandeurs d'entrée d'une même formule sont supposées indépendantes
  par défaut : `calculate_uncertainty` ne prend en compte une covariance
  entre deux entrées que si elle lui est explicitement fournie via le
  paramètre `covariances` (GUM §5.2, formule 13). Le cas le plus fréquent
  — un mesurande qui combine `theta0` et `theta1` issus de la même
  régression linéaire — est traité automatiquement par
  `full_pipeline_regression_to_measurand`, qui calcule la covariance OLS
  analytique Cov(θ₀, θ₁) = -x̄·s²_res/Sxx et la transmet au pipeline.
  Toute autre source de covariance (deux mesurandes distincts dérivés
  d'une même régression, par exemple) reste à la charge de l'appelant.
- La propagation est strictement linéaire au premier ordre (dérivées
  partielles à la valeur nominale) : pas d'alternative Monte-Carlo
  (GUM Supplément 1) pour les modèles fortement non linéaires.
- `linear_regression` n'implémente pas de régression pondérée : les
  points de mesure sont supposés tous porter la même incertitude sur y.
- L'algèbre d'unités siunitx (`_parse_unit`/`_divide_units`) est purement
  symbolique sur le nom du token : aucune conversion numérique entre
  préfixes SI d'une même unité de base (ex : \\kilo\\gram vs \\gram).
"""

import math
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from types import MappingProxyType
import sympy as sp
from scipy import stats as scipy_stats


# ============================================================
# PARTIE 1 — MOTEUR DE CALCUL GUM
# ============================================================

# Ensemble des tokens siunitx reconnus comme unités de base ou dérivées.
# Utilisé par `_parse_unit` pour détecter les fautes d'orthographe au
# moment de la construction de l'expression d'unité.
_KNOWN_SIUNITX_UNITS = {
    r"\meter", r"\gram", r"\second", r"\ampere", r"\kelvin",
    r"\mole", r"\candela", r"\ohm", r"\volt", r"\watt",
    r"\hertz", r"\newton", r"\pascal", r"\joule", r"\coulomb",
    r"\farad", r"\henry", r"\tesla", r"\becquerel", r"\gray",
    r"\lumen", r"\lux", r"\radian", r"\steradian",
}

# Seuil d'incertitude relative u(xi)/|xi| au-delà duquel la propagation au
# premier ordre (GUM §5.1.2) risque de sous-estimer u_c de façon notable.
_NONLINEARITY_RELATIVE_THRESHOLD = 0.10


@dataclass
class UncertaintyInput:
    """
    Conteneur typé pour une source d'incertitude GUM.

    Remplace l'ancien dict brut retourné par les fonctions
    `uncertainty_type_*`. La validation des champs `u` et `type` est
    effectuée dans `__post_init__`, ce qui rend les erreurs de saisie
    détectables dès la construction plutôt qu'au cœur du calcul.
    """
    u: float
    type: str           # "A" | "B" | "exact"
    nu: float = float("inf")
    distribution: str = ""   # "uniform" | "general" | "constant" | ""
    N: int = 0           # Type A : nombre de mesures
    s: float = 0.0       # Type A : écart-type empirique
    a: float = 0.0       # Type B uniform : demi-largeur
    mean: float = 0.0    # Type A : moyenne empirique (0.0 pour les types B/exact)

    def __post_init__(self):
        if self.type not in ("A", "B", "exact"):
            raise ValueError(f"type invalide : {self.type!r}")
        if self.u < 0:
            raise ValueError(f"u doit être positif ou nul : {self.u}")

    @classmethod
    def from_dict(cls, d: dict) -> "UncertaintyInput":
        """Compatibilité arrière : accepte l'ancien format dict."""
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            # Détection des fautes de casse fréquentes (ex : "U" au lieu de "u")
            warnings.warn(
                f"UncertaintyInput.from_dict : clé(s) inconnue(s) ignorée(s) : "
                f"{sorted(unknown)}. Vérifiez les noms de champs "
                f"(attendus : {sorted(known)}).",
                UserWarning,
                stacklevel=2,
            )
        return cls(**{k: v for k, v in d.items() if k in known})


def uncertainty_type_A(values: list[float]) -> UncertaintyInput:
    """
    Incertitude de type A par répétition sur N mesures indépendantes.

    Paramètres
    ----------
    values : liste des valeurs mesurées (au moins 2).

    Retourne un UncertaintyInput avec u = s/sqrt(N), nu = N-1.
    """
    N = len(values)
    if N < 2:
        raise ValueError("Type A requiert au moins 2 mesures.")
    mean = sum(values) / N
    variance = sum((x - mean) ** 2 for x in values) / (N - 1)
    s = math.sqrt(variance)
    u_A = s / math.sqrt(N)
    return UncertaintyInput(u=u_A, type="A", nu=N - 1, N=N, s=s, mean=mean)


def uncertainty_type_B_uniform(half_width: float) -> UncertaintyInput:
    """
    Incertitude de type B — distribution uniforme de demi-largeur a.

    u_B = a / sqrt(3)

    Paramètres
    ----------
    half_width : demi-largeur a de la distribution (même unité que la grandeur).
    """
    if half_width < 0:
        raise ValueError("half_width doit être positif ou nul.")
    u_B = half_width / math.sqrt(3)
    return UncertaintyInput(u=u_B, type="B", distribution="uniform",
                            nu=float("inf"), a=half_width)


def uncertainty_type_B_from_resolution(resolution: float) -> UncertaintyInput:
    """
    Incertitude de type B à partir de la résolution δ d'un instrument numérique.

    La demi-largeur vaut δ/2, donc u_B = (δ/2) / sqrt(3).

    Paramètres
    ----------
    resolution : résolution δ de l'instrument (graduation ou dernier digit).
    """
    if resolution < 0:
        raise ValueError("resolution doit être positive ou nulle.")
    return uncertainty_type_B_uniform(half_width=resolution / 2)


def uncertainty_type_exact() -> UncertaintyInput:
    """
    Constante exacte — incertitude-type nulle par définition.

    À utiliser pour les constantes physiques fondamentales (c, e, h...),
    les masses étalons certifiées, ou toute grandeur dont l'incertitude
    est négligeable devant les autres sources.
    """
    return UncertaintyInput(u=0.0, type="exact", distribution="constant",
                            nu=float("inf"))


def uncertainty_type_B_relative(u_standard: float,
                                 relative_knowledge: float = None) -> UncertaintyInput:
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
    if relative_knowledge is not None:
        if relative_knowledge == 0:
            warnings.warn(
                "uncertainty_type_B_relative : relative_knowledge=0 est physiquement "
                "absurde (connaissance à 0 % de précision) et sera ignoré — "
                "nu reste float('inf'). Vérifiez la valeur fournie.",
                UserWarning,
                stacklevel=2,
            )
        elif relative_knowledge < 0:
            raise ValueError(
                "uncertainty_type_B_relative : relative_knowledge doit être dans ]0, 1] "
                f"— valeur reçue : {relative_knowledge}. "
                "Une connaissance relative négative est physiquement absurde."
            )
        elif relative_knowledge > 1:
            raise ValueError(
                "uncertainty_type_B_relative : relative_knowledge doit être dans ]0, 1] "
                f"— valeur reçue : {relative_knowledge}. "
                "Une connaissance relative à plus de 100% est physiquement absurde."
            )
        elif relative_knowledge > 0:
            nu = 1.0 / (2.0 * relative_knowledge ** 2)
    return UncertaintyInput(u=u_standard, type="B", distribution="general", nu=nu)


def validate_uncertainty_inputs(
    variable_names: list[str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, UncertaintyInput],
) -> None:
    """
    Vérifie la présence de toutes les clés nécessaires avant tout calcul GUM.

    La validation du contenu de chaque UncertaintyInput (u ≥ 0, type valide)
    est déléguée à UncertaintyInput.__post_init__, qui s'exécute à la
    construction. Cette fonction se limite donc à vérifier la présence
    d'une valeur nominale et d'une entrée d'incertitude pour chaque variable.
    """
    missing_nominal = [n for n in variable_names if n not in nominal_values]
    if missing_nominal:
        raise ValueError(f"Valeur nominale manquante pour : {missing_nominal}.")

    missing_unc = [n for n in variable_names if n not in uncertainty_inputs]
    if missing_unc:
        raise ValueError(f"Incertitude manquante pour : {missing_unc}.")


def calculate_uncertainty(
    formula_str: str,
    variable_names: list[str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, UncertaintyInput],
    covariances: dict[tuple[str, str], float] = None,
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
    uncertainty_inputs: dict {nom: UncertaintyInput}.
    covariances       : dict optionnel {(nom_i, nom_j): u(x_i, x_j)} pour
                        toute paire d'entrées corrélées (GUM §5.2, formule
                        13). Chaque paire ajoute un terme croisé
                        2·c_i·c_j·u(x_i, x_j) à la variance composée. Les
                        entrées non listées ici restent supposées
                        indépendantes. Cas d'usage typique : deux
                        paramètres `theta0`/`theta1` issus de la même
                        régression OLS, dont la covariance analytique est
                        fournie automatiquement par
                        `full_pipeline_regression_to_measurand`.

    Retourne un dict avec :
      result            — valeur nominale du mesurande
      uc                — incertitude-type composée (covariances incluses)
      uc_squared         — variance composée (covariances incluses)
      sensitivities      — dict {nom: c_i} coefficients de sensibilité
      contributions      — dict {nom: c_i² * u_i²} contributions diagonales
      covariance_terms   — dict {(nom_i, nom_j): 2·c_i·c_j·u(x_i,x_j)}
      budget             — dict {nom: % de la variance composée} — calculé
                            sur la variance composée totale (diagonale +
                            croisée), donc ne somme à 100 % que si
                            `covariances` est vide.
      partial_derivs     — dict {nom: expression SymPy de ∂f/∂x_i}
    """
    if not variable_names:
        raise ValueError(
            "variable_names ne peut pas être vide : un mesurande "
            "dépend d'au moins une grandeur."
        )
    validate_uncertainty_inputs(variable_names, nominal_values, uncertainty_inputs)

    symbols = {name: sp.Symbol(name) for name in variable_names}
    formula = sp.sympify(formula_str, locals=symbols)

    # Détection des symboles non déclarés dans formula_str : un identifiant
    # absent de variable_names crée un symbole libre non substitué, ce qui
    # fait lever un TypeError cryptique plusieurs appels plus loin.
    undeclared = formula.free_symbols - set(symbols.values())
    if undeclared:
        names_str = ", ".join(sorted(str(s) for s in undeclared))
        raise ValueError(
            f"formula_str contient des symboles non déclarés dans "
            f"variable_names : {{{names_str}}}. "
            f"Ajoutez-les à variable_names ou corrigez formula_str."
        )

    subs = [(symbols[name], nominal_values[name]) for name in variable_names]

    def _safe_float(expr, context: str) -> float:
        """Convertit une expression SymPy en float avec un message d'erreur explicite."""
        nom_str = ", ".join(f"{n}={nominal_values[n]}" for n in variable_names)

        def _fail(detail: str, exc=None):
            raise ValueError(
                f"Impossible d'évaluer {context} au point nominal "
                f"({{{nom_str}}}). Cause : {detail}."
            ) from exc

        # expr.subs() peut lui-même lever (pôle exact rencontré pendant la
        # substitution symbolique, avant même la conversion en float) : ce
        # cas n'était pas couvert par le try précédent, qui ne portait que
        # sur float(evaluated).
        try:
            evaluated = expr.subs(subs)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            _fail(str(exc), exc)

        try:
            value = float(evaluated)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            if getattr(evaluated, "is_infinite", False):
                detail = "singularité (division par zéro ou pôle)"
            elif getattr(evaluated, "is_nan", False):
                detail = "valeur indéfinie (NaN SymPy)"
            elif evaluated.free_symbols:
                detail = f"symbole(s) résiduel(s) : {evaluated.free_symbols}"
            else:
                detail = str(exc)
            _fail(detail, exc)

        # float() peut convertir sp.oo/zoo sans lever (selon la forme
        # algébrique) en renvoyant silencieusement math.inf, qui se
        # propagerait ensuite en NaN dans le budget d'incertitude.
        if not math.isfinite(value):
            _fail("singularité (résultat non fini au point nominal)")
        return value

    result = _safe_float(formula, "la formule")

    partial_derivs = {}
    sensitivities = {}
    for name in variable_names:
        dp = sp.diff(formula, symbols[name])
        partial_derivs[name] = dp
        sensitivities[name] = _safe_float(dp, f"∂f/∂{name}")

    contributions = {}
    uc_squared = 0.0
    for name in variable_names:
        ci = sensitivities[name]
        ui = uncertainty_inputs[name].u
        xi = nominal_values[name]
        # Garde GUM §5.1.2 : le modèle de propagation est strictement au
        # premier ordre. Une incertitude relative importante sur une
        # entrée (typiquement u(T)/T ~ 20% en rayonnement de
        # Stefan-Boltzmann) peut faire sous-estimer u_c de plusieurs
        # dizaines de % sans qu'aucun signal ne soit donné à l'utilisateur.
        if xi != 0 and ui / abs(xi) > _NONLINEARITY_RELATIVE_THRESHOLD:
            warnings.warn(
                f"calculate_uncertainty : u({name})/|{name}| = "
                f"{ui / abs(xi):.1%} dépasse le seuil de "
                f"{_NONLINEARITY_RELATIVE_THRESHOLD:.0%} — l'approximation "
                "linéaire (GUM §5.1.2) peut sous-estimer significativement "
                "u_c pour cette grandeur. Envisagez une approche Monte-Carlo "
                "(GUM Supplément 1) si la non-linéarité du modèle est forte.",
                RuntimeWarning,
                stacklevel=2,
            )
        contrib = ci ** 2 * ui ** 2
        contributions[name] = contrib
        uc_squared += contrib

    # Termes croisés GUM (§5.2, formule 13) : 2·c_i·c_j·u(x_i, x_j) pour
    # chaque paire d'entrées déclarée corrélée. N'affecte que les paires
    # explicitement fournies — toute paire absente de `covariances` reste
    # traitée comme indépendante (terme nul).
    covariance_terms = {}
    for (name_i, name_j), cov_ij in (covariances or {}).items():
        if name_i not in variable_names or name_j not in variable_names:
            raise ValueError(
                f"calculate_uncertainty : covariance déclarée pour "
                f"({name_i!r}, {name_j!r}) mais l'une de ces variables "
                f"n'est pas dans variable_names={variable_names}."
            )
        term = 2.0 * sensitivities[name_i] * sensitivities[name_j] * cov_ij
        covariance_terms[(name_i, name_j)] = term
        uc_squared += term

    if uc_squared < 0:
        raise ValueError(
            "calculate_uncertainty : variance composée négative après prise "
            "en compte des covariances fournies — au moins une covariance "
            "dépasse la borne de Cauchy-Schwarz u(x_i,x_j) <= u(x_i)*u(x_j) "
            "et n'est donc pas physiquement valide. Vérifiez les valeurs "
            "passées à `covariances`."
        )

    uc = math.sqrt(uc_squared)

    budget = {
        name: 100.0 * contributions[name] / uc_squared if uc_squared > 0 else 0.0
        for name in variable_names
    }

    return {
        "result": result,
        "uc": uc,
        "uc_squared": uc_squared,
        "sensitivities": sensitivities,
        "contributions": contributions,
        "covariance_terms": covariance_terms,
        "budget": budget,
        "partial_derivs": partial_derivs,
    }


def welch_satterthwaite(
    variable_names: list[str],
    sensitivities: dict[str, float],
    uncertainty_inputs: dict[str, UncertaintyInput],
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
        ui   = uncertainty_inputs[name].u
        nu_i = uncertainty_inputs[name].nu
        ui_y = ci * ui
        # math.isfinite() rejette correctement inf ET nan,
        # contrairement à != float("inf") qui laisse passer nan.
        if math.isfinite(nu_i) and nu_i > 0 and abs(ui_y) > 0:
            denominator  += (ui_y ** 4) / nu_i
            has_finite_nu = True

    if not has_finite_nu or denominator == 0.0:
        nu_eff = float("inf")
        k      = 2.0
    else:
        nu_eff = (uc_squared ** 2) / denominator
        k_raw  = scipy_stats.t.ppf(0.975, df=nu_eff)
        if not math.isfinite(k_raw):
            warnings.warn(
                f"welch_satterthwaite : facteur k non fini (nu_eff={nu_eff:.3g}). "
                "Les degrés de liberté effectifs sont anormalement proches de 0 — "
                "on retombe sur k = 2.0 par sécurité. Vérifiez les nu fournis.",
                RuntimeWarning,
                stacklevel=2,
            )
            k = 2.0
        else:
            # _round_half_up (et non round(), qui applique l'arrondi bancaire
            # Python) pour rester cohérent avec la convention utilisée
            # partout ailleurs dans le module.
            k = _round_half_up(k_raw, 0.001)

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
      cov_theta01        — covariance OLS Cov(θ₀, θ₁) = -x̄·s²_res/Sxx
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

    for label, data in (("x_data", x_data), ("y_data", y_data)):
        bad = [v for v in data if not math.isfinite(v)]
        if bad:
            raise ValueError(
                f"{label} contient des valeurs non finies ({bad[:3]}). "
                "Nettoyez les données avant la régression."
            )

    # Centrage préalable : élimine la cancellation catastrophique sur les
    # données à grand offset (différence de grands nombres proches dans les
    # sommes Σx², Σxy). Les estimateurs OLS sont identiques à la formule
    # classique — seule la stabilité numérique change.
    x_mean = sum(x_data) / N
    y_mean = sum(y_data) / N
    xc = [x - x_mean for x in x_data]
    yc = [y - y_mean for y in y_data]

    sxx = sum(xi ** 2 for xi in xc)
    sxy = sum(xi * yi for xi, yi in zip(xc, yc))

    if sxx == 0:
        if len(set(x_data)) == 1:
            raise ValueError("Déterminant nul : toutes les valeurs xi sont identiques.")
        # sxx==0 avec des xi pourtant distincts : ce n'est pas une erreur
        # de saisie mais une cancellation catastrophique IEEE 754 — à grand
        # offset commun (ex. 1e15), des xi distincts deviennent
        # indiscernables en double précision dans la somme Σ(xi-x̄)².
        raise ValueError(
            "Déterminant nul : les valeurs xi sont distinctes mais leur écart "
            "est noyé par un offset commun trop grand pour la précision "
            "flottante (double précision IEEE 754). Soustrayez un offset "
            "commun aux données (ex. x_data - x_data[0]) avant la régression."
        )

    theta1 = sxy / sxx
    theta0 = y_mean - theta1 * x_mean

    y_pred    = [theta0 + theta1 * x for x in x_data]
    residuals = [y - yp for y, yp in zip(y_data, y_pred)]
    s2_res    = sum(r ** 2 for r in residuals) / (N - 2)
    s2_res    = max(0.0, s2_res)   # garde-fou cancellation numérique : s2_res peut être légèrement
                                   # négatif (O(1e-16)) par annulation IEEE 754 ; math.sqrt lèverait
                                   # alors ValueError. Le clamp à 0 est physiquement correct.

    # Le garde-fou compare désormais s2_res à Var(y) centrée, qui est
    # l'échelle pertinente pour les résidus — contrairement à max(y²) qui
    # peut masquer une cancellation si les données ont un grand offset.
    syy = sum(yi ** 2 for yi in yc)
    if syy > 0 and s2_res < 1e-12 * syy / (N - 1):
        warnings.warn(
            "linear_regression : variance résiduelle quasi nulle — "
            "données quasi-colinéaires ou cancellation résiduelle. "
            "Vérifier la signification physique de u(theta0)/u(theta1).",
            RuntimeWarning,
        )

    # Var(θ₁) = s²/Sxx, Var(θ₀) = s²·(1/N + x̄²/Sxx) — formes centrées,
    # numériquement équivalentes aux formes classiques mais plus stables.
    u_theta1 = math.sqrt(s2_res / sxx)
    u_theta0 = math.sqrt(s2_res * (1.0 / N + x_mean ** 2 / sxx))
    if not (math.isfinite(u_theta0) and math.isfinite(u_theta1)):
        raise ValueError(
            "linear_regression : incertitude non finie sur theta0/theta1 — "
            "cancellation numérique sévère sur Sxx (offset des données trop "
            "grand devant leur dispersion). Centrez ou réduisez l'offset des "
            "données avant la régression."
        )

    ss_res = sum(r ** 2 for r in residuals)
    if syy > 0:
        r2 = 1.0 - ss_res / syy
    else:
        # Toutes les yi identiques : r2=1.0 afficherait un ajustement
        # "parfait" trompeur, alors que la régression est triviale et sans
        # pouvoir prédictif. NaN signale explicitement l'indétermination.
        warnings.warn(
            "linear_regression : toutes les valeurs yi sont identiques "
            "(syy=0) — r² n'a pas de sens prédictif, fixé à NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        r2 = float("nan")

    # Cov(θ₀, θ₁) = -x̄ · s²_res / Sxx — covariance OLS analytique standard
    # pour la régression linéaire simple (θ₀ = ȳ - θ₁x̄ et Var(θ₁) = s²/Sxx
    # donnent directement ce résultat). Exposée pour permettre à l'appelant
    # de propager correctement un mesurande combinant θ₀ et θ₁ (voir
    # `full_pipeline_regression_to_measurand`) plutôt que de les traiter à
    # tort comme indépendants.
    cov_theta01 = -x_mean * s2_res / sxx

    return {
        "theta0": theta0,
        "theta1": theta1,
        "u_theta0": u_theta0,
        "u_theta1": u_theta1,
        "cov_theta01": cov_theta01,
        "s_res": math.sqrt(s2_res),
        "r2": r2,
        "N": N,
        "y_pred": y_pred,
        "residuals": residuals,
        "nu": N - 2,
    }


def _round_half_up(value: float, step: float) -> float:
    """
    Arrondit value au multiple de step le plus proche, convention
    "moitié vers le haut" symétrique en signe.

    Implémentation via decimal.Decimal (str → Decimal) pour éviter
    l'instabilité IEEE 754 : `0.15 / 0.1 = 1.4999999999999998` en float,
    ce qui fait basculer math.floor du mauvais côté pour les valeurs
    décimales exactement à mi-chemin (0.15, 0.35, 0.45...). La conversion
    str → Decimal reproduit fidèlement la valeur décimale attendue.

    Implémentation unique partagée par `round_to_sig_figs` et
    `format_result` — centraliser l'arrondi évite toute divergence future
    entre les deux fonctions.
    """
    if value == 0:
        return 0.0
    sign = 1 if value >= 0 else -1
    d_abs  = Decimal(str(abs(value)))
    d_step = Decimal(str(step))
    rounded = (d_abs / d_step).to_integral_value(rounding=ROUND_HALF_UP)
    return sign * float(rounded * d_step)


def round_to_sig_figs(value: float, sig_figs: int) -> float:
    """
    Arrondit value à sig_figs chiffres significatifs (voir _round_half_up
    pour la convention d'arrondi retenue).
    """
    if not math.isfinite(value):
        raise ValueError(
            f"round_to_sig_figs : value non finie ({value}) — impossible "
            "de calculer une magnitude décimale (math.log10 diverge)."
        )
    if value == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    step      = 10 ** (magnitude - sig_figs + 1)
    return _round_half_up(value, step)


def format_result(result: float, U: float, sig_figs_exact: int = 4) -> dict:
    """
    Arrondit le résultat et l'incertitude élargie de façon cohérente.

    Si U_rounded == 0 (toutes les grandeurs d'entrée sont exactes), le
    résultat est tout de même arrondi à `sig_figs_exact` chiffres
    significatifs : sans cela, tout code consultant directement
    `result_rounded` (notebook, print de débogage...) sans repasser par
    `_format_result_uncertainty` recevrait la précision flottante brute.
    """
    if not math.isfinite(result):
        raise ValueError(
            f"format_result : result non fini ({result}). Une formule "
            "évaluée à une singularité (pôle, division par zéro) ne peut "
            "pas être arrondie ni affichée."
        )
    if not math.isfinite(U):
        raise ValueError(
            f"format_result : U non finie ({U}). Cela indique typiquement "
            "des degrés de liberté effectifs anormalement proches de 0 "
            "(relative_knowledge mal renseigné, cf. uncertainty_type_B_relative)."
        )
    U_rounded = round_to_sig_figs(U, 2)
    if U_rounded == 0:
        result_rounded = round_to_sig_figs(result, sig_figs_exact) if result != 0 else 0.0
        return {"result": result_rounded, "U": 0.0, "decimals": 0}

    mag_U    = math.floor(math.log10(abs(U_rounded)))
    decimals = -mag_U + 1
    step     = 10 ** (mag_U - 1)
    result_rounded = _round_half_up(result, step)

    return {
        "result": result_rounded,
        "U": U_rounded,
        "decimals": decimals,
    }


def full_gum_analysis(
    formula_str: str,
    variable_names: list[str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, UncertaintyInput],
    k_override: float = None,
    global_sig_figs: int = 4,
    covariances: dict[tuple[str, str], float] = None,
) -> MappingProxyType:
    """
    Pipeline GUM complet : propagation + Welch-Satterthwaite + formatage.

    Retourne un MappingProxyType (lecture seule) pour prévenir toute
    mutation accidentelle du résultat après coup.

    `global_sig_figs` ne pilote que le cas particulier où toutes les
    grandeurs d'entrée sont exactes (U = 0, voir `format_result`) : c'est
    le seul cas où le nombre de chiffres significatifs du résultat n'est
    pas dicté par l'incertitude elle-même. Le défaut (4) préserve le
    comportement historique pour tout appel direct de cette fonction hors
    de `generate_bilan` ; ce dernier transmet son propre `global_sig_figs`
    pour rester cohérent avec le reste du bilan affiché.

    `covariances` est transmis tel quel à `calculate_uncertainty` (voir sa
    docstring) : Welch-Satterthwaite reste calculé à partir des
    contributions diagonales uniquement (la formule classique ne se
    généralise pas simplement aux covariances), ce qui reste l'approche
    standard en présence d'un nombre réduit de paires corrélées.
    """
    calc = calculate_uncertainty(
        formula_str, variable_names, nominal_values, uncertainty_inputs,
        covariances=covariances,
    )
    ws = welch_satterthwaite(
        variable_names,
        calc["sensitivities"],
        uncertainty_inputs,
        calc["uc_squared"],
    )
    k         = k_override if k_override is not None else ws["k"]
    U         = expanded_uncertainty(calc["uc"], k)
    formatted = format_result(calc["result"], U, sig_figs_exact=global_sig_figs)

    return MappingProxyType({
        "result":          calc["result"],
        "uc":              calc["uc"],
        "uc_squared":      calc["uc_squared"],
        "sensitivities":   MappingProxyType(calc["sensitivities"]),
        "contributions":   MappingProxyType(calc["contributions"]),
        "covariance_terms": MappingProxyType(calc["covariance_terms"]),
        "budget":          MappingProxyType(calc["budget"]),
        "partial_derivs":  MappingProxyType(calc["partial_derivs"]),
        "nu_eff":          ws["nu_eff"],
        "k":               k,
        "U":               U,
        "result_rounded":  formatted["result"],
        "U_rounded":       formatted["U"],
        "decimals":        formatted["decimals"],
    })


# ============================================================
# PARTIE 2 — EXPORT LATEX
# ============================================================

# --- Seuils de mise en forme centralisés : toute évolution future de
# l'un de ces seuils ne doit être faite qu'ici, pour ne pas les faire
# diverger silencieusement entre les différentes fonctions qui les
# utilisent. ---
_SCI_NOTATION_MAG_MIN = -2   # en dessous : notation scientifique
_SCI_NOTATION_MAG_MAX = 3    # au-dessus  : notation scientifique
_INLINE_MAX_CHARS     = 70   # au-delà : \boxed{...} bascule en \begin{array}
_ALIGN_MAX_CHARS      = 90   # au-delà : ligne unique \[ \] bascule en align*
_ALIGN_MAX_TERMS      = 3    # à partir de : bascule en align* (même sans dépasser la longueur)

_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
}


def _escape_latex(s: str) -> str:
    """
    Échappe les caractères spéciaux LaTeX d'une chaîne de prose libre.

    Réservé aux champs insérés tels quels en dehors de tout mode
    mathématique (measurand_name, titres de sous-section) : measurand_symbol
    et variable_symbols, eux, sont du LaTeX volontairement saisi par
    l'utilisateur ($\\theta_1$, \\alpha...) et ne doivent surtout pas être
    échappés, sous peine de casser l'affichage mathématique recherché.

    La substitution caractère par caractère (et non une succession de
    `str.replace`) évite le double échappement d'un caractère déjà
    introduit par le remplacement d'un caractère précédent.
    """
    if not s:
        return s
    return "".join(_LATEX_SPECIAL_CHARS.get(c, c) for c in s)


def _magnitude_of(value: float) -> int:
    """
    Magnitude décimale floor(log10(|value|)) robuste à l'instabilité de
    représentation flottante près d'une puissance de 10 exacte
    (ex: log10(10.0) peut valoir 0.999999999... en double précision).

    Source unique partagée par `_mantissa_exp` et `_format_magnitude` :
    centraliser ce calcul évite toute divergence entre les deux branches
    de rendu (décimale vs scientifique) pour une même valeur arrondie.
    """
    mag = math.floor(math.log10(abs(value)))
    if abs(value) / (10 ** mag) >= 10:
        mag += 1
    return mag


def _mantissa_exp(value: float, sig: int):
    """
    Décompose value en (signe, mantisse, exposant) pour la notation
    scientifique mantisse * 10^exposant, avec mantisse dans [1, 10).

    Une seule décision d'arrondi "physique" est appliquée
    (round_to_sig_figs) ; la division par 10**mag qui suit n'est qu'une
    simple lecture de cette valeur déjà arrondie, pour éviter tout double
    arrondi en cascade avec deux conventions différentes.
    """
    if value == 0:
        return "", 0.0, 0
    rounded  = round_to_sig_figs(value, sig)
    sign     = "-" if rounded < 0 else ""
    mag      = _magnitude_of(rounded)
    mantissa = abs(rounded) / (10 ** mag)
    # Garde-fou « retenue » : un arrondi peut faire passer la mantisse à
    # 10.000 pile (ex: 9.996 arrondi à 3 c.s. -> 10.0), ce qui violerait
    # la convention [1, 10) de la notation scientifique. On vérifie sur
    # la mantisse déjà formatée à sig-1 décimales (et non sur la valeur
    # flottante brute, sujette au bruit de représentation binaire) avant
    # de décider de la retenue.
    if round(mantissa, sig - 1) >= 10:
        mag      += 1
        mantissa  = abs(rounded) / (10 ** mag)
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
    défaut est `\\log` (sans base explicite), ce qui peut induire en
    erreur un lecteur habitué à la convention « log = base 10, ln = base
    e ». Même `sympy.log(x, 10)` (base explicite) est décomposé en
    interne par SymPy en `log(x)/log(10)` — il n'existe pas de forme
    `\\log_{10}{...}` produite par le printer LaTeX de SymPy. La
    substitution globale `\\log{` -> `\\ln{` est donc sans risque de faux
    positif sur une base explicite.
    """
    return sp.latex(expr).replace(r"\log{", r"\ln{")


def _sci(value: float, sig: int = 2) -> str:
    """
    Alias historique conservé pour compatibilité d'appel : délègue
    entièrement à `_num`.
    """
    return _num(value, sig=sig)


def _format_magnitude(value: float, sig: int) -> str:
    """
    Formate la magnitude numérique d'une valeur (sans \\num{} ni \\qty{}
    autour) : notation décimale pour -2 <= magnitude <= 3, notation
    scientifique mantisse*10^exposant sinon.

    Helper commun à `_num` et `_si`, qui ne diffèrent que par l'habillage
    final (\\num{} ou \\qty{}{unité}).
    """
    value = round_to_sig_figs(value, sig)
    if value == 0:
        return "0"
    mag  = _magnitude_of(value)
    if _SCI_NOTATION_MAG_MIN <= mag <= _SCI_NOTATION_MAG_MAX:
        decimals = max(0, sig - 1 - mag)
        return f"{value:.{decimals}f}"
    # Branche notation scientifique : sign est utile ici (mantissa est abs())
    sign = "-" if value < 0 else ""
    _, mantissa, mag = _mantissa_exp(value, sig)
    return f"{sign}{mantissa:.{sig-1}f}e{mag}"


def _num(value: float, sig: int = 3) -> str:
    if value == 0:
        return r"\num{0}"
    return rf"\num{{{_format_magnitude(value, sig)}}}"


def _si(value: float, unit: str, sig: int = 3) -> str:
    if value == 0:
        return rf"\qty{{0}}{{{unit}}}" if unit else r"\num{0}"
    if not unit:
        return _num(value, sig=sig)
    return rf"\qty{{{_format_magnitude(value, sig)}}}{{{unit}}}"


def _format_result_uncertainty(
    result_rounded: float,
    U_rounded: float,
    unit: str,
    sig_figs_exact: int = 4,
) -> str:
    """
    Construit la chaîne LaTeX siunitx "valeur +- incertitude" finale pour
    un résultat encadré.

    Le nombre de décimales nécessaire est intégralement recalculé ici, à
    partir de U_rounded seul, qui est la seule source de vérité réellement
    utile (U_rounded porte par construction exactement 2 chiffres
    significatifs, cf. round_to_sig_figs(U, 2) dans format_result).
    """
    if U_rounded == 0:
        # Cas "incertitude pure" : toutes les grandeurs d'entrée sont
        # exactes. On arrondit alors le résultat seul à sig_figs_exact
        # chiffres significatifs, sans tentative de log10(0).
        val = round_to_sig_figs(result_rounded, sig_figs_exact) if result_rounded != 0 else 0.0
        return _si(val, unit, sig=sig_figs_exact) if unit else _num(val, sig=sig_figs_exact)

    # --- Magnitude de l'incertitude, seule grandeur qui pilote le nombre
    # de décimales (directive métrologique : c'est U, jamais le résultat,
    # qui impose le nombre de décimales affichées). ---
    mag_U = math.floor(math.log10(abs(U_rounded)))

    # Le résultat peut s'arrondir exactement à 0 (mesure compatible avec
    # zéro, situation expérimentale parfaitement normale) sans que U le
    # soit : log10(0) n'existe pas. On bascule alors la magnitude de
    # référence du résultat sur celle de U_rounded.
    mag_result = math.floor(math.log10(abs(result_rounded))) if result_rounded != 0 else mag_U

    # decimals_raw : nombre de décimales nécessaires pour afficher les 2
    # chiffres significatifs de U_rounded à l'échelle 1 (non mise à
    # l'échelle). Peut être négatif si mag_U est grand (U >= 10 en valeur
    # absolue) : c'est géré explicitement ici, jamais transmis tel quel à
    # un format f"{x:.{d}f}" qui lèverait ValueError.
    decimals_raw = 1 - mag_U

    # --- Choix de l'échelle d'affichage commune ---
    # Par défaut, on cale l'affichage sur la magnitude du résultat (le
    # plus lisible : c'est la grandeur d'intérêt). Mais si l'incertitude
    # dépasse le résultat de plus d'une décade (mesure très peu précise,
    # résultat "noyé" dans son incertitude), caler l'échelle sur le
    # résultat forcerait à tronquer les chiffres significatifs de U via
    # le clamp max(0, ...) — silencieusement, sans avertissement. On
    # bascule alors la référence sur mag_U : même logique physique que le
    # cas result_rounded == 0 ci-dessus, généralisée à un seuil d'une
    # décade d'écart.
    anchor_on_result = (mag_result - mag_U) >= -1
    ref_mag = mag_result if anchor_on_result else mag_U

    if _SCI_NOTATION_MAG_MIN <= ref_mag <= _SCI_NOTATION_MAG_MAX:
        # --- Notation décimale simple (pas de mise à l'échelle) ---
        # decimals_raw seul pilote d : aucune perte possible des 2 c.s.
        # de U ici, le clamp à 0 ne fait que renoncer à des décimales
        # superflues pour le résultat si celui-ci est, à raison, noyé
        # dans son incertitude (cf. exemple "0.00 +- 0.12").
        d       = max(0, decimals_raw)
        val_str = f"{result_rounded:.{d}f}"
        u_str   = f"{U_rounded:.{d}f}"
        body    = rf"{val_str} +- {u_str}"
    else:
        # --- Notation scientifique, échelle commune 10^ref_mag ---
        d     = max(0, decimals_raw + ref_mag)
        scale = 10 ** ref_mag
        val_scaled = result_rounded / scale
        u_scaled   = U_rounded / scale

        # Garde-fou « retenue », symétrique de celui de _mantissa_exp : un
        # arrondi à d décimales peut faire passer la mantisse de l'ancre
        # (celle qui a défini ref_mag, donc censée rester dans [1, 10)) à
        # 10.000 pile. Seule l'ancre est vérifiée : l'autre grandeur n'a,
        # elle, aucune raison de rester dans [1,10).
        anchor_scaled = val_scaled if anchor_on_result else u_scaled
        if round(anchor_scaled, d) >= 10:
            ref_mag   += 1
            scale      = 10 ** ref_mag
            d          = max(0, decimals_raw + ref_mag)
            val_scaled = result_rounded / scale
            u_scaled   = U_rounded / scale

        val_str = f"{val_scaled:.{d}f}"
        u_str   = f"{u_scaled:.{d}f}"
        body    = rf"{val_str} +- {u_str} e{ref_mag}"

    # separate-uncertainty=true est ajouté en option LOCALE de la macro,
    # plutôt que de dépendre d'un \sisetup global supposé présent dans le
    # document hôte. Sans cette option (locale ou globale), siunitx
    # affiche par défaut l'incertitude au format compact
    # valeur(incertitude) — ex. "1,382(70)e-29" — qui est un comportement
    # documenté de siunitx, mais qui n'est pas le rendu attendu ici.
    # L'option locale rend ce \qty{}/\num{} autosuffisant : il reste
    # correct même collé isolément hors du gabarit TP.
    if unit:
        return rf"\qty[separate-uncertainty=true]{{{body}}}{{{unit}}}"
    return rf"\num[separate-uncertainty=true]{{{body}}}"


# ============================================================
# PARTIE 2b — BLOCS LATEX INTERNES (generate_bilan décomposé)
# ============================================================
#
# generate_bilan orchestre 6 blocs indépendants. Chaque bloc est une
# fonction pure (entrées explicites → liste de lignes LaTeX) : testable
# unitairement, auditable branche par branche, extensible sans toucher
# les autres blocs.
#
# Convention : chaque fonction retourne list[str] sans join final.
# generate_bilan les concatène et retourne "\n".join(lines).
# ============================================================


def _bilan_model_block(
    measurand_symbol: str,
    name_clause: str,
    formula_str: str,
    variable_names: list,
    sym_map: dict,
    unit_map: dict,
    nominal_values: dict,
    measurand_unit: str,
    res,
    global_sig_figs: int,
) -> list:
    """
    Bloc 1 — Modèle de mesure et valeurs nominales.

    Génère : phrase d'introduction, formule encadrée en display math,
    liste des valeurs nominales en prose et valeur nominale du mesurande.
    """
    lines = []

    formula_sympy = sp.sympify(
        formula_str, locals={n: sp.Symbol(n) for n in variable_names}
    )
    formula_sympy = _rename_symbols(formula_sympy, variable_names, sym_map)
    formula_latex = _latex_ln(formula_sympy)

    is_identity = (formula_str.strip() == variable_names[0] and len(variable_names) == 1)
    if is_identity:
        lines.append(
            rf"\noindent ${measurand_symbol}${name_clause} "
            r"est une grandeur mesurée directement."
        )
    else:
        lines.append(
            rf"\noindent Le mesurande ${measurand_symbol}${name_clause} "
            r"est lié aux grandeurs d'entrée par le modèle :"
        )
        lines.append(r"\[")
        lines.append(rf"    {measurand_symbol} = {formula_latex}")
        lines.append(r"\]")

    defs = []
    for n in variable_names:
        defs.append(rf"${sym_map[n]} = {_si(nominal_values[n], unit_map[n], sig=global_sig_figs)}$")
    defs_str = (", ".join(defs[:-1]) + " et " + defs[-1]) if len(defs) > 1 else defs[0]

    nom_val = res["result"]
    lines.append(
        r"\noindent avec " + defs_str
        + rf", ce qui donne la valeur nominale "
          rf"$\left.{measurand_symbol}\right|_{{\mathrm{{nom}}}} = "
          rf"{_si(nom_val, measurand_unit, sig=global_sig_figs)}$."
    )
    lines.append(r"\newline")
    lines.append("")
    return lines


def _bilan_source_blocks(
    variable_names: list,
    sym_map: dict,
    unit_map: dict,
    uncertainty_inputs: dict,
    global_sig_figs: int,
) -> list:
    """
    Bloc 2 — Une section par source d'incertitude.

    Génère : une entrée par variable (exact / type A / type B uniforme /
    type B général), dans l'ordre de variable_names.
    """
    lines = []
    for n in variable_names:
        inp  = uncertainty_inputs[n]
        sym  = sym_map[n]
        unit = unit_map[n]
        u_val = inp.u

        if inp.type == "exact":
            lines.append(
                rf"\noindent La grandeur ${sym}$ est une constante exacte déterminée par définition. "
                rf"Son incertitude-type est nulle : $u({sym}) = 0$."
            )
        elif inp.type == "A":
            N_mes    = inp.N
            s        = inp.s
            sym_mean = sym if sym.strip().startswith(r"\bar{") else rf"\bar{{{sym}}}"
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
        elif inp.type == "B":
            if inp.distribution == "uniform":
                a     = inp.a
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
        lines.append("")
    return lines


def _bilan_sensitivity_block(
    variable_names: list,
    sym_map: dict,
    measurand_symbol: str,
    res,
    global_sig_figs: int,
) -> list:
    """
    Bloc 3 — Coefficients de sensibilité.

    Génère : les c_i = ∂f/∂x_i évalués aux valeurs nominales, en ligne
    unique ou en align* selon le nombre de termes et la longueur.
    """
    lines = []
    lines.append(r"\noindent Les coefficients de sensibilité, évalués aux valeurs nominales, sont :")
    ci_terms = []
    for n in variable_names:
        ci      = res["sensitivities"][n]
        dp_sym  = res["partial_derivs"][n]
        dp_sym  = _rename_symbols(dp_sym, variable_names, sym_map)
        dp_latex = _latex_ln(dp_sym)
        ci_terms.append(
            rf"c_{{{sym_map[n]}}} = \left.\frac{{\partial {measurand_symbol}}}"
            rf"{{\partial {sym_map[n]}}}\right|_{{\mathrm{{nom}}}}"
            rf" = {dp_latex} = {_sci(ci, sig=global_sig_figs)}"
        )

    ci_line = r"\qquad ".join(ci_terms)
    if len(ci_terms) > _ALIGN_MAX_TERMS or len(ci_line) > _ALIGN_MAX_CHARS:
        lines.append(r"\begin{align*}")
        for i, term in enumerate(ci_terms):
            suffix = r" \\" if i < len(ci_terms) - 1 else ""
            lines.append(rf"    {term.replace(' = ', ' &= ', 1)}{suffix}")
        lines.append(r"\end{align*}")
    else:
        lines.append(r"\[")
        lines.append(r"    " + ci_line)
        lines.append(r"\]")
    lines.append("")
    return lines


def _bilan_propagation_block(
    variable_names: list,
    sym_map: dict,
    measurand_symbol: str,
    uncertainty_inputs: dict,
    res,
    measurand_unit: str,
    global_sig_figs: int,
) -> list:
    """
    Bloc 4 — Propagation des incertitudes → u_c.

    Génère : formule de composition quadratique, avec bascule align* si la
    radicande est trop longue ou compte trop de termes.
    """
    lines = []
    uc = res["uc"]
    inner_terms = []
    for n in variable_names:
        if uncertainty_inputs[n].type == "exact":
            continue
        ci = res["sensitivities"][n]
        ui = uncertainty_inputs[n].u
        if ci == 0:
            # Sensibilité nulle au point nominal : contribution nulle à la
            # variance, terme purement cosmétique à exclure (même
            # traitement que les grandeurs "exact").
            continue
        inner_terms.append(
            rf"({_sci(ci, sig=global_sig_figs)})^2 \, ({_num(ui, sig=global_sig_figs)})^2"
        )

    if not inner_terms:
        lines.append(
            rf"\noindent Toutes les grandeurs d'entrée sont soit des constantes "
            rf"exactes, soit de sensibilité nulle au point nominal, "
            rf"l'incertitude-type composée est nulle : $u_c({measurand_symbol}) = 0$."
        )
        lines.append(r"\newline")
        return lines

    lines.append(r"\noindent La propagation des incertitudes pour des grandeurs indépendantes donne :")
    radicand   = " + ".join(inner_terms)
    result_str = _si(uc, measurand_unit, sig=global_sig_figs)
    n_terms    = len(inner_terms)

    if n_terms > 1 and (len(radicand) > _ALIGN_MAX_CHARS or n_terms >= _ALIGN_MAX_TERMS):
        S_val          = uc ** 2
        n_lines        = math.ceil(n_terms / 3)
        terms_per_line = math.ceil(n_terms / n_lines)
        lines.append(r"\begin{align*}")
        for i in range(0, n_terms, terms_per_line):
            chunk  = inner_terms[i:i + terms_per_line]
            joined = " + ".join(chunk)
            prefix = r"    S &= " if i == 0 else r"    &\quad + "
            lines.append(rf"{prefix}{joined} \\")
        lines.append(rf"    &= {_num(S_val, sig=global_sig_figs)}")
        lines.append(r"\end{align*}")
        lines.append(r"\[")
        lines.append(rf"    u_c({measurand_symbol}) = \sqrt{{S}} = {result_str}")
        lines.append(r"\]")
    else:
        lines.append(r"\[")
        lines.append(
            rf"    u_c({measurand_symbol}) = \sqrt{{{radicand}}}"
            rf" = {result_str}"
        )
        lines.append(r"\]")
    lines.append(r"\newline")
    return lines


def _bilan_ws_block(
    variable_names: list,
    sym_map: dict,
    uncertainty_inputs: dict,
    measurand_symbol: str,
    res,
    global_sig_figs: int,
) -> list:
    """
    Bloc 5 — Welch-Satterthwaite et facteur k.

    Génère : le bloc WS (formule + table de Student) si nu_eff est fini,
    sinon la phrase pour k = 2.00 classique.

    Garde défensive sur `descriptions` avant join : ne peut pas être vide
    quand nu_eff est fini (même critère que welch_satterthwaite), mais
    protège contre toute divergence future entre les deux fonctions.
    """
    lines  = []
    nu_eff = res.get("nu_eff", float("inf"))
    k      = res["k"]

    if nu_eff != float("inf"):
        nu_lines     = []
        descriptions = []
        for n in variable_names:
            if uncertainty_inputs[n].type == "exact":
                continue
            ci   = res["sensitivities"][n]
            ui   = uncertainty_inputs[n].u
            nu_i = uncertainty_inputs[n].nu
            # Critère identique à welch_satterthwaite : seules les composantes
            # à nu fini ET contribution non nulle entrent dans la formule.
            if not math.isfinite(nu_i) or abs(ci * ui) == 0:
                continue
            nu_lines.append(
                rf"\dfrac{{({_sci(ci, sig=global_sig_figs)} \cdot {_num(ui, sig=global_sig_figs)})^4}}{{{round(nu_i)}}}"
            )
            type_str = "type~A" if uncertainty_inputs[n].type == "A" else "type~B"
            descriptions.append(rf"${sym_map[n]}$ ({type_str}, $\nu = {round(nu_i)}$)")

        # Garde défensive : ne devrait pas être vide si nu_eff est fini,
        # mais on évite IndexError en cas de divergence future entre les conditions.
        if descriptions:
            desc_str = (
                ", ".join(descriptions[:-1]) + " et " + descriptions[-1]
                if len(descriptions) > 1 else descriptions[0]
            )
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
            # Cas dégénéré (nu_eff fini mais aucune composante ne satisfait
            # les critères d'affichage) : on retombe sur k standard.
            lines.append(
                rf"\noindent Toutes les composantes configurées possèdent des degrés de liberté infinis, "
                rf"on retient la standardisation classique $k = \num{{{k:.2f}}}$ (95\,\%)."
            )
    else:
        lines.append(
            rf"\noindent Toutes les composantes configurées possèdent des degrés de liberté infinis, "
            rf"on retient la standardisation classique $k = \num{{{k:.2f}}}$ (95\,\%)."
        )
    lines.append("")
    return lines


def _bilan_budget_result_block(
    variable_names: list,
    sym_map: dict,
    measurand_symbol: str,
    measurand_unit: str,
    uncertainty_inputs: dict,
    res,
    global_sig_figs: int,
) -> list:
    """
    Bloc 6 — Incertitude élargie U, budget d'incertitudes, résultat encadré.

    Génère : formule U = k·uc, tableau de budget en prose (inline ou
    align*), phrase sur la source dominante, résultat final dans \\boxed{}.
    """
    lines = []
    uc = res["uc"]
    k  = res["k"]
    U  = res["U"]

    lines.append(r"\[")
    lines.append(
        rf"    U({measurand_symbol}) = k\,u_c({measurand_symbol}) = "
        rf"\num{{{k:.2f}}} \times {_num(uc, sig=global_sig_figs)} = "
        rf"{_si(U, measurand_unit, sig=2)}"
    )
    lines.append(r"\]")
    lines.append(r"\newline")

    non_exact_budget = {
        n: v for n, v in res["budget"].items()
        if uncertainty_inputs[n].type != "exact"
    }
    if non_exact_budget:
        lines.append(r"\noindent Le budget d'incertitudes :")
        budget_terms = []
        for n in variable_names:
            if uncertainty_inputs[n].type == "exact":
                continue
            pct = res["budget"][n]
            budget_terms.append(
                rf"\frac{{c_{{{sym_map[n]}}}^2\,u^2({sym_map[n]})}}{{u_c^2({measurand_symbol})}} = \num{{{pct:.1f}}}\,\%"
            )
        budget_line = r"\qquad ".join(budget_terms)
        if len(budget_terms) > _ALIGN_MAX_TERMS or len(budget_line) > _ALIGN_MAX_CHARS:
            lines.append(r"\begin{align*}")
            for i, term in enumerate(budget_terms):
                suffix = r" \\" if i < len(budget_terms) - 1 else ""
                lines.append(rf"    {term.replace(' = ', ' &= ', 1)}{suffix}")
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
    final_expr = _format_result_uncertainty(
        result_rounded=res["result_rounded"],
        U_rounded=res["U_rounded"],
        unit=measurand_unit,
        sig_figs_exact=global_sig_figs,
    )
    box_inline = rf"{measurand_symbol} = {final_expr}"

    lines.append(r"\[")
    if len(box_inline) > _INLINE_MAX_CHARS:
        lines.append(r"    \boxed{")
        lines.append(r"    \begin{array}{c}")
        lines.append(rf"    {measurand_symbol} \\[4pt]")
        lines.append(rf"    {{}} = {final_expr}")
        lines.append(r"    \end{array}")
        lines.append(r"    }")
    else:
        lines.append(rf"    \boxed{{{box_inline}}}")
    lines.append(r"\]")
    return lines


def generate_bilan(
    measurand_name: str,
    measurand_symbol: str,
    formula_str: str,
    variable_names: list[str],
    variable_symbols: dict[str, str],
    variable_units: dict[str, str],
    nominal_values: dict[str, float],
    uncertainty_inputs: dict[str, UncertaintyInput],
    measurand_unit: str = "",
    k_override: float = None,
    subsection: bool = True,
    global_sig_figs: int = 3,
    covariances: dict[tuple[str, str], float] = None,
    _res_precomputed=None,
) -> str:
    """
    Génère le bloc LaTeX complet du bilan d'incertitudes pour un mesurande.

    `_res_precomputed` accepte le MappingProxyType déjà renvoyé par un
    appel antérieur à `full_gum_analysis` sur les mêmes arguments : évite
    de relancer tout le calcul symbolique quand le notebook appelant a déjà
    besoin du résultat numérique en amont (affichage console, chaînage vers
    un autre mesurande). La validation des entrées a alors déjà eu lieu lors
    de ce premier appel ; elle n'est pas refaite ici. Dans ce cas,
    `covariances` est ignoré (déjà pris en compte dans `_res_precomputed`).

    Le corps délègue entièrement à 6 sous-fonctions privées (_bilan_*_block),
    chacune responsable d'un bloc sémantique indépendant. generate_bilan
    n'orchestre que l'en-tête de sous-section et le join final.
    """
    res     = _res_precomputed if _res_precomputed is not None else full_gum_analysis(
        formula_str, variable_names, nominal_values, uncertainty_inputs, k_override,
        global_sig_figs=global_sig_figs, covariances=covariances,
    )
    sym_map  = {n: variable_symbols.get(n, n) for n in variable_names}
    unit_map = {n: variable_units.get(n, "")  for n in variable_names}
    lines    = []

    if subsection:
        lines.append(rf"\subsection{{Bilan --- Grandeur ${measurand_symbol}$}}")
    lines.append("")

    # Clause de nom entre parenthèses (évite de déduire le genre grammatical
    # français) : omise si measurand_name est identique au symbole.
    name_clause = ""
    if measurand_name and measurand_name.strip().lower() != measurand_symbol.strip().lower():
        name_clause = f" ({_escape_latex(measurand_name)})"

    lines += _bilan_model_block(
        measurand_symbol, name_clause, formula_str,
        variable_names, sym_map, unit_map,
        nominal_values, measurand_unit, res, global_sig_figs,
    )
    lines += _bilan_source_blocks(
        variable_names, sym_map, unit_map, uncertainty_inputs, global_sig_figs,
    )
    lines += _bilan_sensitivity_block(
        variable_names, sym_map, measurand_symbol, res, global_sig_figs,
    )
    lines += _bilan_propagation_block(
        variable_names, sym_map, measurand_symbol,
        uncertainty_inputs, res, measurand_unit, global_sig_figs,
    )
    lines += _bilan_ws_block(
        variable_names, sym_map, uncertainty_inputs,
        measurand_symbol, res, global_sig_figs,
    )
    lines += _bilan_budget_result_block(
        variable_names, sym_map, measurand_symbol, measurand_unit,
        uncertainty_inputs, res, global_sig_figs,
    )

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
    uncertainty_inputs_helpers: dict[str, UncertaintyInput],
    measurand_symbol: str,
    measurand_name: str = "",
    measurand_unit: str = "",
    slope_unit: str = "",
    intercept_unit: str = "",
    slope_symbol: str = r"\theta_1",
    intercept_symbol: str = r"\theta_0",
    global_sig_figs: int = 3,
    couple_theta0: bool = False,
    couple_theta1: bool = False,
) -> str:
    """
    Génère d'un seul appel le LaTeX complet : régression + propagation GUM.

    `couple_theta0` / `couple_theta1` déclarent explicitement que le
    mesurande dépend de `theta0` / `theta1` issus de la régression : ceci
    remplace l'ancienne détection implicite par `"theta0" in
    variable_names`, qui débranchait silencieusement la covariance en cas
    de faute de frappe dans `variable_names`. Si l'un des deux drapeaux est
    `True` mais que le symbole correspondant est absent de
    `variable_names`, une `ValueError` est levée immédiatement plutôt que
    de laisser l'erreur passer inaperçue.

    Quand les deux paramètres dépendent de la régression
    (`couple_theta0=couple_theta1=True`), la covariance OLS analytique
    Cov(θ₀, θ₁) — calculée par `linear_regression`, voir sa docstring — est
    transmise à `generate_bilan` via `covariances`, conformément au GUM
    §5.2 (formule 13) : l'incertitude composée n'est donc plus
    systématiquement sous-estimée par hypothèse d'indépendance.
    """
    if couple_theta0 and "theta0" not in variable_names:
        raise ValueError(
            "full_pipeline_regression_to_measurand : couple_theta0=True mais "
            "'theta0' n'apparaît pas dans variable_names="
            f"{variable_names}. Corrigez variable_names ou couple_theta0."
        )
    if couple_theta1 and "theta1" not in variable_names:
        raise ValueError(
            "full_pipeline_regression_to_measurand : couple_theta1=True mais "
            "'theta1' n'apparaît pas dans variable_names="
            f"{variable_names}. Corrigez variable_names ou couple_theta1."
        )

    reg      = linear_regression(x_data, y_data)
    nominals = {**nominal_values_helpers}
    inputs   = {**uncertainty_inputs_helpers}

    if couple_theta1:
        nominals["theta1"] = reg["theta1"]
        inputs["theta1"]   = UncertaintyInput(
            u=reg["u_theta1"], type="B", distribution="general", nu=reg["nu"]
        )
    if couple_theta0:
        nominals["theta0"] = reg["theta0"]
        inputs["theta0"]   = UncertaintyInput(
            u=reg["u_theta0"], type="B", distribution="general", nu=reg["nu"]
        )

    covariances = None
    if couple_theta0 and couple_theta1:
        covariances = {("theta0", "theta1"): reg["cov_theta01"]}

    tex_reg = generate_bilan_regression(
        x_symbol, y_symbol, x_unit, y_unit, x_data, y_data,
        slope_unit=slope_unit,
        intercept_unit=intercept_unit,
        slope_symbol=slope_symbol,
        intercept_symbol=intercept_symbol,
        global_sig_figs=global_sig_figs,
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
        global_sig_figs=global_sig_figs,
        covariances=covariances,
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
_SI_TOTHE_RE = re.compile(r"^\\tothe\{(-?\d+)\}$")


def _parse_unit(unit_str: str) -> tuple[list, list]:
    """
    Découpe une unité siunitx en deux listes [nom, puissance] : numérateur
    (avant \\per) et dénominateur (après \\per). Les préfixes (\\kilo,
    \\milli...) sont rattachés au token d'unité qui suit pour former un
    seul nom (ex: \\kilo\\gram -> "\\kilo\\gram"), les modificateurs de
    puissance préfixes (\\square, \\cubic) sont convertis en exposant
    entier, et le modificateur postfixe \\tothe{n} (puissance >= 4) vient
    corriger la puissance du dernier nom d'unité déjà ajouté à la liste
    courante, puisqu'il s'écrit après l'unité qu'il modifie (ex:
    \\second\\tothe{4}) et non avant comme \\square/\\cubic.

    La regex capture explicitement le bloc `{n}` qui suit \\tothe, y
    compris lorsque n est négatif (\\tothe{-2}) : sans cela, un nombre
    entre accolades serait silencieusement perdu et \\tothe réapparaîtrait
    comme un nom d'unité orphelin — ce qui casse la compilation dès
    qu'une unité de puissance >= 4 (positive OU négative, ex: une unité
    déjà inversée par un appel précédent) est réinjectée dans un nouvel
    appel à `_divide_units` (chaîne de mesurandes).

    Lève ValueError pour tout token non reconnu comme préfixe SI, token
    de puissance, \\per, \\tothe{n} ou unité connue de _KNOWN_SIUNITX_UNITS.

    Limite documentée (non corrigée) : cette algèbre est purement
    symbolique sur le nom du token. \\kilo\\gram et \\gram sont deux clés
    différentes : aucune conversion numérique entre préfixes SI d'une même
    unité de base n'est effectuée. Une unité composée du type
    \\kilo\\gram\\per\\gram peut donc apparaître non simplifiée si le
    numérateur et le dénominateur emploient des préfixes différents pour
    la même grandeur — sans erreur ni avertissement.
    """
    tokens = re.findall(r"\\[A-Za-z]+(?:\{-?\d+\})?", unit_str or "")
    numerator, denominator = [], []
    current = numerator
    pending_prefix, pending_power = "", 1
    for t in tokens:
        if t == _SI_PER:
            current = denominator
            # Un modificateur (préfixe ou puissance) non consommé avant \per
            # est orphelin : le laisser en l'état le ferait se coller
            # silencieusement au premier token du dénominateur.
            pending_prefix, pending_power = "", 1
            continue
        if t in _SI_PREFIX_TOKENS:
            pending_prefix += t
            continue
        if t in _SI_POWER_TOKENS:
            pending_power = _SI_POWER_TOKENS[t]
            continue
        m = _SI_TOTHE_RE.match(t)
        if m:
            if current:
                current[-1][1] = int(m.group(1))
            continue
        if t not in _KNOWN_SIUNITX_UNITS:
            raise ValueError(
                f"Token siunitx inconnu : {t!r}. "
                "Vérifiez l'orthographe ou ajoutez-le à _KNOWN_SIUNITX_UNITS."
            )
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


def _normalize_unit_signs(numerator: list, denominator: list) -> tuple[list, list]:
    """
    Renvoie (numérateur, dénominateur) avec des puissances strictement
    positives, en basculant côté dénominateur toute unité dont la
    puissance serait négative côté numérateur, et réciproquement.

    Une unité réinjectée avec un exposant négatif explicite (\\tothe{-2})
    peut être reconnue par `_parse_unit`, mais se retrouve alors dans la
    liste "numérateur" avec une puissance négative. `_render_unit_list`
    saurait techniquement l'afficher tel quel (\\tothe{-2} compile très
    bien), mais le résultat serait stylistiquement incohérent avec le
    reste du module, qui exprime toujours les puissances négatives via
    \\per côté dénominateur plutôt que via un \\tothe{} négatif côté
    numérateur. Cette fonction rétablit cette convention unique avant le
    rendu final.
    """
    num_clean, den_extra = [], []
    for name, power in numerator:
        (num_clean if power > 0 else den_extra).append([name, abs(power)])

    den_clean, num_extra = [], []
    for name, power in denominator:
        (den_clean if power > 0 else num_extra).append([name, abs(power)])

    return (
        _merge_same_units(num_clean + num_extra),
        _merge_same_units(den_clean + den_extra),
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
    temps) donnerait \\meter\\per\\second\\per\\second, qui compile sans
    erreur mais s'affiche "m/(s s)" au lieu de "m/s²". Cette fonction
    traite chaque unité comme une paire (numérateur, dénominateur),
    combine les deux quotients, fusionne les puissances d'un même nom et
    simplifie les unités communes, avant de reconstruire la chaîne
    siunitx finale (ex: \\meter\\per\\square\\second).

    Les unités composées à \\per multiples (ex:
    \\kilo\\gram\\per\\meter\\per\\second) sont gérées correctement, la
    réaffectation de `current` vers `denominator` étant idempotente au
    second \\per. Les unités déjà porteuses d'une puissance \\tothe{n}
    positive OU négative, issues d'un appel précédent (chaîne de
    mesurandes), sont également réinjectées proprement.
    """
    num_y, den_y = _parse_unit(numerator_unit)
    num_x, den_x = _parse_unit(denominator_unit)
    combined_num = _merge_same_units(num_y + den_x)
    combined_den = _merge_same_units(den_y + num_x)
    combined_num, combined_den = _cancel_units(combined_num, combined_den)
    combined_num, combined_den = _normalize_unit_signs(combined_num, combined_den)
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
    global_sig_figs: int = 3,
    _reg_precomputed: dict = None,
) -> str:
    reg   = _reg_precomputed if _reg_precomputed is not None else linear_regression(x_data, y_data)
    lines = []

    s_unit = slope_unit if slope_unit else _divide_units(y_unit, x_unit)
    i_unit = intercept_unit if intercept_unit else y_unit

    lines.append(rf"\subsection{{Bilan --- {_escape_latex(subsection_title)}}}")
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
    lines.append(rf"    {slope_symbol} = {_si(reg['theta1'], s_unit, sig=global_sig_figs)}")
    lines.append(r"\]")
    lines.append(r"\[")
    lines.append(rf"    {intercept_symbol} = {_si(reg['theta0'], i_unit, sig=global_sig_figs)}")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        rf"Les incertitudes-types associées "
        rf"($s^2_{{\mathrm{{res}}}} = {_sci(reg['s_res']**2, sig=global_sig_figs)}$) sont :"
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

    # On réutilise format_result (Partie 1) puis _format_result_uncertainty
    # (Partie 2) pour appliquer la même règle métrologique (le nombre de
    # décimales de la valeur est imposé par les 2 c.s. de son incertitude)
    # et le même rendu siunitx autosuffisant (separate-uncertainty=true
    # local) que pour le bilan GUM classique, plutôt que de reconstruire
    # une mise en forme \pm indépendante non synchronisée.
    fmt_intercept = format_result(reg['theta0'], reg['u_theta0'], sig_figs_exact=global_sig_figs)
    fmt_slope     = format_result(reg['theta1'], reg['u_theta1'], sig_figs_exact=global_sig_figs)
    intercept_expr = _format_result_uncertainty(
        fmt_intercept['result'], fmt_intercept['U'], i_unit, sig_figs_exact=global_sig_figs
    )
    slope_expr = _format_result_uncertainty(
        fmt_slope['result'], fmt_slope['U'], s_unit, sig_figs_exact=global_sig_figs
    )

    intercept_term = rf"\left({intercept_expr}\right)"
    slope_term     = rf"\left({slope_expr}\right)"
    box_inline = rf"{y_symbol} = {intercept_term} + {slope_term} \cdot {x_symbol}"

    lines.append(r"\[")
    if len(box_inline) > _INLINE_MAX_CHARS:
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
    body = "\n\n".join(bilans)
    return r"\section{Annexe : Bilans d'incertitudes}" + "\n\n" + body