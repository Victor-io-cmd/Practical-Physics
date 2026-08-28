"""
gum_calc.py — Moteur de calcul GUM

LIMITATIONS CONNUES
--------------------
- Les grandeurs d'entrée d'une même formule sont supposées indépendantes
  par défaut : `calculate_uncertainty` ne prend en compte une covariance
  entre deux entrées que si elle lui est explicitement fournie via le
  paramètre `covariances` (GUM §5.2, formule 13). Le cas le plus fréquent
  — un mesurande qui combine `theta0` et `theta1` issus de la même
  régression linéaire — est traité automatiquement par
  `gum_export.full_pipeline_regression_to_measurand`, qui calcule la
  covariance OLS analytique Cov(θ₀, θ₁) = -x̄·s²_res/Sxx et la transmet au
  pipeline. Toute autre source de covariance (deux mesurandes distincts
  dérivés d'une même régression, par exemple) reste à la charge de
  l'appelant.
- La propagation est strictement linéaire au premier ordre (dérivées
  partielles à la valeur nominale) : pas d'alternative Monte-Carlo
  (GUM Supplément 1) pour les modèles fortement non linéaires.
- `linear_regression` n'implémente pas de régression pondérée : les
  points de mesure sont supposés tous porter la même incertitude sur y.
- `compatibility_test` (test de compatibilité théorie/mesure par variable
  de Student réduite) suppose la valeur théorique de référence exacte :
  toute incertitude propre à cette valeur théorique elle-même n'entre pas
  dans le calcul du risque associé.
- `nonlinear_regression` retourne la matrice de covariance asymptotique de
  `scipy.optimize.curve_fit` (linéarisation locale du modèle au voisinage
  de l'optimum) : comme `linear_regression`, elle n'est pas pondérée par
  défaut (`sigma=None`) et ne couvre aucun cas fortement non identifiable
  (paramètres corrélés à 100 %).

ARCHITECTURE
------------
Ce module ne fait AUCUN calcul lié à la rédaction : il prend des mesures
et des incertitudes en entrée, et renvoie des dicts de nombres (résultat,
incertitude-type, sensibilités, degrés de liberté...). Aucune fonction
d'ici ne produit de chaîne LaTeX.

La mise en forme LaTeX/siunitx (bilans rédigés, `\\boxed{}`, algèbre
d'unités siunitx) vit dans `gum_export.py`, qui importe ce module. Un
programme qui n'a besoin que des valeurs numériques (interface graphique,
notebook d'exploration, tracé de courbes) peut donc dépendre uniquement
de `gum_calc.py`, sans jamais importer `gum_export.py`.
"""

import math
import warnings
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from types import MappingProxyType
import numpy as np
import sympy as sp
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

# ============================================================
# PARTIE 1 — MOTEUR DE CALCUL GUM
# ============================================================


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


def _k_factor(nu_eff: float) -> float:
    """
    Facteur d'élargissement k (intervalle de confiance à 95 %, test
    bilatéral) à partir des degrés de liberté effectifs nu_eff.

    k = 2.0 si nu_eff est infini (repli loi normale) ; sinon quantile
    0.975 de la loi de Student à nu_eff degrés de liberté, arrondi à
    0.001 par _round_half_up (et non round(), qui applique l'arrondi
    bancaire Python — cohérence avec le reste du module).

    Factorisée hors de `welch_satterthwaite` pour être réutilisable par
    `generate_bilan_compatibilite`, qui a besoin du même k pour afficher
    U = k·u_c dans sa phrase d'introduction sans dupliquer la logique ni
    risquer une divergence future entre les deux calculs.
    """
    if not math.isfinite(nu_eff):
        return 2.0
    k_raw = scipy_stats.t.ppf(0.975, df=nu_eff)
    if not math.isfinite(k_raw):
        warnings.warn(
            f"_k_factor : facteur k non fini (nu_eff={nu_eff:.3g}). "
            "Les degrés de liberté effectifs sont anormalement proches de 0 — "
            "on retombe sur k = 2.0 par sécurité. Vérifiez les nu fournis.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 2.0
    return _round_half_up(k_raw, 0.001)


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
    else:
        nu_eff = (uc_squared ** 2) / denominator

    return {"nu_eff": nu_eff, "k": _k_factor(nu_eff)}


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


def nonlinear_regression(
    x_data: list[float],
    y_data: list[float],
    model_func,
    p0: list[float],
    param_names: list[str],
    sigma: list[float] = None,
) -> dict:
    """
    Régression non linéaire par moindres carrés (scipy.optimize.curve_fit).

    Modèle : y = model_func(x, *params), avec `model_func` un callable
    Python pur de signature scipy-compatible — ex, pour une décharge RC :
        model_func = lambda t, V0, tau: V0 * np.exp(-t / tau)
        param_names = ["V0", "tau"]

    Analogue non affine de `linear_regression` : mêmes familles de clés en
    retour (params/u_params plutôt que theta0/theta1, nu = N - p plutôt
    que N - 2), pour rester lisible par comparaison directe avec le cas
    affine. Aucune des deux fonctions n'appelle l'autre : `linear_regression`
    reste le cas affine dédié, validé sur des CR réels de L2, et n'est pas
    modifiée par l'ajout de celle-ci.

    Paramètres
    ----------
    x_data, y_data : listes de même longueur (N > nombre de paramètres,
                     pour que nu = N - p soit strictement positif).
    model_func     : callable(x, *params) -> y, signature scipy-compatible.
                     Reçoit x_data sous forme de tableau numpy (pas une
                     liste Python) à l'intérieur de curve_fit : utiliser
                     des opérations numpy (np.exp, np.sin, np.sqrt...),
                     jamais le module math, sous peine de TypeError.
    p0             : estimation initiale des paramètres, même ordre que
                     param_names. Un p0 trop éloigné de la solution est la
                     cause la plus fréquente de non-convergence.
    param_names    : noms des paramètres, même ordre que p0 et que la
                     signature de model_func après x.
    sigma          : incertitude-type sur y, point par point (liste de
                     longueur N) — None par défaut (régression non
                     pondérée). Si fourni, curve_fit est appelé avec
                     absolute_sigma=True : la covariance retournée reflète
                     alors directement les incertitudes fournies
                     (propagation Type B classique, GUM §5.1.2). Si None,
                     absolute_sigma=False : la covariance est mise à
                     l'échelle par le chi² réduit des résidus — incertitude
                     estimée empiriquement à partir de la dispersion des
                     données elles-mêmes, analogue à s²_res dans
                     `linear_regression`.

    Retourne un dict avec :
      params      — dict {nom: valeur estimée}
      u_params    — dict {nom: incertitude-type}, sqrt(diag(pcov))
      cov         — matrice de covariance complète (pcov de curve_fit),
                    dans l'ordre de param_names — à fournir à
                    `calculate_uncertainty`/`generate_bilan` (paramètre
                    `covariances`) pour tout mesurande combinant deux
                    paramètres corrélés (fréquent entre amplitude et
                    taux de décroissance d'une exponentielle)
      nu          — degrés de liberté, N - p (p = nombre de paramètres)
      residuals, y_pred, r2
      model_func, param_names — conservés pour que plot_courbes puisse
                    retracer la courbe sans recalculer l'ajustement
      N           — nombre de points de mesure
    """
    N = len(x_data)
    if N != len(y_data):
        raise ValueError("nonlinear_regression : x_data et y_data doivent avoir la même longueur.")
    if len(p0) != len(param_names):
        raise ValueError(
            f"nonlinear_regression : p0 (longueur {len(p0)}) et param_names "
            f"(longueur {len(param_names)}) doivent avoir la même longueur."
        )
    p = len(param_names)
    if N <= p:
        raise ValueError(
            f"nonlinear_regression : N = {N} points pour p = {p} paramètres — "
            "il faut strictement plus de points de mesure que de paramètres "
            "libres (nu = N - p doit être positif)."
        )

    for label, data in (("x_data", x_data), ("y_data", y_data)):
        bad = [v for v in data if not math.isfinite(v)]
        if bad:
            raise ValueError(
                f"nonlinear_regression : {label} contient des valeurs non "
                f"finies ({bad[:3]}). Nettoyez les données avant l'ajustement."
            )

    if sigma is not None:
        if len(sigma) != N:
            raise ValueError(
                f"nonlinear_regression : sigma (longueur {len(sigma)}) doit "
                f"avoir la même longueur que x_data/y_data (N = {N})."
            )
        bad_sigma = [v for v in sigma if not math.isfinite(v) or v <= 0]
        if bad_sigma:
            raise ValueError(
                "nonlinear_regression : sigma doit être strictement positif "
                f"et fini pour chaque point ({bad_sigma[:3]} invalide(s))."
            )

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    try:
        popt, pcov = curve_fit(
            model_func, x_arr, y_arr, p0=p0, sigma=sigma,
            absolute_sigma=(sigma is not None), maxfev=10000,
        )
    except RuntimeError as exc:
        raise ValueError(
            "nonlinear_regression : l'ajustement n'a pas convergé "
            "(scipy.optimize.curve_fit). Cause probable : p0 trop éloigné "
            "de la solution, ou modèle mal posé pour ces données. Message "
            f"scipy d'origine : {exc}."
        ) from exc
    except TypeError as exc:
        raise ValueError(
            "nonlinear_regression : signature de model_func incompatible "
            f"avec param_names (p = {p}). Vérifiez que model_func(x, "
            f"*params) accepte exactement {p} paramètre(s) après x. "
            f"Message d'origine : {exc}."
        ) from exc

    if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(pcov)):
        raise ValueError(
            "nonlinear_regression : résultat non fini en sortie de "
            "curve_fit (paramètres ou covariance). Modèle probablement "
            "non identifiable avec ces données (paramètres corrélés à "
            "100 %, ou p0 sur un point singulier du modèle)."
        )

    # Clamp cancellation numérique (O(1e-16)) sur la diagonale, même garde-
    # fou que s2_res dans linear_regression : math.sqrt lèverait sinon.
    u_params_arr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

    y_pred    = np.asarray(model_func(x_arr, *popt), dtype=float)
    residuals = y_arr - y_pred
    ss_res    = float(np.sum(residuals ** 2))
    y_mean    = float(np.mean(y_arr))
    syy       = float(np.sum((y_arr - y_mean) ** 2))
    if syy > 0:
        r2 = 1.0 - ss_res / syy
    else:
        # Toutes les yi identiques : cf. avertissement identique dans
        # linear_regression, même justification.
        warnings.warn(
            "nonlinear_regression : toutes les valeurs yi sont identiques "
            "(syy=0) — r² n'a pas de sens prédictif, fixé à NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        r2 = float("nan")

    return {
        "params":      dict(zip(param_names, (float(v) for v in popt))),
        "u_params":    dict(zip(param_names, (float(v) for v in u_params_arr))),
        "cov":         pcov,
        "nu":          N - p,
        "N":           N,
        "residuals":   residuals.tolist(),
        "y_pred":      y_pred.tolist(),
        "r2":          r2,
        "model_func":  model_func,
        "param_names": list(param_names),
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
# MISE EN FORME NUMÉRIQUE (décimale / scientifique)
# ============================================================
#
# Ces fonctions ne produisent aucun LaTeX : `format_magnitude` est
# directement utilisable pour un affichage console, une légende
# matplotlib, ou tout autre contexte texte brut. `gum_export.py` les
# réutilise telles quelles (`_format_magnitude`) comme brique commune
# pour ses propres macros siunitx (`_num`, `_si`), afin que la même
# règle de bascule décimale/scientifique s'applique partout dans le
# projet sans y être dupliquée.
# ============================================================

_SCI_NOTATION_MAG_MIN = -2   # en dessous : notation scientifique
_SCI_NOTATION_MAG_MAX = 3    # au-dessus  : notation scientifique

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


def format_magnitude(value: float, sig: int, mathtext: bool = False) -> str:
    """
    Version publique de `_format_magnitude`, utilisable hors contexte LaTeX
    (ex : légende matplotlib d'app.py). Même règle de bascule que le reste
    des bilans GUM : notation décimale pour -2 <= magnitude <= 3, notation
    scientifique mantisse*10^exposant sinon (seuils `_SCI_NOTATION_MAG_MIN`
    / `_SCI_NOTATION_MAG_MAX`).

    `mathtext=True` réécrit la branche scientifique en syntaxe mathtext
    matplotlib (`mantisse \\times 10^{exposant}`) au lieu de la notation
    Python "e", pour un rendu correct dans une légende entourée de $...$ —
    matplotlib ne comprend pas les macros siunitx `\\num{}` / `\\qty{}`
    utilisées par `_num`/`_si` pour l'export LaTeX.
    """
    raw = _format_magnitude(value, sig)
    if not mathtext or "e" not in raw:
        return raw
    mantissa_str, exp_str = raw.split("e")
    return rf"{mantissa_str} \times 10^{{{int(exp_str)}}}"


# ============================================================
# TEST DE COMPATIBILITÉ THÉORIE / MESURE
# ============================================================
#
# Répond à une question différente de `full_gum_analysis` : non pas
# "quelle est l'incertitude de ce mesurande ?" mais "ce résultat est-il
# compatible avec une valeur théorique de référence ?". Méthode issue du
# cours de métrologie L2/L3 (UPEC, N. Bochud — "Déclaration de conformité
# et incertitude de mesure") : comparaison par variable de Student
# réduite t(nu_eff) = |y - y_theo| / u_c(y), lue via la loi de Student
# (ou normale si nu_eff est infini) pour estimer un risque associé au
# rejet de la compatibilité.
#
# Callable sur un mesurande direct (uc=res["uc"], nu_eff=res["nu_eff"])
# ou sur un paramètre de régression (uc=reg["u_theta0"/"u_theta1"],
# nu_eff=reg["nu"]) : les deux usages se contentent de transmettre uc et
# nu_eff déjà produits ailleurs dans le module.
#
# `gum_export.generate_bilan_compatibilite` rédige le paragraphe LaTeX
# correspondant à partir du dict renvoyé ici — cette fonction-ci reste
# pure numérique.
# ============================================================

# Risque bilatéral (%) en dessous duquel la mesure est déclarée
# incompatible avec la valeur théorique. Seuil cohérent avec le facteur
# k=2 (~95% de confiance) retenu par défaut ailleurs dans le module
# (cf. welch_satterthwaite, scipy_stats.t.ppf(0.975, ...)).
_COMPATIBILITY_RISK_THRESHOLD = 5.0


def compatibility_test(
    y_mesure: float,
    y_theorique: float,
    uc: float,
    nu_eff: float,
) -> dict:
    """
    Test de compatibilité théorie/mesure par variable de Student réduite.

    t(nu_eff) = |y_mesure - y_theorique| / u_c

    Le risque associé (test bilatéral) est la probabilité, sous
    l'hypothèse que y_mesure et y_theorique sont compatibles, d'observer
    un écart au moins aussi grand que celui mesuré. Il est lu par le
    complément à 2 queues de la loi de Student à nu_eff degrés de
    liberté (scipy.stats.t.sf), ou de la loi normale si nu_eff est
    infini — cas limite de la loi de Student, cohérent avec le repli
    k=2.00 utilisé par welch_satterthwaite dans la même situation.

    Paramètres
    ----------
    y_mesure    : valeur mesurée — mesurande direct (res["result"]) ou
                  paramètre de régression (reg["theta0"]/reg["theta1"]).
    y_theorique : valeur théorique de référence à laquelle comparer
                  y_mesure. Supposée exacte : toute incertitude propre à
                  y_theorique elle-même n'est pas prise en compte ici
                  (voir LIMITATIONS CONNUES en tête de module).
    uc          : incertitude-type composée de y_mesure — res["uc"]
                  (mesurande direct) ou reg["u_theta0"]/reg["u_theta1"]
                  (paramètre de régression).
    nu_eff      : degrés de liberté effectifs associés à uc —
                  res["nu_eff"] (mesurande direct, Welch-Satterthwaite)
                  ou reg["nu"] (régression, = N - 2).

    Retourne un dict avec :
      s          — écart relatif |y_mesure - y_theorique| / |y_theorique|,
                   None si y_theorique = 0 (écart relatif non défini).
      ecart_abs  — écart absolu |y_mesure - y_theorique|.
      t_stat     — variable de Student réduite t(nu_eff). Vaut float('inf')
                   si uc = 0 et ecart_abs > 0 (incertitude nulle mais
                   désaccord non nul), 0.0 si uc = 0 et ecart_abs = 0.
      nu_eff     — repris tel quel, pour affichage dans le paragraphe LaTeX.
      risk       — risque associé en %, borné à [0, 100].
      compatible — bool, True si risk >= _COMPATIBILITY_RISK_THRESHOLD.
    """
    if uc < 0:
        raise ValueError(f"compatibility_test : uc doit être positif ou nul : {uc}")
    if not math.isfinite(y_mesure) or not math.isfinite(y_theorique):
        raise ValueError(
            "compatibility_test : y_mesure et y_theorique doivent être "
            f"finis (reçu y_mesure={y_mesure}, y_theorique={y_theorique})."
        )

    ecart_abs = abs(y_mesure - y_theorique)
    s = (ecart_abs / abs(y_theorique)) if y_theorique != 0 else None

    if uc == 0:
        t_stat = float("inf") if ecart_abs > 0 else 0.0
    else:
        t_stat = ecart_abs / uc

    if t_stat == 0.0:
        risk = 100.0
    elif not math.isfinite(t_stat):
        risk = 0.0
    elif math.isfinite(nu_eff):
        risk = 2.0 * scipy_stats.t.sf(t_stat, df=nu_eff) * 100.0
    else:
        risk = 2.0 * scipy_stats.norm.sf(t_stat) * 100.0

    return {
        "s":          s,
        "ecart_abs":  ecart_abs,
        "t_stat":     t_stat,
        "nu_eff":     nu_eff,
        "risk":       risk,
        "compatible": risk >= _COMPATIBILITY_RISK_THRESHOLD,
    }