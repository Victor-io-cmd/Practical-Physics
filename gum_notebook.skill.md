---
name: gum-notebook
description: >
  Génère un notebook Jupyter GUM complet pour les comptes rendus de TP L3 Physique.
  S'active sur "mode incertitudes". Produit un .ipynb prêt à tourner, câblé sur
  gum_calc.py (moteur de calcul) et gum_export.py (rédaction LaTeX), avec une
  cellule par mesurande, régression affine ou non linéaire (exponentielle,
  sinusoïde...), et l'export LaTeX final.
triggers:
  - "mode incertitudes"
  - "calcule les incertitudes"
  - "génère le notebook GUM"
  - grandeurs + formules + instruments dans un contexte de TP
output: fichier .ipynb valide nommé GUM_<sujet_TP>.ipynb
---

# SKILL — gum_notebook
## Génération automatique de notebooks Jupyter GUM pour comptes rendus L3 Physique

---

## 1. DÉCLENCHEMENT

Ce skill s'active quand l'étudiant demande le **mode incertitudes** dans le contexte d'un TP.
Formulations typiques :
- "mode incertitudes", "calcule les incertitudes", "génère le notebook GUM"
- mention de grandeurs + formules + instruments dans un contexte de TP

Résultat livré : un fichier `.ipynb` complet, nommé `GUM_<sujet_TP>.ipynb`, prêt à être exécuté localement.

---

## 2. ARCHITECTURE DU MOTEUR

Le moteur repose sur **deux modules distincts**, ni l'un ni l'autre ne doit jamais être réécrit ni simulé dans le notebook. Toute la logique passe par leurs fonctions publiques ; ce skill génère **uniquement** le notebook qui les appelle.

| Module | Rôle | Contenu |
|---|---|---|
| `gum_calc.py` | Moteur de calcul pur | sources d'incertitude, propagation GUM, Welch-Satterthwaite, régressions linéaire/non linéaire, test de compatibilité, arrondi métrologique. Ne produit **aucune** chaîne LaTeX. |
| `gum_export.py` | Rédaction LaTeX | met en forme, en siunitx, les résultats déjà calculés par `gum_calc.py` (qu'il importe). `gum_calc.py` ne dépend jamais de `gum_export.py`. |

**Conséquence directe pour le notebook généré** : l'import se fait en **deux blocs séparés**, un par module (voir Cellule HEADER, §4) — jamais un seul `from gum_calc import (...)` regroupant les fonctions des deux fichiers. Le `sys.path.insert` reste unique (les deux fichiers vivent dans le même dossier), mais les fonctions de rédaction LaTeX (`generate_bilan*`, `generate_annexe`, `full_pipeline_regression_to_measurand`) s'importent depuis `gum_export`, jamais depuis `gum_calc`.

### Fonctions disponibles et signatures exactes

```python
# ============================================================
# gum_calc.py — MOTEUR DE CALCUL (aucune fonction ne renvoie de LaTeX)
# ============================================================

# --- Sources d'incertitude ---

uncertainty_type_A(values: list[float]) -> UncertaintyInput
# Type A par répétition. Requiert N >= 2.
# Attributs accessibles : .u, .nu, .N, .s, .mean
# NE PAS écrire u_X["u"] ou u_X["mean"] → TypeError garanti.
# Accès par attribut UNIQUEMENT : u_X.u, u_X.mean, u_X.s, u_X.N, u_X.nu

uncertainty_type_B_from_resolution(resolution: float) -> UncertaintyInput
# Type B depuis résolution δ d'un instrument numérique.
# u_B = (δ/2) / sqrt(3)
# Attributs : .u, .a (demi-largeur), .nu = inf

uncertainty_type_B_uniform(half_width: float) -> UncertaintyInput
# Type B distribution uniforme, demi-largeur a connue directement.
# u_B = a / sqrt(3)

uncertainty_type_B_relative(u_standard: float, relative_knowledge: float = None) -> UncertaintyInput
# Type B — u fourni directement (calibration, notice, ou propagation en chaîne).
# ATTENTION : passer u_c, jamais U = k·u_c.
# relative_knowledge optionnel : si fourni, nu = 1/(2r²) ; sinon nu = inf.

uncertainty_type_exact() -> UncertaintyInput
# Constante exacte : u = 0, nu = inf.

# --- Analyse complète ---

full_gum_analysis(
    formula_str: str,           # expression SymPy, ex: "U / I"
    variable_names: list[str],  # liste ordonnée des noms Python
    nominal_values: dict,       # {nom: valeur float}
    uncertainty_inputs: dict,   # {nom: dict uncertainty_type_*}
    k_override: float = None,   # force k si fourni, sinon Welch-Satterthwaite auto
    covariances: dict[tuple[str, str], float] = None,  # {(nom_i,nom_j): u(x_i,x_j)}, GUM §5.2 formule 13
) -> dict
# Retourne : result, uc, uc_squared, sensitivities, contributions,
#            covariance_terms, budget, partial_derivs, nu_eff, k, U,
#            result_rounded, U_rounded, decimals

# --- Régressions (linéaire ET non linéaire) ---

linear_regression(x_data: list[float], y_data: list[float]) -> dict
# Modèle y = theta0 + theta1 * x, moindres carrés ordinaires. N >= 3.
# Retourne : theta0, theta1, u_theta0, u_theta1, cov_theta01
#            (= Cov(theta0,theta1), covariance OLS analytique), s_res,
#            r2, N, nu (= N-2), y_pred, residuals

nonlinear_regression(
    x_data: list[float],
    y_data: list[float],
    model_func,                 # callable(x, *params) -> y — OPÉRATIONS NUMPY
                                 # UNIQUEMENT (np.exp, np.sin...), jamais math.*
    p0: list[float],            # estimation initiale, même ordre que param_names
    param_names: list[str],     # ex: ["V0", "tau"]
    sigma: list[float] = None,  # incertitude sur y point par point (optionnel)
) -> dict
# Retourne : params (dict {nom: valeur}), u_params (dict {nom: incertitude-type}),
#            cov (matrice pcov complète, ordre de param_names), nu (= N - p),
#            N, residuals, y_pred, r2, model_func, param_names

# --- Test de compatibilité théorie/mesure (calcul pur, pas de LaTeX) ---

compatibility_test(
    y_mesure: float,      # valeur mesurée : res["result"] ou reg["theta0"/"theta1"]
    y_theorique: float,   # valeur théorique de référence, supposée exacte
    uc: float,            # res["uc"] (mesurande direct) ou reg["u_theta0"/"u_theta1"]
    nu_eff: float,        # res["nu_eff"] (mesurande direct) ou reg["nu"] (régression)
) -> dict
# Retourne s, ecart_abs, t_stat, nu_eff, risk (%), compatible (bool).
# Se contente de comparer un couple (uc, nu_eff) déjà produit ailleurs
# à une valeur théorique — aucun recalcul de propagation ici.

# ============================================================
# gum_export.py — RÉDACTION LATEX (importe gum_calc, jamais l'inverse)
# ============================================================

generate_bilan(
    measurand_name: str,        # nom en toutes lettres, ex: "Résistance"
    measurand_symbol: str,      # symbole LaTeX, ex: "R"
    formula_str: str,
    variable_names: list[str],
    variable_symbols: dict,     # {nom_python: "symbole_LaTeX"}
    variable_units: dict,       # {nom_python: "unité_siunitx"}
    nominal_values: dict,
    uncertainty_inputs: dict,
    measurand_unit: str = "",   # unité siunitx du mesurande, ex: r"\ohm"
    k_override: float = None,
    subsection: bool = True,
    global_sig_figs: int = 3,
    covariances: dict[tuple[str, str], float] = None,  # transmis tel quel à full_gum_analysis
    _res_precomputed = None,    # réutilise un full_gum_analysis déjà calculé — voir §4
) -> str  # bloc LaTeX complet

generate_bilan_regression(
    x_symbol: str, y_symbol: str,
    x_unit: str, y_unit: str,
    x_data: list[float], y_data: list[float],
    subsection_title: str = "Régression linéaire",
    slope_unit: str = "",
    intercept_unit: str = "",
    slope_symbol: str = r"\theta_1",
    intercept_symbol: str = r"\theta_0",
    global_sig_figs: int = 3,
    functional_notation: bool = False,
    _reg_precomputed: dict = None,
) -> str

full_pipeline_regression_to_measurand(
    x_data, y_data,
    x_symbol, y_symbol, x_unit, y_unit,
    formula_str,
    variable_names,             # doit contenir "theta1" et/ou "theta0"
    variable_symbols, variable_units,
    nominal_values_helpers,     # valeurs des variables hors régression
    uncertainty_inputs_helpers, # incertitudes des variables hors régression
    measurand_symbol, measurand_name="", measurand_unit="",
    slope_unit="", intercept_unit="",
    slope_symbol=r"\theta_1", intercept_symbol=r"\theta_0",
    global_sig_figs: int = 3,
    couple_theta0: bool = False,
    couple_theta1: bool = False,
    functional_notation: bool = False,
) -> str  # LaTeX régression + GUM en un seul appel — voir 3.4ter-a

generate_bilan_nonlinear_regression(
    x_symbol: str, y_symbol: str,
    x_unit: str, y_unit: str,
    x_data: list[float], y_data: list[float],
    model_func, p0: list[float], param_names: list[str],
    param_symbols: dict,        # {nom_python: "symbole_LaTeX"}
    param_units: dict,          # {nom_python: "unité_siunitx"}
    model_latex: str,           # gabarit avec UN '{}' par paramètre, MÊME ORDRE
    subsection_title: str = "Régression non linéaire",
    global_sig_figs: int = 3,
    functional_notation: bool = False,
    sigma: list[float] = None,
    _reg_precomputed: dict = None,
) -> str  # bloc LaTeX complet

generate_annexe(bilans: list[str]) -> str
# Assemble tous les bilans dans \section{Annexe : Bilans d'incertitudes}
# (le titre de section n'est pas généré ici, le template le porte déjà).

generate_bilan_compatibilite(
    measurand_symbol: str,      # symbole LaTeX, ex: "R" ou r"\theta_1"
    y_mesure: float,
    uc: float,
    nu_eff: float,
    y_theorique: float,         # paramètre obligatoire — distingue cet usage de generate_bilan
    measurand_unit: str = "",
    measurand_name: str = "",
    subsection: bool = False,
    subsection_title: str = "Confrontation à la valeur théorique",
    global_sig_figs: int = 3,
    k: float = None,             # facteur k déjà connu (mesurande direct) — sinon recalculé via nu_eff
    U: float = None,             # incertitude élargie déjà connue — sinon U = k * uc
    annexe_ref: bool = True,     # si True, la phrase d'intro renvoie à l'annexe pour le détail de l'incertitude
    _compat_precomputed: dict = None,  # réutilise un compatibility_test déjà calculé
) -> str  # paragraphe LaTeX : rappel de la mesure, écart relatif s, t(nu_eff), risque en %, conclusion
```

---

## 3. RÈGLES DE GÉNÉRATION DU NOTEBOOK

### 3.1 Structure invariante

Le notebook généré contient **toujours** ces cellules dans cet ordre :

1. **Cellule HEADER** — imports + sys.path (chemin à adapter par l'étudiant)
2. **N cellules MESURANDE** — une par grandeur calculée
3. **Cellules COMPATIBILITÉ (optionnelles)** — une par grandeur confrontée à une valeur théorique, insérée juste après la cellule MESURANDE (ou de régression) concernée, dès que l'étudiant fournit ou demande une valeur théorique de référence
4. **Cellule LATEX** — `generate_annexe([bilan_1, bilan_2, ...])`, qui n'inclut jamais les blocs de compatibilité

### 3.2 Nommage des variables Python

Convention stricte : `suffixe_grandeur` où le suffixe identifie la grandeur.

- `u_R` pour une incertitude sur R
- `nominales_R`, `incertitudes_R` pour les dicts de R
- `res_R` pour le résultat de `full_gum_analysis` sur R
- `bilan_R` pour le résultat de `generate_bilan` sur R

Les noms Python doivent rester cohérents avec les `variable_names` passés aux fonctions.

### 3.3 Choix de la fonction d'incertitude

| Situation | Fonction à appeler |
|---|---|
| N mesures répétées (N ≥ 2) | `uncertainty_type_A(valeurs)` |
| Instrument numérique, résolution δ connue | `uncertainty_type_B_from_resolution(δ)` |
| Demi-largeur a connue directement | `uncertainty_type_B_uniform(a)` |
| u fourni par calibration ou notice | `uncertainty_type_B_relative(u_standard=u)` |
| Propagation depuis une étape précédente | `uncertainty_type_B_relative(u_standard=res_X["uc"])` — **jamais res_X["U"]** |
| Constante physique fondamentale | `uncertainty_type_exact()` |

**Accès aux attributs de `UncertaintyInput` (dataclass — jamais un dict) :**

```python
# CORRECT — accès par attribut
u_X.u     # incertitude-type
u_X.nu    # degrés de liberté
u_X.N     # nombre de mesures (type A seulement)
u_X.s     # écart-type empirique (type A seulement)
u_X.mean  # moyenne empirique (type A seulement) — utiliser pour nominales_X

# INTERDIT — lève TypeError garanti
u_X["u"]      # ← TypeError
u_X["mean"]   # ← TypeError
```

**Règle pour les valeurs nominales de type A :**
```python
# CORRECT — mean synchronisé avec la liste de mesures
X_values = [1.23, 1.24, 1.22]
u_X = uncertainty_type_A(X_values)
nominales_Y = {"X": u_X.mean, ...}   # jamais un littéral codé en dur

# INTERDIT — valeur pouvant diverger silencieusement si on change les mesures
nominales_Y = {"X": 1.23, ...}   # ← désynchronisé dès qu'on édite X_values
```

### 3.4 Régression linéaire

Deux cas :

**Cas A — Régression seule (pas de mesurande dérivé) :**
```python
bilan_reg = generate_bilan_regression(
    x_symbol="t", y_symbol="U",
    x_unit=r"\second", y_unit=r"\volt",
    x_data=[...], y_data=[...],
    slope_symbol=r"\alpha", intercept_symbol=r"\beta",
)
```

**Cas B — Régression + mesurande dérivé :**
```python
bilan_complet = full_pipeline_regression_to_measurand(
    x_data=[...], y_data=[...],
    x_symbol="t", y_symbol="U",
    x_unit=r"\second", y_unit=r"\volt",
    formula_str="theta1 / (2 * pi * C)",  # theta1 = pente, theta0 = ordonnée
    variable_names=["theta1", "C"],
    variable_symbols={"theta1": r"\alpha", "C": "C"},
    variable_units={"theta1": r"\volt\per\second", "C": r"\farad"},
    nominal_values_helpers={"C": 1e-6},
    uncertainty_inputs_helpers={"C": uncertainty_type_B_from_resolution(1e-9)},
    measurand_symbol="f_0",
    measurand_unit=r"\hertz",
    slope_symbol=r"\alpha",
    couple_theta1=True,   # OBLIGATOIRE dès que formula_str contient theta1 — voir 3.4ter
)
```

### 3.4bis Test de compatibilité théorie/mesure

Fonction distincte de `generate_bilan` : elle répond à "ce résultat est-il compatible avec une valeur théorique de référence ?", pas à "quelle est son incertitude ?".

**Critère de routage** : la présence d'une valeur théorique à comparer (`y_theorique`) détermine seule quelle fonction appeler.

| Présence de `y_theorique` | Fonction | Placement dans le notebook |
|---|---|---|
| Non | `generate_bilan` | Cellule MESURANDE, résultat empilé dans le bloc `generate_annexe` final |
| Oui | `generate_bilan_compatibilite` | Cellule de la grandeur concernée (ou de la courbe, juste après `linear_regression`/`plot_regression`), **jamais** dans `generate_annexe` |

**Callable sur deux natures de résultat**, sans recalcul :

```python
# Cas A — mesurande direct (sortie de full_gum_analysis)
compat_R = generate_bilan_compatibilite(
    measurand_symbol = "R",
    measurand_name   = "Résistance",
    y_mesure         = res_R["result"],
    uc               = res_R["uc"],
    nu_eff           = res_R["nu_eff"],
    y_theorique      = R_theorique,
    measurand_unit   = r"\ohm",
)
print(compat_R)

# Cas B — paramètre de régression (sortie de linear_regression)
reg = linear_regression(x_data, y_data)
compat_theta1 = generate_bilan_compatibilite(
    measurand_symbol = r"\theta_1",
    y_mesure         = reg["theta1"],
    uc               = reg["u_theta1"],
    nu_eff           = reg["nu"],
    y_theorique      = pente_theorique,
    measurand_unit   = r"<unité>",
)
print(compat_theta1)
```

Contenu généré, toujours dans cet ordre : écart relatif $s$, variable de Student réduite $t(\nu_{\text{eff}})$ avec le degré de liberté utilisé, risque associé en % (test bilatéral, `scipy.stats.t.sf` ou loi normale si $\nu_{\text{eff}} \to \infty$), conclusion explicite de compatibilité ou de rejet.

**Règle absolue** : `print(compat_<G>)` s'affiche directement dans sa propre cellule (ou juste après), il n'entre **jamais** dans la liste passée à `generate_annexe`.

### 3.4ter-a Piège n°1 — `couple_theta0` / `couple_theta1` sont **obligatoires**, jamais implicites

`full_pipeline_regression_to_measurand` ne construit **pas automatiquement** `nominal_values`/`uncertainty_inputs` à partir de la régression. En interne, la fonction part de `nominals = {**nominal_values_helpers}` (donc uniquement ce que l'appelant fournit dans les helpers), puis n'y injecte `theta1`/`theta0` — et leur `UncertaintyInput` associé — que si le drapeau correspondant vaut `True` :

```python
if couple_theta1:
    nominals["theta1"] = reg["theta1"]
    inputs["theta1"]   = UncertaintyInput(u=reg["u_theta1"], type="B", distribution="general", nu=reg["nu"])
if couple_theta0:
    nominals["theta0"] = reg["theta0"]
    inputs["theta0"]   = UncertaintyInput(u=reg["u_theta0"], type="B", distribution="general", nu=reg["nu"])
```

**Règle absolue** : dès que `formula_str` référence `"theta1"`, l'appel doit obligatoirement porter `couple_theta1=True` ; de même pour `"theta0"` avec `couple_theta0=True`. Omettre le drapeau alors que `variable_names` contient `"theta1"` ou `"theta0"` produit un `ValueError: Valeur nominale manquante pour : ['theta1']` (ou `'theta0'`) au moment de l'appel — jamais un résultat silencieusement faux, mais l'erreur ne pointe pas vers sa cause réelle (le drapeau manquant), donc **avant de générer toute cellule utilisant `full_pipeline_regression_to_measurand`, vérifier explicitement** : si `"theta1" in variable_names` → `couple_theta1=True` ; si `"theta0" in variable_names` → `couple_theta0=True`.

### 3.4ter-a-bis Rappel — type de retour de `full_pipeline_regression_to_measurand`

Cette fonction retourne **une seule valeur : le bloc LaTeX complet (`-> str`)**, jamais un tuple. Elle ne renvoie ni `res` (sortie de `full_gum_analysis` sur le mesurande) ni `reg` (sortie de `linear_regression`) : ces deux objets sont calculés en interne et ne sortent pas de la fonction.

```python
# CORRECT
bilan_h = full_pipeline_regression_to_measurand(...)   # une seule variable

# INTERDIT — lève ValueError: too many values to unpack
res_h, reg_h, bilan_h = full_pipeline_regression_to_measurand(...)
```

**Conséquence directe** : si une cellule COMPATIBILITÉ ou toute autre cellule ultérieure a besoin de `res` (pour `uc`, `nu_eff`, `result`) ou de `reg` (pour `theta1`, `u_theta1`, `nu`) après un appel à `full_pipeline_regression_to_measurand`, ces objets doivent être **recalculés séparément** dans cette cellule — voir 3.4ter-d pour la procédure complète et le piège associé.

### 3.4ter-b Piège n°2 — sens de la pente dans une régression inversée

`linear_regression(x_data, y_data)` ajuste toujours le modèle $y = \theta_0 + \theta_1 x$, donc $\theta_1$ a pour unité (unité de $y$)/(unité de $x$) — **jamais l'inverse**. Si le mesurande final dépend physiquement de $x/y$ plutôt que de $y/x$ (typique d'une mesure de vitesse ou de célérité où l'on trace un temps en fonction d'une distance, alors que la grandeur cherchée est une distance par unité de temps), il faut utiliser $1/\theta_1$ dans `formula_str`, pas $\theta_1$ directement.

Avant d'écrire `formula_str`, vérifier systématiquement par analyse dimensionnelle : écrire l'unité de $\theta_1$ telle que retournée par la régression (unité de `y_data` / unité de `x_data`), puis vérifier que la formule du mesurande, une fois les unités de $\theta_1$ substituées, redonne bien l'unité attendue du mesurande. Si ce n'est pas le cas, le signe de l'exposant sur $\theta_1$ dans `formula_str` est probablement inversé. Ce piège ne lève **aucune erreur** à l'exécution — le calcul aboutit avec un résultat numérique plausible en ordre de grandeur trompeur mais faux, donc il faut le contrôler manuellement, pas compter sur `gum_calc` pour le détecter.

Exemple concret : régression $\Delta T = \theta_0 + \theta_1 D$ (temps en fonction d'une distance), où $\theta_1$ est en ns/cm. Si le mesurande est une vitesse $v = 2D/\Delta T$, alors $v = 2/\theta_1$ (formule correcte, `"2 * 1e7 / theta1"` avec le facteur de conversion d'unités adéquat) — **jamais** $v = 2\theta_1$, qui donnerait un résultat dimensionnellement faux tout en s'exécutant sans erreur.

### 3.4ter-d Piège n°3 — reconstruire `res` pour la cellule COMPATIBILITÉ après un pipeline régression → mesurande

Cas fréquent : le mesurande final ($h$, une vitesse, une constante physique...) est obtenu par `full_pipeline_regression_to_measurand`, et l'étudiant fournit une valeur théorique de référence à confronter. Puisque le pipeline ne renvoie que le LaTeX (3.4ter-a-bis), la cellule COMPATIBILITÉ doit reconstruire `res` à la main, à l'identique du calcul interne du pipeline — **avec les mêmes degrés de liberté**, sous peine de désynchroniser $U$(mesurande) entre le bilan d'annexe et le test de compatibilité.

Procédure obligatoire, dans cet ordre :

```python
# Étape 1 — refaire la régression (mêmes données que le pipeline)
reg_h = linear_regression(x_data, y_data)

# Étape 2 — reconstruire l'incertitude sur theta1 (ou theta0) AVEC nu=reg_h["nu"]
# PIÈGE : uncertainty_type_B_relative(u_standard=...) seul fixe nu=inf par défaut.
# Sans le report explicite de nu, k passe de ~2.57 (Student, nu petit) à 2.00
# (gaussien, nu=inf), ce qui change U(mesurande) par rapport au bilan LaTeX
# généré en cellule MESURANDE — deux valeurs différentes pour la même grandeur
# dans le même notebook, silencieusement, sans aucune erreur ni warning.
u_theta1 = uncertainty_type_B_relative(u_standard=reg_h["u_theta1"])
u_theta1.nu = reg_h["nu"]          # OBLIGATOIRE — jamais laisser nu=inf par défaut ici

# Étape 3 — recalculer res avec full_gum_analysis, mêmes formula_str/variable_names
# que dans l'appel à full_pipeline_regression_to_measurand de la cellule MESURANDE
res_h = full_gum_analysis(
    formula_str        = "theta1 * e",              # identique à la cellule MESURANDE
    variable_names      = ["theta1", "e"],
    nominal_values      = {"theta1": reg_h["theta1"], "e": e_valeur},
    uncertainty_inputs  = {"theta1": u_theta1, "e": u_e},
)

# Étape 4 — compatibilité, à partir de res_h reconstruit
compat_h = generate_bilan_compatibilite(
    measurand_symbol = "h",
    y_mesure         = res_h["result"],
    uc               = res_h["uc"],
    nu_eff           = res_h["nu_eff"],
    y_theorique      = h_theorique,
    measurand_unit   = r"\joule\second",
)
```

**Vérification avant livraison** : `res_h["U_rounded"]` (cellule COMPATIBILITÉ) doit être **numériquement identique** à l'incertitude élargie affichée dans le bloc `\boxed{}` du bilan LaTeX généré par le pipeline en cellule MESURANDE. Un écart entre les deux est le signe que `nu` n'a pas été reporté à l'étape 2 — recalculer mentalement ou comparer les deux `k` affichés (`res_h["k"]` doit correspondre au `k` cité dans le bilan LaTeX, pas à `k=2.00` par défaut).

### 3.4quater Régression non linéaire

À utiliser quand le TP nécessite un ajustement non affine (exponentielle, sinusoïde, loi de puissance, gaussienne...) plutôt qu'une droite. `nonlinear_regression`/`generate_bilan_nonlinear_regression` sont des fonctions **sœurs** de `linear_regression`/`generate_bilan_regression`, pas des remplacements : ne jamais utiliser `linear_regression` sur un modèle non affine, et ne jamais utiliser `nonlinear_regression` pour une droite (`linear_regression` reste plus simple et déjà validée pour ce cas).

**Étape 1 — écrire `model_func` en numpy pur.**

```python
import numpy as np

def modele_decharge_RC(t, V0, tau):
    return V0 * np.exp(-t / tau)   # np.exp, JAMAIS math.exp
```

`nonlinear_regression` appelle `model_func` sur le tableau `x_data` entier (pas point par point) : une fonction écrite avec `math.exp`/`math.sin` lève un `TypeError` explicite à l'exécution — jamais un résultat silencieusement faux, mais autant l'éviter d'emblée.

**Étape 2 — choisir `p0` en lisant le nuage de points**, pas au hasard : la cause la plus fréquente de non-convergence de `curve_fit` est un `p0` trop éloigné de la solution. Pour une exponentielle décroissante, lire directement sur les données : `V0 ≈ y(x=0)`, `tau ≈` l'abscisse où `y` a chuté d'un facteur $e$.

**Étape 3 — appeler `nonlinear_regression` puis `generate_bilan_nonlinear_regression`** avec `_reg_precomputed` pour ne pas relancer `curve_fit` une seconde fois (même logique que `_res_precomputed`/`_reg_precomputed` ailleurs dans le module) :

```python
reg_RC = nonlinear_regression(
    x_data=t_data, y_data=V_data,
    model_func=modele_decharge_RC,
    p0=[5.0, 2.0],
    param_names=["V0", "tau"],
)

bilan_RC = generate_bilan_nonlinear_regression(
    x_symbol="t", y_symbol="V",
    x_unit=r"\milli\second", y_unit=r"\volt",
    x_data=t_data, y_data=V_data,
    model_func=modele_decharge_RC,
    p0=[5.0, 2.0], param_names=["V0", "tau"],
    param_symbols={"V0": "V_0", "tau": r"\tau"},
    param_units={"V0": r"\volt", "tau": r"\milli\second"},
    model_latex=r"{} \exp\left(-t/{}\right)",
    subsection_title="Décharge RC",
    _reg_precomputed=reg_RC,
)
```

**Piège — `model_latex` : un `{}` positionnel PAR paramètre, dans le MÊME ordre que `param_names`.** Pas de placeholder nommé, pas de LaTeX substitué par introspection de `model_func` (fragile, jugé indéboguable en pleine rédaction de CR — cf. feuille de route §2.2). `generate_bilan_nonlinear_regression` compte les `{}` de `model_latex` et lève une `ValueError` explicite si leur nombre ne correspond pas à `len(param_names)` — vérifier ce compte avant de générer la cellule plutôt que de compter sur l'erreur pour le signaler.

| param_names | model_latex correct | model_latex INCORRECT |
|---|---|---|
| `["V0", "tau"]` | `r"{} \exp\left(-t/{}\right)"` | `r"{} \exp\left(-t/\tau\right)"` (1 seul `{}`) |
| `["A", "omega", "phi"]` | `r"{} \sin\left({} t + {}\right)"` | `r"{} \sin(\omega t + \phi)"` (0 `{}`) |

**Convention de mise en forme du `\boxed{}` final** : comme pour `theta0`/`theta1` dans `generate_bilan_regression`, le résultat encadré utilise directement l'incertitude-type `u_params` (jamais une incertitude élargie $k \cdot u$) — le facteur $k$ de Welch-Satterthwaite ne s'applique qu'au mesurande final qui consommera ces paramètres via `full_gum_analysis`/`generate_bilan`, pas au paramètre de régression lui-même.

**Chaînage vers un mesurande dérivé** : si un mesurande ultérieur dépend d'un seul paramètre de la régression non linéaire (ex : une grandeur proportionnelle à `tau`), propager exactement comme pour `theta0`/`theta1` (cf. 3.5) :

```python
u_tau_chaine = uncertainty_type_B_relative(u_standard=reg_RC["u_params"]["tau"])
u_tau_chaine.nu = reg_RC["nu"]   # OBLIGATOIRE — sinon nu retombe à l'infini par défaut
```

Si le mesurande combine **deux** paramètres de la même régression non linéaire dans une seule formule (ex : `A * omega`), leur covariance `reg["cov"][i][j]` (indices dans l'ordre de `param_names`) doit être transmise via le paramètre `covariances` de `full_gum_analysis`/`generate_bilan`, à l'identique de `cov_theta01` pour la régression affine (voir `full_pipeline_regression_to_measurand`, qui n'a pas d'équivalent pour le cas non linéaire — construire cet appel à la main dans ce cas).

**Il n'existe pas de `full_pipeline_nonlinear_regression_to_measurand`** : contrairement au cas affine, aucune fonction ne chaîne automatiquement régression non linéaire → mesurande dérivé en un seul appel. Enchaîner manuellement `nonlinear_regression` → (`uncertainty_type_B_relative` avec report de `nu`, ou `covariances` si deux paramètres corrélés) → `full_gum_analysis`/`generate_bilan`, comme pour toute chaîne de mesurandes classique (3.5).

**Tracé (hors périmètre de ce skill, mentionné pour information)** : `plot_courbes.plot_nonlinear_regression(reg=reg_RC, ...)` est la fonction sœur de `plot_regression` pour ce cas — mais ce skill ne génère que des notebooks GUM (calcul + export LaTeX), jamais de cellule de tracé matplotlib ; ne pas en ajouter au notebook généré sauf demande explicite de l'étudiant.

### 3.5 Chaîne de mesurandes (cascade)

Quand un mesurande Y dépend d'un mesurande X déjà calculé :

```python
# Étape 1 — calcul de X
res_X = full_gum_analysis("formule_X", [...], nominales_X, incertitudes_X)

# Passage en chaîne — récupérer uc(X), JAMAIS U(X)
u_X_chaine = uncertainty_type_B_relative(u_standard=res_X["uc"])

# Étape 2 — calcul de Y
incertitudes_Y = {
    "X": u_X_chaine,
    "autre_var": uncertainty_type_B_from_resolution(0.001),
}
res_Y = full_gum_analysis("formule_Y", [...], nominales_Y, incertitudes_Y)
```

**Règle absolue** : passer `res_X["uc"]` (incertitude-type), jamais `res_X["U"]` (incertitude élargie). Doubler k serait une erreur GUM.

Note : les degrés de liberté de X ne se propagent pas dans Y (nu = inf côté Y). Pour un TP de L3, c'est acceptable.

### 3.5bis Piège n°4 — mesurandes intermédiaires partageant une variable source commune

Le patron 3.5 suppose implicitement que les mesurandes intermédiaires combinés en aval sont **statistiquement indépendants**. Ce n'est pas toujours vrai : si deux mesurandes intermédiaires $M_1$ et $M_2$ sont chacun fonction d'une **même variable d'entrée** $E$ (typique d'un protocole de double pesée par comparaison à un étalon commun, ou de deux mesures partageant un même zéro d'instrument), alors $M_1$ et $M_2$ sont corrélés même si chacun est calculé séparément par `full_gum_analysis`.

Exemple concret : $M_1 = E + Z_1$, $M_2 = E + Z_2$, avec $E$ l'étalon commun et $Z_1$, $Z_2$ des erreurs systématiques indépendantes. Alors $\text{Cov}(M_1, M_2) = \text{Var}(E) = u^2(E) \neq 0$. Traiter $M_1$ et $M_2$ comme deux mesurandes indépendants dans une cellule `M = M1 + M2` ultérieure (patron 3.5 appliqué tel quel) **omet silencieusement** le terme croisé $2\,c_{M_1}c_{M_2}\,u(M_1,M_2)$ de la variance composée de $M$. Le calcul s'exécute sans erreur et donne un résultat d'ordre de grandeur plausible — ce piège ne se détecte donc pas à l'exécution, seulement par relecture du modèle physique.

**Détection obligatoire avant toute cellule MESURANDE aval** : dès qu'un mesurande $Y$ combine plusieurs mesurandes intermédiaires déjà calculés séparément (patron 3.5), lister explicitement les variables d'entrée primitives de chacun et vérifier qu'aucune n'apparaît dans plus d'un intermédiaire. Si une variable est partagée, le patron 3.5 seul est insuffisant.

**Deux solutions, dans cet ordre de préférence :**

1. **Reformuler `formula_str` directement en variables primitives**, en substituant les mesurandes intermédiaires par leur expression complète, de sorte que la variable partagée n'apparaisse qu'une seule fois dans la formule. `full_gum_analysis` dérive alors symboliquement la formule entière et capture nativement la corrélation, sans covariance à déclarer à la main.
   ```python
   # Au lieu de M = M1 + M2 avec M1, M2 mesurandes séparés :
   # rho = 4M / (pi d^2 L) avec M = M1 + M2 = 2E + Z1 + Z2
   res_rho = full_gum_analysis(
       formula_str        = "4 * (2*E + Z1 + Z2) / (pi * d**2 * L)",
       variable_names      = ["E", "Z1", "Z2", "d", "L"],
       nominal_values      = {...},
       uncertainty_inputs  = {...},
   )
   ```
   C'est la solution à privilégier chaque fois que la formule reste lisible une fois développée : elle élimine le risque d'oubli et ne nécessite aucun calcul de covariance manuel.

2. **Si la reformulation directe est impraticable** (formule trop lourde, ou mesurandes intermédiaires imposés par la structure du compte rendu), garder les mesurandes séparés et déclarer la covariance explicitement via le paramètre `covariances` de `full_gum_analysis`/`generate_bilan` :
   ```python
   res_M = full_gum_analysis(
       formula_str        = "M1 + M2",
       variable_names      = ["M1", "M2"],
       nominal_values      = {"M1": res_M1["result"], "M2": res_M2["result"]},
       uncertainty_inputs  = {"M1": u_M1_chaine, "M2": u_M2_chaine},
       covariances         = {("M1", "M2"): u_E_value**2},  # Cov(M1,M2) = Var(E)
   )
   ```
   Cette voie exige de recalculer à la main la covariance entre les deux mesurandes à partir de leur(s) variable(s) source commune(s) — source d'erreur supplémentaire, à réserver aux cas où l'option 1 n'est pas applicable.

**Avant de générer toute cellule combinant des mesurandes intermédiaires (patron 3.5)** : recenser les variables primitives de chaque intermédiaire impliqué et confirmer explicitement leur indépendance mutuelle. Ne jamais supposer l'indépendance par défaut dès qu'un protocole de comparaison à un étalon, une référence, ou un zéro commun est mentionné dans l'énoncé.

### 3.6 Unités siunitx

Utiliser la syntaxe siunitx dans tous les champs `*_unit` :
- `r"\ohm"`, `r"\volt"`, `r"\ampere"`, `r"\second"`, `r"\meter"`
- `r"\kilo\gram"`, `r"\meter\per\second"`, `r"\newton\meter"`
- Chaînes vides `""` si l'unité est adimensionnelle

### 3.7 Formules SymPy

Les `formula_str` doivent être des expressions SymPy valides :
- Multiplication : `*` (jamais implicite)
- Puissance : `**`
- Fonctions : `sin(x)`, `cos(x)`, `sqrt(x)`, `exp(x)`, `log(x)`
- Pi : `pi` (reconnu par SymPy)
- Les noms de variables dans `formula_str` doivent correspondre **exactement** aux clés de `variable_names`

---

## 4. TEMPLATE DU NOTEBOOK GÉNÉRÉ

Le notebook suit ce squelette JSON. Adapter le nombre de cellules MESURANDE selon les grandeurs du TP.

### Cellule 1 — HEADER
```python
# ============================================================
# GUM — Incertitudes de Mesures — <SUJET_TP>
# ============================================================

import sys
sys.path.insert(0, r"C:\Users\<USER>\<CHEMIN_VERS_GUM_CALC>")  # À adapter — dossier contenant
                                                                # gum_calc.py ET gum_export.py

# --- gum_calc.py : moteur de calcul pur, aucune fonction ne renvoie de LaTeX ---
from gum_calc import (
    uncertainty_type_A,
    uncertainty_type_B_from_resolution,
    uncertainty_type_B_uniform,
    uncertainty_type_B_relative,
    uncertainty_type_exact,
    UncertaintyInput,
    full_gum_analysis,
    linear_regression,
    nonlinear_regression,                 # uniquement si le TP a un ajustement non affine
    compatibility_test,                   # uniquement si une confrontation théorie/mesure est demandée
)

# --- gum_export.py : rédaction LaTeX, importe gum_calc en interne ---
from gum_export import (
    generate_bilan,
    generate_bilan_regression,
    generate_bilan_nonlinear_regression,  # uniquement si le TP a un ajustement non affine
    generate_bilan_compatibilite,         # uniquement si une confrontation théorie/mesure est demandée
    generate_annexe,
    full_pipeline_regression_to_measurand,
)

print("gum_calc et gum_export chargés avec succès.")
```

### Cellule N — MESURANDE (répéter pour chaque grandeur)
```python
# ============================================================
# MESURANDE — <NOM_GRANDEUR> (<SYMBOLE>)
# ============================================================

# --- Sources d'incertitude ---
u_<var1> = uncertainty_type_<TYPE>(<ARGS>)
u_<var2> = uncertainty_type_<TYPE>(<ARGS>)

# --- Valeurs nominales ---
nominales_<G> = {
    "<var1>": <valeur>,
    "<var2>": <valeur>,
}

# --- Incertitudes ---
incertitudes_<G> = {
    "<var1>": u_<var1>,
    "<var2>": u_<var2>,
}

# --- Analyse GUM ---
res_<G> = full_gum_analysis(
    formula_str        = "<formule_sympy>",
    variable_names     = ["<var1>", "<var2>"],
    nominal_values     = nominales_<G>,
    uncertainty_inputs = incertitudes_<G>,
)

# --- Bilan LaTeX ---
bilan_<G> = generate_bilan(
    measurand_name     = "<Nom en toutes lettres>",
    measurand_symbol   = "<Symbole_LaTeX>",
    formula_str        = "<formule_sympy>",
    variable_names     = ["<var1>", "<var2>"],
    variable_symbols   = {"<var1>": "<Sym1>", "<var2>": "<Sym2>"},
    variable_units     = {"<var1>": r"<unité1>", "<var2>": r"<unité2>"},
    nominal_values     = nominales_<G>,
    uncertainty_inputs = incertitudes_<G>,
    measurand_unit     = r"<unité_mesurande>",
    _res_precomputed   = res_<G>,   # évite de relancer full_gum_analysis une 2e fois
)

# --- Affichage console ---
print(f"<G> = {res_<G>['result_rounded']} ± {res_<G>['U_rounded']}")
print(f"uc = {res_<G>['uc']:.4g}  |  k = {res_<G>['k']:.3f}  |  ν_eff = {res_<G>['nu_eff']:.1f}")
print("Budget :", {k: f"{v:.1f}%" for k, v in res_<G>['budget'].items()})
```

### Cellule COMPATIBILITÉ (répéter pour chaque grandeur confrontée à une théorie)
```python
# ============================================================
# COMPATIBILITÉ THÉORIE / MESURE — <NOM_GRANDEUR> (<SYMBOLE>)
# ============================================================

<G>_theorique = <valeur>   # valeur théorique de référence

compat_<G> = generate_bilan_compatibilite(
    measurand_symbol = "<Symbole_LaTeX>",
    measurand_name   = "<Nom en toutes lettres>",
    y_mesure         = res_<G>["result"],       # ou reg["theta0"/"theta1"] si régression
    uc               = res_<G>["uc"],           # ou reg["u_theta0"/"u_theta1"]
    nu_eff           = res_<G>["nu_eff"],       # ou reg["nu"]
    y_theorique      = <G>_theorique,
    measurand_unit   = r"<unité_mesurande>",
)

print(compat_<G>)
# NE PAS ajouter compat_<G> à la liste passée à generate_annexe : ce
# bloc reste dans sa propre cellule, jamais dans le bloc annexe final.
```

### Cellule RÉGRESSION NON LINÉAIRE (si le TP contient un ajustement non affine — voir 3.4quater)
```python
# ============================================================
# RÉGRESSION NON LINÉAIRE — <NOM_GRANDEUR> (<SYMBOLE>)
# ============================================================

import numpy as np

def modele_<G>(<x_symbole_python>, <param1>, <param2>):
    return <expression numpy uniquement, ex: <param1> * np.exp(-<x_symbole_python> / <param2>)>

# --- p0 lu sur le nuage de points, jamais au hasard (cf. 3.4quater) ---
p0_<G> = [<estimation_param1>, <estimation_param2>]

reg_<G> = nonlinear_regression(
    x_data       = <x_data>,
    y_data       = <y_data>,
    model_func   = modele_<G>,
    p0           = p0_<G>,
    param_names  = ["<param1>", "<param2>"],
)

# --- Bilan LaTeX : _reg_precomputed réutilise l'ajustement ci-dessus ---
bilan_<G> = generate_bilan_nonlinear_regression(
    x_symbol         = "<x_symbole_LaTeX>",
    y_symbol         = "<y_symbole_LaTeX>",
    x_unit           = r"<unité_x>",
    y_unit           = r"<unité_y>",
    x_data           = <x_data>,
    y_data           = <y_data>,
    model_func       = modele_<G>,
    p0               = p0_<G>,
    param_names      = ["<param1>", "<param2>"],
    param_symbols    = {"<param1>": "<Sym1_LaTeX>", "<param2>": "<Sym2_LaTeX>"},
    param_units      = {"<param1>": r"<unité1>", "<param2>": r"<unité2>"},
    # UN '{}' PAR paramètre, MÊME ORDRE que param_names — voir 3.4quater
    model_latex      = r"<gabarit avec un {} par paramètre>",
    subsection_title = "<Titre>",
    _reg_precomputed = reg_<G>,
)

# --- Affichage console ---
print(f"<param1> = {reg_<G>['params']['<param1>']:.4g} ± {reg_<G>['u_params']['<param1>']:.2g}")
print(f"<param2> = {reg_<G>['params']['<param2>']:.4g} ± {reg_<G>['u_params']['<param2>']:.2g}")
print(f"nu = {reg_<G>['nu']}  |  r² = {reg_<G>['r2']:.4f}")
```

### Cellule finale — LATEX
```python
# ============================================================
# GÉNÉRATION DU LATEX
# ============================================================

print(generate_annexe([
    bilan_<G1>,
    bilan_<G2>,
    # ... tous les bilans dans l'ordre du CR
]))
```

---

## 5. PROCÉDURE DE GÉNÉRATION

À chaque demande de mode incertitudes, procéder dans cet ordre :

1. **Recenser** toutes les grandeurs à calculer et identifier les dépendances (chaînes éventuelles). **Dès qu'un mesurande combine plusieurs mesurandes intermédiaires (patron 3.5)** : lister les variables primitives de chacun et vérifier qu'aucune n'est partagée (piège 3.5bis — typique d'un protocole de comparaison à un étalon commun). Si une variable est partagée, reformuler `formula_str` en variables primitives (solution 1 de 3.5bis) plutôt que d'empiler des mesurandes intermédiaires indépendants.
2. **Identifier** pour chaque variable : type de source (A/B résolution/B uniforme/B relative/exact) et valeur numérique
3. **Identifier** les formules SymPy exactes
4. **Identifier** les unités siunitx de chaque variable et du mesurande
5. **Identifier** les grandeurs à confronter à une valeur théorique (si l'étudiant en fournit une) et générer la cellule COMPATIBILITÉ correspondante, juste après la cellule MESURANDE ou de régression concernée
6. **Si `full_pipeline_regression_to_measurand` est utilisé** : vérifier `"theta1" in variable_names` → `couple_theta1=True`, `"theta0" in variable_names` → `couple_theta0=True` (piège 3.4ter-a), et vérifier par analyse dimensionnelle que l'exposant de $\theta_1$/$\theta_0$ dans `formula_str` correspond au sens réel de la régression $y=\theta_0+\theta_1 x$ (piège 3.4ter-b)
6bis. **Si une cellule COMPATIBILITÉ suit un `full_pipeline_regression_to_measurand`** : ne jamais attendre `res`/`reg` en retour de cette fonction (elle ne renvoie que le LaTeX, 3.4ter-a-bis) — reconstruire `res` via `linear_regression` puis `full_gum_analysis` en reportant explicitement `nu=reg["nu"]` sur l'incertitude de $\theta_1$/$\theta_0$ (piège 3.4ter-d), et vérifier avant livraison que `U_rounded` de cette cellule coïncide avec l'incertitude élargie du bilan LaTeX de la cellule MESURANDE
6ter. **Si le TP contient un ajustement non affine** (exponentielle, sinusoïde, loi de puissance...) : ne jamais utiliser `linear_regression`/`generate_bilan_regression` pour ce cas, employer `nonlinear_regression`/`generate_bilan_nonlinear_regression` (3.4quater) — vérifier que `model_func` n'utilise que des opérations numpy, que `p0` est lu sur le nuage de points plutôt que deviné, et que `model_latex` porte exactement un `{}` par paramètre de `param_names`, dans le même ordre
7. **Générer** le `.ipynb` complet en JSON valide, avec la cellule HEADER portant deux blocs d'import séparés (`from gum_calc import (...)` puis `from gum_export import (...)`, voir §2 et §4) — jamais un seul bloc regroupant les deux modules
8. **Avant livraison, si possible, exécuter mentalement ou réellement le notebook** pour confirmer l'absence d'erreur ET la plausibilité physique du résultat (ordre de grandeur cohérent avec le contexte du TP) — une exécution sans erreur ne garantit pas un résultat physiquement correct (piège 3.4ter-b)
9. **Livrer** le fichier via `present_files`

Si des informations manquent (valeurs numériques, résolutions des instruments), demander **une seule question** avant de générer, groupant tous les manquants.

---

## 6. CONTRAINTES ABSOLUES

- Ne jamais recalculer ce que `gum_calc` fait : aucune propagation manuelle dans le notebook. Ne jamais reformuler en prose ce que `gum_export` rédige : aucun bilan LaTeX écrit à la main dans une cellule
- Ne jamais importer une fonction de rédaction LaTeX (`generate_bilan*`, `generate_annexe`, `full_pipeline_regression_to_measurand`) depuis `gum_calc` : ces fonctions vivent dans `gum_export.py`, qui doit être importé séparément — voir Cellule HEADER, §4
- Ne jamais passer `res["U"]` comme source dans une chaîne — toujours `res["uc"]`
- Ne jamais empiler des mesurandes intermédiaires (patron 3.5) sans avoir vérifié qu'ils ne partagent aucune variable source commune — voir piège 3.5bis. Un étalon, une référence, ou un zéro d'instrument commun à deux grandeurs dérivées crée une covariance non nulle entre elles, silencieusement omise si chacune est traitée comme mesurande indépendant
- Dans tout appel à `full_pipeline_regression_to_measurand`, `couple_theta1=True` et/ou `couple_theta0=True` sont obligatoires dès que `formula_str` référence `theta1`/`theta0` — voir 3.4ter-a
- Avant d'écrire `formula_str` pour un mesurande dérivé d'une régression, vérifier par analyse dimensionnelle le sens de $\theta_1$ (unité de $y$/unité de $x$, jamais l'inverse) — voir 3.4ter-b
- `full_pipeline_regression_to_measurand` ne retourne **que** le LaTeX (`-> str`), jamais un tuple `(res, reg, bilan)` — voir 3.4ter-a-bis
- Dans une cellule COMPATIBILITÉ reconstruisant `res` après un pipeline régression → mesurande, reporter systématiquement `nu=reg["nu"]` sur l'incertitude de la pente/ordonnée avant de rappeler `full_gum_analysis` ; sans ce report, `nu` retombe à l'infini par défaut et `U`(mesurande) diverge silencieusement de la valeur du bilan LaTeX — voir 3.4ter-d
- Le chemin `sys.path.insert` est un placeholder commenté — l'étudiant l'adapte
- Les `variable_names` dans `formula_str`, `full_gum_analysis` et `generate_bilan` doivent être **identiques**
- Un seul `generate_annexe` en fin de notebook, regroupant tous les bilans
- `generate_bilan_compatibilite` ne rentre jamais dans `generate_annexe` : c'est `y_theorique` qui distingue son usage de `generate_bilan`, et son résultat s'affiche dans sa propre cellule, au plus près de la grandeur ou de la courbe concernée
- Ne jamais utiliser `linear_regression` sur un modèle non affine, ni `nonlinear_regression` sur une droite — voir 3.4quater
- Dans `model_func` passé à `nonlinear_regression`/`generate_bilan_nonlinear_regression`, n'utiliser que des opérations numpy (`np.exp`, `np.sin`, `np.sqrt`...), jamais le module `math` — voir 3.4quater
- `model_latex` doit porter exactement un `{}` positionnel par paramètre de `param_names`, dans le même ordre — `generate_bilan_nonlinear_regression` lève une `ValueError` explicite sinon, mais vérifier ce compte avant de générer la cellule
- Il n'existe pas de `full_pipeline_nonlinear_regression_to_measurand` : le chaînage vers un mesurande dérivé d'une régression non linéaire s'enchaîne manuellement (`uncertainty_type_B_relative` + report de `nu`, ou `covariances` si deux paramètres corrélés) — voir 3.4quater
- Le notebook livré doit être du JSON `.ipynb` valide, exécutable sans modification autre que le `sys.path`
