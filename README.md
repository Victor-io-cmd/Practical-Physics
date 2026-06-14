# gum-calc

Moteur de calcul d'incertitudes GUM pour comptes rendus de TP de physique expérimentale, avec export LaTeX.

---

## Auteur

**Victorio BONNEVILLE DIAZ** — Étudiant L3 Physique Générale, UPEC (Université Paris-Est Créteil)

Code généré avec [Claude (Anthropic)](https://www.anthropic.com) à partir d'une architecture et d'une logique définies par l'auteur, sur la base des cours de métrologie GUM de L2 Physique de l'Université de Paris-Est Créteil.

---

## Contexte

Ce projet a pour but de simplifier et d'automatiser les calculs d'incertitudes pour des comptes rendus de travaux pratiques, à travers un moteur de calcul et un export LaTeX.

Le moteur est conforme à la norme **GUM** (*Guide to the Expression of Uncertainty in Measurement*). Il couvre les incertitudes de types A et B, la propagation par dérivation via SymPy, le facteur d'élargissement par la formule de Welch-Satterthwaite, et la régression linéaire par la méthode des moindres carrés.

---

## Structure du projet

```
gum-calc/
├── GUM_-_Incertitudes_de_Mesures.ipynb   # Notebook template
├── gum_calc.py                           # Moteur de calcul et export LaTeX
└── README.md
```

---

## Installation et configuration

Bibliothèques utilisées :

- `sympy` — dérivation symbolique des coefficients de sensibilité
- `scipy` — table de Student pour le facteur d'élargissement (Welch-Satterthwaite)
- `math` — calculs numériques de base (bibliothèque standard Python, pas d'installation requise)
- `jupyter` — environnement d'exécution du notebook template

Installation des dépendances :

```bash
pip install sympy scipy jupyter
```

**Configuration du chemin dans le notebook :**

En tête du notebook, la ligne suivante pointe vers le dossier contenant `gum_calc.py`. À adapter selon votre machine :

```python
import sys
sys.path.insert(0, r"C:\chemin\vers\gum-calc")
```

**Versions testées :** Python 3.11.9 · SymPy 1.14.0 · SciPy 1.17.1

---

## Fonctionnement global

Le workflow suit la logique suivante :

1. On déclare les incertitudes de chaque grandeur d'entrée
2. On calcule le mesurande avec `full_gum_analysis`
3. On génère le bilan LaTeX avec `generate_bilan`
4. En fin de notebook, `generate_annexe` assemble tout

Les sections suivantes expliquent chaque étape dans l'ordre du code.

---

## Étape 1 — Déclarer les incertitudes

Avant de calculer quoi que ce soit, on caractérise chaque grandeur d'entrée par son incertitude-type. `gum_calc.py` propose cinq fonctions selon la situation.

---

### Incertitude de type A

L'incertitude de type A s'applique quand on dispose d'une série de mesures répétées de la même grandeur. On exploite la dispersion statistique : l'incertitude-type est l'écart-type de la moyenne, soit

```
u_A = s / √N
```

où `s` est l'écart-type empirique de la série et `N` le nombre de mesures.

```python
u_x = uncertainty_type_A(values)
```

| Paramètre | Type | Description |
|---|---|---|
| `values` | `list[float]` | Liste des N valeurs mesurées (N ≥ 2) |

La valeur utile pour la suite est `u_x["u"]`. La clé `nu` (= N−1) est retenue pour Welch-Satterthwaite si N < 30.

---

### Incertitude de type B — résolution d'un instrument

L'incertitude de type B s'applique quand on ne répète pas la mesure. On modélise l'ignorance sur la valeur vraie par une distribution uniforme de largeur δ (la résolution de l'instrument). L'incertitude-type associée est

```
u_B = (δ/2) / √3
```

```python
u_x = uncertainty_type_B_from_resolution(resolution)
```

| Paramètre | Type | Description |
|---|---|---|
| `resolution` | `float` | Résolution δ de l'instrument (dernier digit ou graduation) |

---

### Incertitude de type B — demi-largeur connue directement

Même principe que ci-dessus, mais quand la notice constructeur donne directement une tolérance ±a plutôt qu'une résolution δ.

```python
u_x = uncertainty_type_B_uniform(half_width)
```

| Paramètre | Type | Description |
|---|---|---|
| `half_width` | `float` | Demi-largeur a de l'intervalle (même unité que la grandeur) |

> Note : `uncertainty_type_B_from_resolution(δ)` est un raccourci qui appelle `uncertainty_type_B_uniform(δ/2)` en interne.

---

### Incertitude de type B — valeur fournie directement

Quand l'incertitude-type est connue directement — par calibration, notice constructeur, ou propagée depuis un mesurande intermédiaire déjà calculé.

```python
u_x = uncertainty_type_B_relative(u_standard=valeur)
```

| Paramètre | Type | Description |
|---|---|---|
| `u_standard` | `float` | Incertitude-type u (pas l'incertitude élargie U = k·u) |
| `relative_knowledge` | `float` ou `None` | Connaissance relative sur u — si fourni, estime les degrés de liberté par ν = 1/(2r²) |

> En chaîne de mesurandes, toujours passer `res["uc"]`, jamais `res["U"]`. Passer U = k·u_c doublerait le facteur k — c'est une erreur GUM.

---

### Constante exacte

Pour une constante physique fondamentale (c, e, h…) ou toute grandeur dont l'incertitude est négligeable devant les autres sources.

```python
u_x = uncertainty_type_exact()
```

L'incertitude-type est nulle. La variable est exclue du budget d'incertitudes.

---

## Étape 2 — Calculer le mesurande

`full_gum_analysis` est la fonction principale. Elle prend la formule du mesurande, les valeurs nominales et les incertitudes déclarées à l'étape 1, et retourne le résultat complet : propagation GUM par dérivation symbolique, Welch-Satterthwaite si nécessaire, et arrondi cohérent.

```python
res = full_gum_analysis(
    formula_str        = "...",
    variable_names     = [...],
    nominal_values     = {...},
    uncertainty_inputs = {...},
)
```

| Paramètre | Description |
|---|---|
| `formula_str` | Expression SymPy du mesurande — `*` pour la multiplication, `**` pour la puissance, `sqrt()`, `log()`, `sin()`, etc. |
| `variable_names` | Liste ordonnée des noms de variables — doit être identique aux clés de `nominal_values` et `uncertainty_inputs` |
| `nominal_values` | Dict `{nom: valeur nominale}` |
| `uncertainty_inputs` | Dict `{nom: dict uncertainty_type_*}` |
| `k_override` | Facteur k forcé (optionnel) — si absent, Welch-Satterthwaite détermine k automatiquement |

Les clés utiles du résultat :

| Clé | Description |
|---|---|
| `res["result_rounded"]` | Valeur nominale arrondie, cohérente avec U |
| `res["U_rounded"]` | Incertitude élargie U = k·u_c, arrondie à 2 chiffres significatifs |
| `res["uc"]` | Incertitude-type composée u_c (à passer en cas de chaîne — voir plus bas) |
| `res["k"]` | Facteur d'élargissement retenu |
| `res["nu_eff"]` | Degrés de liberté effectifs |
| `res["budget"]` | Dict `{nom: %}` — contribution de chaque source à la variance composée |

---

## Étape 3 — Générer le bilan LaTeX

`generate_bilan` prend les mêmes entrées que `full_gum_analysis` et produit un bloc LaTeX complet : modèle de mesure, valeurs nominales, incertitudes source par source, coefficients de sensibilité, Welch-Satterthwaite si nécessaire, budget, et résultat encadré.

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

| Paramètre | Description |
|---|---|
| `measurand_name` | Nom en toutes lettres — utilisé dans la phrase d'introduction du bilan |
| `measurand_symbol` | Symbole LaTeX du mesurande |
| `formula_str` | Même expression que dans `full_gum_analysis` |
| `variable_names` | Même liste que dans `full_gum_analysis` |
| `variable_symbols` | Dict `{nom_python: symbole_LaTeX}` |
| `variable_units` | Dict `{nom_python: unité_siunitx}` |
| `nominal_values` | Même dict que dans `full_gum_analysis` |
| `uncertainty_inputs` | Même dict que dans `full_gum_analysis` |
| `measurand_unit` | Unité siunitx du mesurande |
| `k_override` | Facteur k forcé (optionnel) |
| `subsection` | `True` par défaut — génère un `\subsection{}` en tête du bilan |
| `global_sig_figs` | Chiffres significatifs pour les valeurs intermédiaires (défaut : 3) |

Les unités suivent la syntaxe du package siunitx : `r"\ohm"`, `r"\volt"`, `r"\ampere"`, `r"\second"`, `r"\meter"`, `r"\kilo\gram"`, `r"\meter\per\second"`, etc. Chaîne vide `""` si adimensionnel.

---

## Étape 4 — Assembler l'annexe LaTeX

`generate_annexe` prend la liste de tous les bilans dans l'ordre du compte rendu et produit la section LaTeX complète.

```python
print(generate_annexe([bilan_1, bilan_2, ...]))
```

---

## Cas particuliers

### Régression linéaire seule

Quand on veut caractériser une droite expérimentale sans en dériver un autre mesurande. Le modèle est y = θ₀ + θ₁·x. Le bilan inclut les estimateurs, leurs incertitudes-types dérivées de la variance résiduelle, et le coefficient r².

```python
bilan_reg = generate_bilan_regression(
    x_symbol         = "...",
    y_symbol         = "...",
    x_unit           = r"...",
    y_unit           = r"...",
    x_data           = [...],
    y_data           = [...],
    subsection_title = "...",
    slope_unit       = r"...",
    intercept_unit   = r"...",
    slope_symbol     = r"\theta_1",
    intercept_symbol = r"\theta_0",
)
```

---

### Régression + mesurande dérivé en un appel

Quand le mesurande final dépend de la pente ou de l'ordonnée à l'origine d'une régression. Les noms `"theta1"` (pente) et `"theta0"` (ordonnée) dans `variable_names` sont reconnus automatiquement et alimentés par la régression — pas besoin de les fournir dans `nominal_values_helpers` ni `uncertainty_inputs_helpers`.

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
    measurand_unit             = r"...",
    slope_symbol               = r"\theta_1",
    intercept_symbol           = r"\theta_0",
)
print(latex)
```

La sortie contient le bilan de régression suivi du bilan GUM du mesurande, séparés par un `\clearpage`.

---

### Chaîne de mesurandes

Quand un mesurande Y dépend d'un mesurande X calculé précédemment, on passe `res_X["uc"]` via `uncertainty_type_B_relative`.

```python
res_X = full_gum_analysis(...)

incertitudes_Y = {
    "X":     uncertainty_type_B_relative(u_standard=res_X["uc"]),  # jamais res_X["U"]
    "autre": uncertainty_type_B_from_resolution(...),
}
res_Y = full_gum_analysis(...)
```

> Ne jamais passer `res_X["U"]` — ce serait appliquer k deux fois.

---
