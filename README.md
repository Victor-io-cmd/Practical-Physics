# gum-calc

Un moteur de calcul qui transforme des mesures brutes de TP en une analyse d'incertitude complète et formatée, prête à être collée dans un compte rendu LaTeX.

---

## Le principe

En physique expérimentale, chaque mesure s'accompagne d'une incertitude, et chaque compte rendu de TP nécessite une section rigoureuse expliquant d'où vient cette incertitude et comment elle se propage jusqu'au résultat final. Cette démarche suit une norme internationale, le **GUM** (Guide pour l'expression de l'incertitude de mesure), et la réaliser à la main pour chaque grandeur d'un compte rendu est lent, répétitif et source d'erreurs.

`gum-calc` automatise l'ensemble de cette chaîne de traitement. On lui fournit une formule, des valeurs mesurées, et la façon dont chaque valeur a été obtenue, et il renvoie le résultat propagé accompagné d'une rédaction LaTeX prête à l'emploi : le modèle de mesure, les coefficients de sensibilité, le bilan d'incertitude, et le résultat final correctement arrondi.

Développé pour mes propres comptes rendus de TP en tant qu'étudiant en L3 Physique à l'UPEC (Université Paris-Est Créteil). La sortie LaTeX générée est en français, puisqu'elle est directement destinée à des comptes rendus rédigés en français.

---

## Installation

```bash
pip install sympy scipy numpy
```

Pour le tracé de courbes (`plot_courbes.py`), ajouter :

```bash
pip install matplotlib
```

Aucune autre dépendance. Fonctionne avec Python 3.10+ (utilisation de `list[float]` en annotation de type).

---

## Structure du projet

```
gum-calc/
├── gum_calc.py               # Moteur de calcul et export LaTeX
├── plot_courbes.py           # Tracé et export PDF des courbes (basé sur gum_calc, ne recalcule rien)
├── gum_notebook.skill        # Skill Claude : génère automatiquement le notebook ci-dessous
├── gum_uncertainties.ipynb   # Notebook modèle, une cellule par mesurande
└── README.md
```

`gum_notebook.skill` est un [Claude Skill](https://www.anthropic.com) : un ensemble d'instructions qui permet à Claude de générer automatiquement `gum_uncertainties.ipynb` pour un nouveau TP, câblé sur `gum_calc.py`, plutôt que de dupliquer et remplir le modèle à la main pour chaque compte rendu.

---

## Usage rapide

```python
from gum_calc import (
    uncertainty_type_A,
    uncertainty_type_B_from_resolution,
    full_gum_analysis,
    generate_bilan,
)

# Tension : 5 mesures répétées → incertitude de type A
u_U = uncertainty_type_A([5.02, 5.01, 5.03, 5.02, 5.00])

# Courant : lu sur un ampèremètre de résolution 1 mA → type B
u_I = uncertainty_type_B_from_resolution(resolution=0.001)

res = full_gum_analysis(
    formula_str="U / I",
    variable_names=["U", "I"],
    nominal_values={"U": u_U.mean, "I": 0.502},
    uncertainty_inputs={"U": u_U, "I": u_I},
)

print(f"R = {res['result_rounded']} ± {res['U_rounded']}")
# R = 10.0 ± 0.1

# Génère directement le bloc LaTeX correspondant, prêt pour Overleaf
bilan = generate_bilan(
    measurand_name="Résistance", measurand_symbol="R",
    formula_str="U / I", variable_names=["U", "I"],
    variable_symbols={"U": "U", "I": "I"},
    variable_units={"U": r"\volt", "I": r"\ampere"},
    nominal_values={"U": u_U.mean, "I": 0.502},
    uncertainty_inputs={"U": u_U, "I": u_I},
    measurand_unit=r"\ohm",
    _res_precomputed=res,
)
```

Pour un TP complet avec plusieurs mesurandes, utiliser directement `gum_uncertainties.ipynb` : une cellule par grandeur, le tout assemblé en annexe par `generate_annexe()` en fin de notebook.

### Comparaison à une valeur théorique

Question différente de `generate_bilan` (« quelle est l'incertitude ? ») : « ce résultat est-il compatible avec une valeur théorique de référence ? ». Comparaison par variable de Student réduite, avec le risque associé lu dans la loi de Student (ou normale si $\nu_{\text{eff}} \to \infty$) :

```python
from gum_calc import generate_bilan_compatibilite

compat = generate_bilan_compatibilite(
    measurand_symbol="R", measurand_name="Résistance",
    y_mesure=res["result"], uc=res["uc"], nu_eff=res["nu_eff"],
    y_theorique=10.0, measurand_unit=r"\ohm",
)
print(compat)
```

Ce bloc reste dans sa propre cellule/section du CR, il n'entre jamais dans `generate_annexe()`. `compatibility_test()` (sans le `generate_bilan_`) renvoie le même calcul sous forme de dict plutôt que de LaTeX, si seule la valeur numérique du risque est utile.

### Régression non affine

Pour un ajustement non linéaire (exponentielle, sinusoïde, loi de puissance...), `nonlinear_regression` est l'analogue non affine de la régression linéaire, basé sur `scipy.optimize.curve_fit` :

```python
import numpy as np
from gum_calc import nonlinear_regression, generate_bilan_nonlinear_regression

def decharge_RC(t, V0, tau):
    return V0 * np.exp(-t / tau)

reg = nonlinear_regression(
    x_data=t_data, y_data=V_data,
    model_func=decharge_RC, p0=[5.0, 2.0], param_names=["V0", "tau"],
)

bilan = generate_bilan_nonlinear_regression(
    x_symbol="t", y_symbol="V", x_unit=r"\milli\second", y_unit=r"\volt",
    x_data=t_data, y_data=V_data, model_func=decharge_RC,
    p0=[5.0, 2.0], param_names=["V0", "tau"],
    param_symbols={"V0": "V_0", "tau": r"\tau"},
    param_units={"V0": r"\volt", "tau": r"\milli\second"},
    model_latex=r"{} \exp\left(-t/{}\right)",   # un '{}' par paramètre, même ordre
    _reg_precomputed=reg,
)
```

### Tracé de courbes

`plot_courbes.py` trace et exporte en PDF les résultats de `gum_calc.py`, sans rien recalculer : nuage de points avec barres d'erreur, droite ou courbe ajustée, bande d'incertitude autour de l'ajustement, résidus, séries multiples.

```python
from plot_courbes import plot_regression, plot_nonlinear_regression, export_figure

fig, ax = plot_nonlinear_regression(
    t_data, V_data, reg=reg,
    x_label="Temps", y_label="Tension", show_band=True,
)
export_figure(fig, "decharge_RC")   # -> figures/decharge_RC.pdf
```

`plot_regression` (cas affine) et `plot_nonlinear_regression` (cas non affine, bande d'incertitude propagée par différences finies) sont deux fonctions sœurs indépendantes, chacune dédiée à son cas.

---

## Fonctionnalités

**Typage des incertitudes.** Cinq fonctions couvrent les cas standards du GUM : mesures répétées (type A), résolution d'instrument ou tolérance connue (type B), une incertitude-type directement connue (par exemple propagée depuis un calcul précédent), et les constantes exactes.

**Propagation symbolique.** La fonction centrale prend une formule en texte brut (comme `"U / I"`), la parse avec SymPy, et la différentie automatiquement par rapport à chaque variable. Plus besoin de dériver les formules de propagation à la main pour chaque nouveau mesurande.

**Facteur d'élargissement de Welch-Satterthwaite.** Lorsque des sources d'incertitude de fiabilités différentes sont combinées (une source de type B bien connue à côté d'une source de type A estimée sur peu de mesures), le moteur calcule le nombre de degrés de liberté effectif et lit le facteur d'élargissement `k` correct dans la table de Student, plutôt que de systématiquement supposer `k = 2`.

**Régression linéaire avec covariance.** Un ajustement par moindres carrés ordinaires qui renvoie aussi la covariance entre la pente et l'ordonnée à l'origine, une grandeur presque toujours oubliée dans les calculs à la main mais qui compte dès qu'un résultat final dépend des deux.

**Régression non affine.** Pour les modèles non linéaires (exponentielle, sinusoïde, loi de puissance...), `nonlinear_regression` s'appuie sur `scipy.optimize.curve_fit` et renvoie la même famille d'informations que la régression linéaire (paramètres, incertitudes-types, matrice de covariance complète, degrés de liberté `N - p`), pour rester exploitable exactement de la même façon en aval.

**Comparaison à une valeur théorique.** Au-delà du simple bilan d'incertitude, une comparaison mesure/théorie par variable de Student réduite : écart relatif, statistique de test, risque associé lu dans la loi de Student (ou normale si les degrés de liberté effectifs sont infinis), et conclusion de compatibilité — la question typique de fin de TP, formalisée plutôt qu'estimée à l'œil sur le bilan d'incertitude.

**Arrondi correct, à chaque fois.** C'est toujours l'incertitude qui détermine le nombre de chiffres significatifs affichés pour le résultat, jamais l'inverse — l'une des erreurs les plus fréquentes en TP. Cette règle vit dans une seule fonction, pour ne jamais être appliquée de façon incohérente d'une section à l'autre.

**Génération LaTeX.** Rédactions complètes utilisant le package `siunitx` : le modèle de mesure, une description de chaque source d'incertitude, les coefficients de sensibilité, l'étape de Welch-Satterthwaite quand elle est pertinente, le bilan d'incertitude, et le résultat final encadré. Plusieurs rédactions peuvent être assemblées en une seule annexe.

**Tracé de courbes.** `plot_courbes.py` lit les dicts produits par `gum_calc.py` sans rien recalculer : nuage de points avec barres d'erreur, régression affine ou non affine superposée, bande d'incertitude autour de l'ajustement (propagation analytique pour le cas affine, par différences finies pour le cas non affine), résidus, séries multiples sur les mêmes axes, export direct en PDF vectoriel pour Overleaf.

---

## Architecture et démarche

Le code est délibérément séparé en deux parties qui ne se mélangent jamais : une partie de calcul qui ne connaît rien au LaTeX, et une partie de formatage qui n'exécute aucune physique, se contentant de réutiliser des résultats que la couche de calcul a déjà produits.

J'ai conçu l'architecture et la logique de calcul moi-même, sur la base du cours de métrologie de L2 Physique à l'UPEC, puis j'ai travaillé avec [Claude (Anthropic)](https://www.anthropic.com) pour traduire cette logique en Python : le moteur de propagation SymPy, la mise en forme LaTeX, le pipeline de régression. Comprendre le GUM assez bien pour vérifier chaque résultat a compté davantage que l'écriture du code elle-même — une IA peut produire du code correct autour d'une idée fausse aussi facilement qu'autour d'une idée juste.

---

## Pistes d'amélioration

- **Covariance optionnelle, pas automatique.** Le moteur ne prend en compte une corrélation entre deux grandeurs d'entrée que si elle lui est explicitement fournie. Les seuls cas traités automatiquement sont un mesurande construit à partir de la pente et de l'ordonnée à l'origine d'une régression linéaire (`full_pipeline_regression_to_measurand`) — il n'existe pas d'équivalent pour la régression non affine, où la covariance entre deux paramètres corrélés (fréquent entre amplitude et taux de décroissance) doit être transmise à la main via `reg["cov"]`.
- **Propagation strictement linéaire (premier ordre).** Dérivées partielles évaluées à la valeur nominale, conforme au GUM pour des modèles bien comportés. Pour des modèles fortement non linéaires, le supplément du GUM recommande une approche Monte-Carlo, non implémentée — cette limitation s'applique aussi à la matrice de covariance de `nonlinear_regression`, qui reste l'approximation asymptotique locale de `curve_fit` (linéarisation au voisinage de l'optimum), et à la bande d'incertitude de `plot_nonlinear_regression`, propagée par différences finies plutôt que par dérivation symbolique.
- **Pas de régression linéaire pondérée.** `linear_regression` suppose que chaque point de mesure porte la même incertitude sur y. `nonlinear_regression`, plus récente, accepte en revanche un paramètre `sigma` optionnel pour pondérer l'ajustement par une incertitude connue point par point.
- **Algèbre des unités symbolique, pas numérique.** Le moteur traite les noms d'unités comme des tokens : il ne simplifie pas automatiquement `\kilo\gram` face à `\gram`, mélanger les préfixes produit un résultat non simplifié mais pas incorrect.

Une extension serait une suite de tests comparant la sortie du moteur à des exemples du GUM vérifiés à la main, pour détecter automatiquement les régressions au fil de l'évolution du code — à ce jour, chaque nouvelle fonctionnalité est vérifiée manuellement (cas connu comparé à un calcul de référence, relecture du LaTeX généré dans Overleaf) plutôt que par une suite de tests persistée dans le dépôt.

---

## Licence

MIT — voir [LICENSE](LICENSE).