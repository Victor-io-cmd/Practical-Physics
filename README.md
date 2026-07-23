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
pip install sympy scipy
```

Aucune autre dépendance. Fonctionne avec Python 3.10+ (utilisation de `list[float]` en annotation de type).

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

---

## Structure du projet

```
gum-calc/
├── gum_calc.py               # Moteur de calcul et export LaTeX
├── gum_notebook.skill        # Skill Claude : génère automatiquement le notebook ci-dessous
├── gum_uncertainties.ipynb   # Notebook modèle, une cellule par mesurande
└── README.md
```

`gum_notebook.skill` est un [Claude Skill](https://www.anthropic.com) : un ensemble d'instructions qui permet à Claude de générer automatiquement `gum_uncertainties.ipynb` pour un nouveau TP, câblé sur `gum_calc.py`, plutôt que de dupliquer et remplir le modèle à la main pour chaque compte rendu.

---

## Fonctionnalités

**Typage des incertitudes.** Cinq fonctions couvrent les cas standards du GUM : mesures répétées (type A), résolution d'instrument ou tolérance connue (type B), une incertitude-type directement connue (par exemple propagée depuis un calcul précédent), et les constantes exactes.

**Propagation symbolique.** La fonction centrale prend une formule en texte brut (comme `"U / I"`), la parse avec SymPy, et la différentie automatiquement par rapport à chaque variable. Plus besoin de dériver les formules de propagation à la main pour chaque nouveau mesurande.

**Facteur d'élargissement de Welch-Satterthwaite.** Lorsque des sources d'incertitude de fiabilités différentes sont combinées (une source de type B bien connue à côté d'une source de type A estimée sur peu de mesures), le moteur calcule le nombre de degrés de liberté effectif et lit le facteur d'élargissement `k` correct dans la table de Student, plutôt que de systématiquement supposer `k = 2`.

**Régression linéaire avec covariance.** Un ajustement par moindres carrés ordinaires qui renvoie aussi la covariance entre la pente et l'ordonnée à l'origine, une grandeur presque toujours oubliée dans les calculs à la main mais qui compte dès qu'un résultat final dépend des deux.

**Arrondi correct, à chaque fois.** C'est toujours l'incertitude qui détermine le nombre de chiffres significatifs affichés pour le résultat, jamais l'inverse — l'une des erreurs les plus fréquentes en TP. Cette règle vit dans une seule fonction, pour ne jamais être appliquée de façon incohérente d'une section à l'autre.

**Génération LaTeX.** Rédactions complètes utilisant le package `siunitx` : le modèle de mesure, une description de chaque source d'incertitude, les coefficients de sensibilité, l'étape de Welch-Satterthwaite quand elle est pertinente, le bilan d'incertitude, et le résultat final encadré. Plusieurs rédactions peuvent être assemblées en une seule annexe.

---

## Architecture et démarche

Le code est délibérément séparé en deux parties qui ne se mélangent jamais : une partie de calcul qui ne connaît rien au LaTeX, et une partie de formatage qui n'exécute aucune physique, se contentant de réutiliser des résultats que la couche de calcul a déjà produits.

J'ai conçu l'architecture et la logique de calcul moi-même, sur la base du cours de métrologie de L2 Physique à l'UPEC, puis j'ai travaillé avec [Claude (Anthropic)](https://www.anthropic.com) pour traduire cette logique en Python : le moteur de propagation SymPy, la mise en forme LaTeX, le pipeline de régression. Comprendre le GUM assez bien pour vérifier chaque résultat a compté davantage que l'écriture du code elle-même — une IA peut produire du code correct autour d'une idée fausse aussi facilement qu'autour d'une idée juste.

---

## Pistes d'amélioration

- **Covariance optionnelle, pas automatique.** Le moteur ne prend en compte une corrélation entre deux grandeurs d'entrée que si elle lui est explicitement fournie. Le seul cas traité automatiquement est un mesurande construit à partir de la pente et de l'ordonnée à l'origine d'une régression.
- **Propagation strictement linéaire (premier ordre).** Dérivées partielles évaluées à la valeur nominale, conforme au GUM pour des modèles bien comportés. Pour des modèles fortement non linéaires, le supplément du GUM recommande une approche Monte-Carlo, non implémentée.
- **Pas de régression pondérée.** L'ajustement linéaire suppose que chaque point de mesure porte la même incertitude sur y.
- **Algèbre des unités symbolique, pas numérique.** Le moteur traite les noms d'unités comme des tokens : il ne simplifie pas automatiquement `\kilo\gram` face à `\gram`, mélanger les préfixes produit un résultat non simplifié mais pas incorrect.

Une extension naturelle serait une suite de tests comparant la sortie du moteur à des exemples du GUM vérifiés à la main, pour détecter automatiquement les régressions au fil de l'évolution du code.

---

## Licence

MIT — voir [LICENSE](LICENSE).