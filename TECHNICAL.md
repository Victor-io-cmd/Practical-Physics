# gum_calc.py — Technical document

Ce document explique ce que fait chaque morceau de `gum_calc.py`, en partant de zéro. L'idée est d'expliquer **le code lui-même** : qu'est-ce qu'une fonction reçoit, qu'est-ce qu'elle renvoie, et pourquoi elle est écrite comme ça.

Avant de commencer, trois notions Python qui reviennent partout dans ce fichier :

- **Une fonction** est un bloc de code auquel on donne un nom. On lui passe des informations (les *paramètres*, entre parenthèses), elle fait un calcul, et elle *renvoie* (`return`) un résultat.
- **Un dict** (dictionnaire) est une boîte qui associe des clés à des valeurs, comme un index. `{"U": 5.0, "I": 0.5}` veut dire : la clé `"U"` vaut `5.0`, la clé `"I"` vaut `0.5`. On lit une valeur avec `dico["U"]`.
- **Une liste** (`list`) est une suite ordonnée de valeurs : `[1, 2, 3]`.

Le fichier est coupé en trois grandes parties : le moteur de calcul GUM, l'export LaTeX, et le pipeline qui enchaîne régression + calcul GUM. On les prend dans l'ordre.

---

## Partie 1 — Le moteur de calcul GUM

### `UncertaintyInput` — la fiche d'identité d'une incertitude

C'est une "classe", c'est-à-dire un moule pour fabriquer un objet qui regroupe plusieurs informations liées. Ici, l'objet regroupe tout ce qui caractérise l'incertitude d'une grandeur : sa valeur `u`, son `type` (A, B ou exact), ses degrés de liberté `nu`, etc.

Avant cette classe, le code stockait ça dans un simple dict. Le problème d'un dict, c'est qu'il accepte n'importe quelle clé sans broncher — si on écrit `"U"` au lieu de `"u"` par erreur, rien ne le signale avant que le calcul plante (ou pire, donne un résultat faux silencieusement). La classe `UncertaintyInput`, elle, vérifie dès sa création que `type` est bien `"A"`, `"B"` ou `"exact"`, et que `u` n'est pas négatif. Si ce n'est pas le cas, elle arrête tout immédiatement avec un message clair.

Elle propose aussi `from_dict`, qui accepte l'ancien format dict pour ne pas casser le code existant, mais qui prévient (sans bloquer) si une clé inconnue traîne dedans.

### Les cinq fonctions `uncertainty_type_*`

Ce sont les portes d'entrée : à partir d'une mesure brute, elles fabriquent un `UncertaintyInput` prêt à l'emploi.

- **`uncertainty_type_A(values)`** : on lui donne la liste des mesures répétées. Elle calcule la moyenne, l'écart-type, et en déduit l'incertitude-type `u = s/√N`. Elle exige au moins 2 mesures (en dessous, un écart-type n'a pas de sens).

- **`uncertainty_type_B_uniform(half_width)`** : on connaît une demi-largeur `a` (par exemple ±0,5 sur une graduation). Elle renvoie `u = a/√3`, qui est la formule pour une distribution uniforme.

- **`uncertainty_type_B_from_resolution(resolution)`** : variante pratique de la précédente. On donne directement la résolution `δ` de l'appareil (le dernier digit affiché), et la fonction divise par deux en interne avant d'appeler `uncertainty_type_B_uniform`.

- **`uncertainty_type_exact()`** : pour une constante parfaite (comme la vitesse de la lumière). Incertitude nulle, point.

- **`uncertainty_type_B_relative(u_standard, relative_knowledge=None)`** : pour quand l'incertitude est déjà connue numériquement (notice constructeur, ou résultat d'un calcul précédent). Le paramètre `relative_knowledge` est optionnel : s'il est fourni, la fonction estime les degrés de liberté avec la formule `ν = 1/(2r²)` (GUM annexe G) ; sinon elle laisse `ν = infini`, ce qui correspond à une incertitude de type B "classique" parfaitement connue.

### `validate_uncertainty_inputs` — le videur à l'entrée

Avant tout calcul, cette fonction vérifie juste une chose : est-ce que chaque variable annoncée dans `variable_names` a bien une valeur nominale ET une incertitude associée ? Si une grandeur manque quelque part, elle arrête tout avec la liste précise de ce qui manque, plutôt que de laisser Python planter plus loin avec une erreur cryptique du type "clé introuvable".

### `calculate_uncertainty` — le cœur du moteur

C'est la fonction la plus importante du fichier. Elle prend :
- une formule mathématique sous forme de texte (ex : `"U / I"`),
- la liste des noms de variables (`["U", "I"]`),
- leurs valeurs nominales,
- leurs incertitudes (objets `UncertaintyInput`),
- en option, des covariances entre certaines paires de variables.

Et voici, étape par étape, ce qu'elle fait réellement :

**1. Elle transforme le texte en formule mathématique manipulable.** La bibliothèque `sympy` (sympy = "symbolic Python") sait faire du calcul symbolique : pas juste évaluer `5/0.5 = 10`, mais manipuler `U/I` comme une vraie expression mathématique, dont on peut prendre la dérivée. `sp.sympify("U / I", ...)` convertit le texte `"U / I"` en un objet mathématique que sympy comprend.

**2. Elle vérifie qu'aucune variable inconnue ne traîne dans la formule.** Si on écrit `"U / Z"` mais qu'on a oublié de déclarer `Z` dans `variable_names`, le code le détecte et arrête tout avec un message explicite, plutôt que de laisser une erreur Python obscure surgir plus tard.

**3. Elle calcule la valeur nominale du résultat.** Elle remplace chaque symbole par sa valeur numérique (`U` devient `5.0`, `I` devient `0.5`) et évalue l'expression. Une fonction interne (`_safe_float`) gère les cas pièges : division par zéro, résultat infini, résultat "NaN" (Not a Number, une valeur mathématiquement indéfinie comme `0/0`). Dans tous ces cas, elle renvoie une erreur claire qui dit *pourquoi* ça a planté, plutôt qu'un message Python générique.

**4. Elle calcule les dérivées partielles — les "coefficients de sensibilité".** C'est le cœur de la méthode GUM : pour savoir comment l'incertitude sur `U` se répercute sur le résultat, il faut savoir "à quelle vitesse" le résultat change quand `U` change. C'est exactement ce que mesure la dérivée partielle `∂f/∂U`. Sympy calcule cette dérivée automatiquement (`sp.diff`), sans qu'on ait à la poser à la main — c'est tout l'intérêt d'utiliser du calcul symbolique plutôt que de taper la formule de propagation soi-même pour chaque TP.

**5. Elle assemble la variance composée.** Pour chaque variable, elle calcule sa contribution : `(coefficient de sensibilité)² × (incertitude)²`. Elle additionne tout. C'est la formule GUM classique de propagation pour des grandeurs indépendantes.

Au passage, elle surveille un point délicat : si l'incertitude relative d'une grandeur (son `u` divisé par sa valeur) dépasse 10 %, elle envoie un avertissement. Pourquoi : la méthode GUM standard suppose que la formule est à peu près une droite autour du point de mesure (approximation "au premier ordre"). Si l'incertitude est trop grande par rapport à la valeur, cette approximation peut sous-estimer le résultat final. Le code ne bloque pas — il prévient juste, pour que tu puisses juger si une approche Monte-Carlo serait plus appropriée.

**6. Elle ajoute les termes croisés si des covariances sont fournies.** Par défaut, le moteur suppose que toutes les grandeurs d'entrée sont indépendantes (ex : la tension mesurée n'a rien à voir avec le courant mesuré). Mais il existe un cas fréquent où ce n'est pas vrai : quand deux paramètres viennent de la *même* régression linéaire (la pente et l'ordonnée à l'origine sont mathématiquement liées). Si on fournit ces covariances, la fonction ajoute le terme correctif `2 × c_i × c_j × covariance` prévu par le GUM (§5.2, formule 13).

**7. Elle construit le "budget".** C'est simplement : quel pourcentage de la variance totale vient de chaque source. Pratique pour répondre à "qu'est-ce qui limite la précision de ma mesure ?"

Elle renvoie un dict avec tout ça : `result`, `uc` (l'incertitude composée), `sensitivities`, `contributions`, `budget`, etc.

### `welch_satterthwaite` — combien de chiffres on peut vraiment faire confiance

Quand on combine plusieurs incertitudes, certaines sont très bien connues (type B, infiniment fiables en théorie) et d'autres sont estimées sur peu de mesures (type A avec N petit, donc moins fiables). La formule de Welch-Satterthwaite calcule un nombre "effectif" de degrés de liberté qui tient compte de ce mélange.

Concrètement : si une variable a `ν = infini` (typiquement type B sans info supplémentaire), elle ne limite pas la confiance qu'on peut avoir. Si une variable a `ν` fini (type A avec peu de mesures, ou type B avec connaissance relative), elle pèse dans le calcul. Le code ne prend en compte que les `ν` finis dans la somme.

Une fois `ν_eff` calculé, le facteur d'élargissement `k` n'est plus automatiquement 2 (qui correspond à une confiance "infinie", loi normale) : il est lu dans la table de Student via `scipy_stats.t.ppf`, ce qui donne un `k` légèrement plus grand quand on a peu de degrés de liberté — c'est la pénalité logique pour une estimation moins solide.

Si aucune variable n'a de `ν` fini, la fonction retombe directement sur `k = 2.0` sans passer par la table de Student.

### `expanded_uncertainty` — la toute dernière étape

Une ligne : `U = k × uc`. C'est l'incertitude "élargie", celle qu'on affiche dans le rapport final (généralement à 95 % de confiance).

### `linear_regression` — la droite des moindres carrés

On lui donne une liste de points `(x, y)`, elle trouve la droite `y = θ₀ + θ₁·x` qui minimise la somme des carrés des écarts (méthode des moindres carrés ordinaires, "OLS").

Détail technique important : avant de calculer, le code **centre** les données (il soustrait la moyenne de `x` et de `y` à chaque point). Ce n'est pas une question de style : si tes données sont par exemple toutes autour de `1 000 000`, calculer directement des sommes de carrés peut perdre en précision numérique (deux très grands nombres très proches se soustraient mal en informatique — c'est ce qu'on appelle la "cancellation catastrophique"). Centrer les données élimine ce problème sans changer le résultat final.

La fonction renvoie, entre autres :
- `theta0`, `theta1` : l'ordonnée à l'origine et la pente,
- `u_theta0`, `u_theta1` : leurs incertitudes-types, dérivées de la dispersion des résidus (l'écart entre les points mesurés et la droite ajustée),
- `cov_theta01` : la covariance entre `theta0` et `theta1`. C'est un point souvent oublié dans les calculs d'incertitude faits à la main : la pente et l'ordonnée à l'origine d'une même régression ne sont *pas* indépendantes. Cette covariance est calculée analytiquement (`-x̄·s²_res/Sxx`) et peut être réinjectée dans `calculate_uncertainty` via le paramètre `covariances` pour un calcul rigoureux,
- `r2` : le coefficient de détermination, qui indique la qualité de l'ajustement (1 = parfait).

Le code a aussi des garde-fous : moins de 3 points refusés (une droite a besoin d'au moins 3 points pour qu'on puisse juger sa qualité), toutes les valeurs de `x` identiques refusées (la droite n'est pas définie), et une détection spécifique du cas où les `x` sont distincts mais "écrasés" par un offset commun trop grand (encore la cancellation numérique).

### `_round_half_up` et `round_to_sig_figs` — arrondir correctement

Ce sont des fonctions discrètes mais essentielles. Le problème : la fonction native de Python, `round()`, utilise l'arrondi "bancaire" (0.5 arrondit parfois vers le bas), ce qui n'est pas la convention attendue en sciences (où 0,5 arrondit toujours vers le haut). De plus, les nombres décimaux ne sont pas stockés exactement par l'ordinateur (`0.15` est en réalité stocké comme `0.1499999...`), ce qui peut faire basculer un arrondi du mauvais côté.

`_round_half_up` contourne ce problème en passant par le type `Decimal` de Python, qui représente les nombres décimaux exactement (comme on les écrirait à la main), plutôt que dans le format binaire approximatif utilisé par défaut.

`round_to_sig_figs` l'utilise pour arrondir à un nombre donné de *chiffres significatifs* (pas de décimales — la différence compte : 3 chiffres significatifs de `0.00123` donnent `0.00123`, alors que 3 décimales donneraient `0.001`).

### `format_result` — la règle d'or de la métrologie

Cette fonction applique une règle simple mais qu'on oublie souvent en TP : **c'est l'incertitude qui décide du nombre de décimales affichées pour le résultat, jamais l'inverse.** On arrondit d'abord `U` à 2 chiffres significatifs, puis on arrondit le résultat à la même décimale que `U`. Exemple : si `U` arrondi vaut `0.012`, le résultat est arrondi au millième, peu importe combien de décimales il avait au départ.

Cas particulier géré : si toutes les grandeurs d'entrée sont exactes, `U` vaut zéro, et il n'y a alors plus de règle d'arrondi pilotée par l'incertitude — le résultat est simplement arrondi à un nombre de chiffres significatifs fixé par défaut (`global_sig_figs`).

### `full_gum_analysis` — le chef d'orchestre

Cette fonction ne fait (presque) aucun calcul elle-même : elle appelle dans l'ordre `calculate_uncertainty`, `welch_satterthwaite`, `expanded_uncertainty`, puis `format_result`, et rassemble tous les résultats dans un seul objet.

Ce qu'elle renvoie n'est pas un dict ordinaire mais un `MappingProxyType` — une version "verrouillée en lecture seule" d'un dict. L'idée : une fois le calcul fait, personne (même par erreur, plus loin dans le notebook) ne peut modifier `res["uc"]` sans faire exprès. C'est une sécurité, pas une fonctionnalité au sens scientifique.

---

## Partie 2 — Export LaTeX

Cette partie ne fait *aucun* calcul scientifique nouveau. Son seul travail : transformer les nombres déjà calculés en Partie 1 en texte LaTeX correctement formaté, pour qu'on puisse le coller dans Overleaf.

### `_escape_latex`

LaTeX a des caractères réservés (`%`, `&`, `_`, etc.) qui ont un sens spécial. Si un nom de mesurande contient un de ces caractères par accident, ça casserait la compilation. Cette fonction les remplace par leur version "neutralisée" (ex : `%` devient `\%`). Elle ne s'applique qu'au texte libre (les noms), jamais aux symboles mathématiques saisis volontairement par l'utilisateur (comme `\theta_1`), qui doivent rester du LaTeX brut.

### `_magnitude_of`, `_mantissa_exp`

Des fonctions utilitaires pour décomposer un nombre en notation scientifique (`mantisse × 10^exposant`), en gérant proprement les cas limites (ex : un arrondi qui ferait passer la mantisse de `9.99` à `10.0`, ce qui demanderait de réajuster l'exposant).

### `_num` et `_si`

Ce sont les deux fonctions qui produisent le LaTeX final pour un nombre, en s'appuyant sur le package `siunitx` :
- `_num(valeur)` produit `\num{...}` pour un nombre sans unité,
- `_si(valeur, unité)` produit `\qty{...}{...}` pour un nombre avec unité.

Elles choisissent automatiquement la notation décimale ou scientifique selon l'ordre de grandeur du nombre (entre `10⁻²` et `10³`, c'est en décimal ; en dehors, en scientifique).

### `_format_result_uncertainty`

C'est la fonction qui produit la ligne finale `valeur ± incertitude` en LaTeX. Elle gère un détail piège : il faut que la valeur et l'incertitude soient affichées avec le *même* nombre de décimales et, en notation scientifique, la *même* puissance de 10 — sinon le résultat est illisible (`1.234 ± 0.01e-3` n'a pas de sens). Elle gère aussi le cas où le résultat est lui-même proche de zéro (mesure compatible avec zéro) sans planter sur le calcul de magnitude.

### Les blocs `_bilan_*_block`

`generate_bilan` (qui produit le LaTeX complet d'un mesurande) est découpé en six petites fonctions, chacune responsable d'une seule partie du texte final :

1. `_bilan_model_block` — la phrase d'intro et la formule du modèle.
2. `_bilan_source_blocks` — une explication par grandeur d'entrée (type A, type B, exact).
3. `_bilan_sensitivity_block` — les coefficients de sensibilité (les dérivées calculées en Partie 1, mais affichées en LaTeX).
4. `_bilan_propagation_block` — la formule de propagation et le calcul de `u_c`.
5. `_bilan_ws_block` — le paragraphe Welch-Satterthwaite (ou la phrase "k = 2" si non applicable).
6. `_bilan_budget_result_block` — le budget d'incertitudes et le résultat encadré final.

Pourquoi découper comme ça plutôt que tout mettre dans une seule fonction géante : chaque bloc peut être testé et corrigé indépendamment, et un bug dans l'affichage du budget ne risque pas de casser l'affichage de la formule du modèle.

Un détail récurrent dans ces fonctions : la bascule automatique entre une ligne simple (`\[ ... \]`) et un environnement `align*` à plusieurs lignes quand le texte devient trop long ou compte trop de termes. C'est pour éviter qu'une formule à 5 variables ne déborde de la page dans le PDF final — purement esthétique, mais important pour un rapport propre.

### L'algèbre d'unités (`_parse_unit`, `_divide_units`, etc.)

Ce groupe de fonctions répond à un besoin précis : quand on calcule une pente de régression, son unité est `y_unit / x_unit`. Si on faisait ça par simple "collage de texte", diviser une vitesse (`\meter\per\second`) par un temps (`\second`) donnerait `\meter\per\second\per\second` — qui compile en LaTeX mais s'affiche moche (`m/(s·s)` au lieu de `m/s²`).

Le code traite donc chaque unité comme une vraie fraction (un numérateur et un dénominateur), les combine algébriquement, simplifie ce qui s'annule, et reconstruit une unité propre (`\meter\per\square\second`). C'est de la plomberie interne, transparente pour toi : tu donnes juste `x_unit` et `y_unit`, le code se débrouille.

Limite assumée et documentée dans le code : cette algèbre est purement basée sur le *nom* du token. Elle ne convertit pas `\kilo\gram` en `\gram` automatiquement — ce sont vus comme deux unités différentes. Si tu mélanges des préfixes différents pour la même grandeur physique, l'unité ne se simplifiera pas (sans erreur, juste un affichage non simplifié).

---

## Partie 3 — Le pipeline intégré

### `full_pipeline_regression_to_measurand`

Cette fonction répond à un cas précis et fréquent en TP : un mesurande final dépend de la pente ou de l'ordonnée à l'origine d'une régression linéaire (par exemple, une résistance déduite de la pente d'une droite U = f(I)).

Elle enchaîne, dans l'ordre :
1. `linear_regression` sur les données expérimentales,
2. injection automatique de `theta0`/`theta1` (et leurs incertitudes) dans les valeurs nominales et incertitudes du mesurande final,
3. si les deux paramètres `theta0` ET `theta1` sont utilisés dans la formule finale, récupération de leur covariance OLS et transmission à `generate_bilan` pour un calcul rigoureux (et non une fausse hypothèse d'indépendance),
4. génération du bilan LaTeX de la régression, puis du bilan LaTeX du mesurande, séparés par un saut de page.

Les paramètres `couple_theta0` et `couple_theta1` sont volontairement explicites (à toi de dire "oui, mon mesurande dépend de theta0") plutôt que devinés automatiquement à partir du nom des variables — ça évite qu'une simple faute de frappe dans `variable_names` désactive silencieusement la prise en compte de la covariance sans prévenir.

### `generate_bilan_regression`

Produit le bloc LaTeX pour une régression utilisée seule (sans mesurande dérivé) : modèle, estimateurs, incertitudes, R², résultat encadré. Réutilise `_format_result_uncertainty` pour garder exactement la même règle d'arrondi (incertitude → 2 chiffres significatifs → décimales du résultat) que le reste du fichier, plutôt que d'avoir une mise en forme différente pour la régression seule.

### `generate_annexe`

La fonction la plus simple du fichier : elle prend la liste de tous les bilans déjà générés (un texte LaTeX par mesurande) et les colle ensemble sous un `\section{Annexe : Bilans d'incertitudes}`. Aucun calcul, juste de l'assemblage.

---

## Pourquoi cette architecture

Trois choix de conception reviennent dans tout le fichier, et valent la peine d'être explicités :

**Séparer calcul et affichage.** `calculate_uncertainty` ne sait rien du LaTeX, et `generate_bilan` ne refait aucun calcul (elle réutilise le `res` déjà produit). Ça veut dire que tu peux changer la mise en forme LaTeX sans jamais toucher à la physique, et inversement.

**Échouer fort plutôt qu'échouer en silence.** Presque chaque fonction vérifie ses entrées avant de calculer, et lève une erreur explicite (`ValueError`) avec un message qui dit précisément quoi corriger, plutôt que de laisser un résultat faux passer inaperçu dans un rapport.

**Centraliser chaque règle à un seul endroit.** L'arrondi half-up n'existe qu'une fois (`_round_half_up`), la règle "l'incertitude pilote les décimales" n'existe qu'une fois (`format_result`). Toute évolution future ne se fait qu'à un seul endroit, ce qui évite que deux fonctions du fichier appliquent silencieusement deux conventions différentes.