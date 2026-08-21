"""plot_courbes.py — tracé et export de courbes TP

Ne recalcule rien : lit les dicts de gum_calc (linear_regression,
full_gum_analysis) et les UncertaintyInput associés pour tracer/exporter.
Chaque fonction retourne (fig, ax) et accepte ax= en option pour superposer
plusieurs tracés sur les mêmes axes.

Périmètre : régression affine et ses résidus (plot_regression,
plot_residuals), régression non affine (plot_nonlinear_regression, cf.
gum_calc.nonlinear_regression), séries multiples sur les mêmes axes/même
échelle (plot_multi_series), et annotation ponctuelle (annotate_point).

plot_measurement_series et plot_measurand_vs_parameter restent hors
périmètre, jugées redondantes avec les usages réels du TP.

add_twin_series (axe y secondaire) a été retirée : ne correspondait pas
au besoin réel, qui est plusieurs courbes sur les MÊMES axes, à la même
échelle (cf. plot_multi_series) plutôt que deux grandeurs de nature
différente sur deux échelles superposées.

plot_nonlinear_regression est une fonction SŒUR de plot_regression, pas
une généralisation de celle-ci : plot_regression reste intouchée (zéro
risque de régression sur les CR déjà écrits, cf. feuille de route
nonlinear_regression, §3) — les deux fonctions ne partagent que les
utilitaires internes (_new_ax, _validate_data).

Pour l'annotation ponctuelle, annotate_point s'appelle directement sur
l'ax= retourné par n'importe laquelle des fonctions de tracé ci-dessus —
depuis la notebook, ou depuis l'app Streamlit qui l'expose comme option
dans chaque onglet.
"""

import math
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

_OUTPUT_DIR = "figures"


def set_output_dir(path: str) -> None:
    global _OUTPUT_DIR
    _OUTPUT_DIR = path


@dataclass
class ExportedFigure:
    """Résultat d'export_figure : lie path/name/output_dir pour éviter
    la recopie manuelle de `name` entre export_figure et
    latex_figure_block (cf. rapport, section 5 point 3)."""
    path: str
    name: str
    output_dir: str


def export_figure(fig, name: str, output_dir: str = None) -> ExportedFigure:
    """Exporte fig en PDF vectoriel.

    Retourne un ExportedFigure (path, name, output_dir) — le passer
    directement à latex_figure_block(exported, ...) au lieu de retaper
    `name` à la main élimine tout risque de désynchronisation entre les
    deux appels (faute de frappe, oubli). L'ancien usage par chaîne reste
    accepté côté latex_figure_block pour compatibilité.
    """
    out_dir = output_dir if output_dir is not None else _OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(path)
    print(f"Figure exportée : {path}")
    return ExportedFigure(path=path, name=name, output_dir=out_dir)


def latex_figure_block(
    name,
    caption: str,
    label: str = None,
    width: float = 0.8,
    path_prefix: str = "",
) -> str:
    """Bloc \\includegraphics pour la figure exportée sous `name`.

    `name` accepte deux formes :
    - un ExportedFigure retourné par export_figure (recommandé — le nom
      est lu directement dessus, aucune recopie manuelle possible, donc
      aucune désynchronisation possible) ;
    - une chaîne brute (ancien usage).

    Par défaut, AUCUN préfixe de dossier n'est ajouté devant le nom du
    PDF (`{fig_name}.pdf`, pas `figures/{fig_name}.pdf`) : le cas normal
    est un PDF déposé directement à la racine du projet Overleaf, à côté
    du .tex compilé (organisation locale en ROOT_DIR/figures/ non
    reproduite côté Overleaf). Avant, le préfixe était dérivé
    automatiquement du dossier passé à set_output_dir, ce qui cassait
    l'import dès que l'arborescence Overleaf ne reprenait pas ce même
    sous-dossier `figures/` — cas le plus courant en pratique.

    Si un TP a réellement besoin d'un sous-dossier côté Overleaf (ex.
    projet organisé en `figures/tp1/...`), passer explicitement
    path_prefix="figures/tp1" (avec ou sans / final).
    """
    fig_name = name.name if isinstance(name, ExportedFigure) else name
    fig_label = label if label is not None else f"fig:{fig_name}"
    prefix = path_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return (
        "\\begin{figure}[H]\n"
        "    \\centering\n"
        f"    \\includegraphics[width={width}\\linewidth]{{{prefix}{fig_name}.pdf}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{fig_label}}}\n"
        "\\end{figure}"
    )


def axis_label(nom: str, symbole: str, unite: str = "") -> str:
    """axis_label("Tension","U","Volt") -> "Tension $U$ (Volt)"."""
    label = f"{nom} ${symbole}$"
    if unite:
        label += f" ({unite})"
    return label


def _new_ax(ax, figsize):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _validate_data(**named_lists):
    """Valide que toutes les listes fournies (non None) ont la même longueur
    et ne contiennent aucune valeur non finie (NaN, +-inf). Ignore les
    scalaires (int/float) : u_x/u_y acceptent une incertitude constante
    partagée par tous les points, ce n'est pas une série à valider terme à
    terme.

    Lève un message explicite plutôt que de laisser matplotlib échouer
    plus loin avec une trace illisible pour un étudiant — cas classique
    d'un fichier Excel mal rempli (cellule vide -> NaN via pandas)."""
    sequences = {
        name: v for name, v in named_lists.items()
        if v is not None and not isinstance(v, (int, float))
    }

    lengths = {name: len(v) for name, v in sequences.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in lengths.items())
        raise ValueError(
            f"plot_courbes : longueurs incohérentes entre séries ({detail}). "
            "Toutes les listes fournies à une même fonction de tracé doivent "
            "avoir la même longueur (une valeur scalaire est acceptée pour "
            "une incertitude constante partagée)."
        )

    for name, seq in sequences.items():
        bad_idx = [i for i, v in enumerate(seq) if not math.isfinite(v)]
        if bad_idx:
            shown = bad_idx[:5]
            suffix = "..." if len(bad_idx) > 5 else ""
            raise ValueError(
                f"plot_courbes : valeur(s) non finie(s) dans '{name}' à "
                f"l'indice(s) {shown}{suffix} (NaN ou infini — probablement "
                "une cellule vide ou mal formatée dans le fichier source). "
                "Nettoyez ces valeurs avant le tracé."
            )


def _fit_label(reg):
    r"""Repli SANS aucune incertitude connue (theta0/theta1 bruts, 4 c.s.
    fixes via :.4g). N'est utilisé QUE si l'appelant ne fournit pas
    fit_label explicitement.

    ATTENTION COHÉRENCE GUM : ce repli n'est PAS synchronisé avec la
    règle métrologique de gum_calc.format_result (décimales imposées par
    2 c.s. de l'incertitude élargie). Pour que la légende du graphe
    affiche exactement le même nombre que le bilan LaTeX du même TP,
    l'appelant (le notebook, qui a accès à gum_calc) doit construire lui-
    même la chaîne via format_result et la passer en fit_label=. Exemple :

        fmt0 = format_result(reg['theta0'], reg['u_theta0'])
        fmt1 = format_result(reg['theta1'], reg['u_theta1'])
        fit_label = (
            rf"$y = {fmt0['result']:.{fmt0['decimals']}f} "
            rf"+ {fmt1['result']:.{fmt1['decimals']}f}\,x$"
        )
        plot_regression(..., fit_label=fit_label)

    plot_courbes ne peut pas appeler format_result lui-même (il n'importe
    jamais gum_calc, cf. architecture tranchée) : la responsabilité de la
    cohérence de rond reste donc du côté de l'appelant.
    """
    sign = "+" if reg["theta1"] >= 0 else "-"
    return rf"$y = {reg['theta0']:.4g} {sign} {abs(reg['theta1']):.4g}\,x$ (non arrondi GUM)"


def plot_regression(
    x_data: list,
    y_data: list,
    reg: dict = None,
    u_x=None,
    u_y=None,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    data_label: str = "Mesures",
    fit_label: str = None,
    data_color: str = "tab:blue",
    fit_color: str = "tab:red",
    marker: str = "o",
    show_fit: bool = True,
    show_band: bool = False,
    k_band: float = 1.0,
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple = (7, 5),
    ax=None,
):
    """Nuage de points (+ barres d'erreur u_x/u_y) et, en option, droite
    ajustée issue de gum_calc.linear_regression (reg["theta0"/"theta1"/"y_pred"]).

    show_fit=False : trace uniquement le nuage de points, sans ajustement.
    `reg` devient alors optionnel — utile pour explorer un nuage avant de
    décider si un modèle affine est pertinent, sans même lancer le calcul
    de régression (pas seulement le masquer visuellement). show_fit=True
    exige reg (ValueError explicite sinon, plutôt qu'un plantage plus loin).

    show_band=True superpose une bande à ±k_band·u autour de la droite,
    propagée depuis u_theta0/u_theta1/cov_theta01 (reg) selon
    Var(ŷ(x)) = u_theta0² + x²·u_theta1² + 2x·cov_theta01. k_band=1 par
    défaut : c'est une incertitude-type, jamais l'incertitude élargie U,
    sauf changement explicite de k_band. Ignoré si show_fit=False.

    data_color/fit_color/marker : paramétrables pour permettre une
    superposition à plusieurs couches de couleurs distinctes (ex. cette
    régression + un nuage supplémentaire tracé par-dessus via ax=) sans
    toucher au code de la fonction. Valeurs par défaut inchangées.
    """
    if show_fit and reg is None:
        raise ValueError(
            "plot_regression : show_fit=True nécessite reg (dict retourné "
            "par gum_calc.linear_regression). Passe show_fit=False pour "
            "tracer uniquement le nuage de points sans reg."
        )
    _validate_data(x_data=x_data, y_data=y_data, u_x=u_x, u_y=u_y)
    fig, ax = _new_ax(ax, figsize)

    if u_x is not None or u_y is not None:
        ax.errorbar(
            x_data, y_data, xerr=u_x, yerr=u_y,
            fmt=marker, color=data_color, ecolor=data_color,
            elinewidth=1, capsize=3, markersize=5,
            label=data_label, zorder=3,
        )
    else:
        ax.scatter(x_data, y_data, color=data_color, marker=marker, zorder=3, label=data_label)

    if show_fit:
        if fit_label is None:
            fit_label = _fit_label(reg)

        order = sorted(range(len(x_data)), key=lambda i: x_data[i])
        x_sorted = [x_data[i] for i in order]
        y_pred_sorted = [reg["y_pred"][i] for i in order]
        ax.plot(x_sorted, y_pred_sorted, color=fit_color, linewidth=1.5, label=fit_label, zorder=2)

        if show_band:
            u_theta0, u_theta1, cov01 = reg["u_theta0"], reg["u_theta1"], reg["cov_theta01"]
            u_y_pred = [
                math.sqrt(max(u_theta0 ** 2 + xi ** 2 * u_theta1 ** 2 + 2 * xi * cov01, 0.0))
                for xi in x_sorted
            ]
            lower = [y - k_band * u for y, u in zip(y_pred_sorted, u_y_pred)]
            upper = [y + k_band * u for y, u in zip(y_pred_sorted, u_y_pred)]
            band_label = r"$\pm u(\hat y)$" if k_band == 1.0 else rf"$\pm {k_band:g}\,u(\hat y)$"
            ax.fill_between(x_sorted, lower, upper, color=fit_color, alpha=0.15, zorder=1, label=band_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if xscale != "linear":
        ax.set_xscale(xscale)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _fit_label_nonlinear(reg):
    r"""Repli SANS incertitude connue (paramètres bruts, 4 c.s. via :.4g).
    N'est utilisé QUE si l'appelant ne fournit pas fit_label explicitement.

    ATTENTION COHÉRENCE GUM (même avertissement que _fit_label) : ce repli
    n'est PAS synchronisé avec la règle métrologique de
    gum_calc.format_result. Pour que la légende affiche exactement les
    mêmes chiffres que le bilan LaTeX du même TP, l'appelant doit
    construire lui-même la chaîne via format_result sur chaque paramètre
    de reg["params"]/reg["u_params"], et la passer en fit_label=.
    plot_courbes ne peut pas appeler format_result lui-même (il n'importe
    jamais gum_calc, cf. architecture tranchée) : la responsabilité de la
    cohérence de rond reste donc du côté de l'appelant.
    """
    parts = [f"{name}={value:.4g}" for name, value in reg["params"].items()]
    return "Ajustement (" + ", ".join(parts) + ") — non arrondi GUM"


def _numeric_jacobian(model_func, x_grid, params, rel_step: float = 1e-6):
    """Jacobien numérique J(x)[i, j] = d model_func(x_i) / d params[j],
    par différences finies centrées, pas h = rel_step * max(|params[j]|, 1).

    Utilisé exclusivement par plot_nonlinear_regression (show_band=True) :
    approche numérique retenue en V1 plutôt qu'une dérivation symbolique
    (cf. feuille de route nonlinear_regression, §3 point 2) — pas de
    dépendance croisée à sympy pour un model_func utilisateur qui n'a
    aucune obligation d'être une expression sympy, différentiation
    centrée plutôt qu'avant/arrière pour une meilleure précision locale à
    coût de calcul quasi identique.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    n_params = len(params)
    J = np.empty((x_grid.size, n_params))
    for j, p in enumerate(params):
        h = rel_step * max(abs(p), 1.0)
        params_plus = list(params); params_plus[j] = p + h
        params_minus = list(params); params_minus[j] = p - h
        y_plus = np.asarray(model_func(x_grid, *params_plus), dtype=float)
        y_minus = np.asarray(model_func(x_grid, *params_minus), dtype=float)
        J[:, j] = (y_plus - y_minus) / (2 * h)
    return J


def plot_nonlinear_regression(
    x_data: list,
    y_data: list,
    reg: dict = None,
    u_x=None,
    u_y=None,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    data_label: str = "Mesures",
    fit_label: str = None,
    data_color: str = "tab:blue",
    fit_color: str = "tab:red",
    marker: str = "o",
    show_fit: bool = True,
    show_band: bool = False,
    k_band: float = 1.0,
    n_curve_points: int = 200,
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple = (7, 5),
    ax=None,
):
    """Fonction SŒUR de plot_regression (pas une généralisation — voir le
    docstring de module) dédiée aux régressions non affines issues de
    gum_calc.nonlinear_regression.

    `reg` : dict retourné par gum_calc.nonlinear_regression(...). Doit
    contenir "model_func", "params" (dict {nom: valeur}) et "param_names"
    (ordre) ; "cov" est requis en plus dès que show_band=True.

    Contrairement à plot_regression (qui relie les y_pred aux points de
    mesure x_data, valide pour une droite), la courbe ajustée est ici
    évaluée sur une grille fine de n_curve_points points régulièrement
    espacés entre min(x_data) et max(x_data) : un modèle non affine
    dessiné en ne reliant que les points de mesure produirait une ligne
    brisée trompeuse dès que les points sont espacés de façon inégale ou
    trop clairsemée pour restituer la courbure réelle du modèle.

    show_band=True superpose une bande à ±k_band·u autour de la courbe,
    propagée par différences finies (pas de dérivation symbolique, cf.
    feuille de route nonlinear_regression, §3 point 2) :
        Var(ŷ(x)) = J(x)^T · Cov(θ) · J(x)
    où J(x) est le vecteur des dérivées partielles du modèle par rapport
    à chaque paramètre, évalué en x (cf. _numeric_jacobian). k_band=1 par
    défaut : c'est une incertitude-type, jamais l'incertitude élargie U,
    sauf changement explicite de k_band. Ignoré si show_fit=False.

    fit_label/data_color/fit_color/marker : mêmes conventions que
    plot_regression (repli non arrondi GUM si fit_label omis — voir
    _fit_label_nonlinear).
    """
    if show_fit and reg is None:
        raise ValueError(
            "plot_nonlinear_regression : show_fit=True nécessite reg (dict "
            "retourné par gum_calc.nonlinear_regression). Passe "
            "show_fit=False pour tracer uniquement le nuage de points "
            "sans reg."
        )
    _validate_data(x_data=x_data, y_data=y_data, u_x=u_x, u_y=u_y)
    fig, ax = _new_ax(ax, figsize)

    if u_x is not None or u_y is not None:
        ax.errorbar(
            x_data, y_data, xerr=u_x, yerr=u_y,
            fmt=marker, color=data_color, ecolor=data_color,
            elinewidth=1, capsize=3, markersize=5,
            label=data_label, zorder=3,
        )
    else:
        ax.scatter(x_data, y_data, color=data_color, marker=marker, zorder=3, label=data_label)

    if show_fit:
        model_func   = reg["model_func"]
        param_names  = reg["param_names"]
        params       = [reg["params"][n] for n in param_names]

        if fit_label is None:
            fit_label = _fit_label_nonlinear(reg)

        x_grid = np.linspace(min(x_data), max(x_data), n_curve_points)
        y_grid = np.asarray(model_func(x_grid, *params), dtype=float)

        ax.plot(x_grid, y_grid, color=fit_color, linewidth=1.5, label=fit_label, zorder=2)

        if show_band:
            cov = reg.get("cov")
            if cov is None:
                raise ValueError(
                    "plot_nonlinear_regression : show_band=True nécessite "
                    "reg['cov'] (matrice de covariance retournée par "
                    "nonlinear_regression)."
                )
            J = _numeric_jacobian(model_func, x_grid, params)
            var_y = np.einsum("ij,jk,ik->i", J, cov, J)
            u_y_grid = np.sqrt(np.clip(var_y, 0.0, None))
            lower = y_grid - k_band * u_y_grid
            upper = y_grid + k_band * u_y_grid
            band_label = r"$\pm u(\hat y)$" if k_band == 1.0 else rf"$\pm {k_band:g}\,u(\hat y)$"
            ax.fill_between(x_grid, lower, upper, color=fit_color, alpha=0.15, zorder=1, label=band_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if xscale != "linear":
        ax.set_xscale(xscale)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_residuals(
    x_data: list,
    reg: dict,
    u_y=None,
    x_label: str = "x",
    y_label: str = r"Résidu $y_i - \hat{y}_i$",
    title: str = "",
    data_color: str = "tab:blue",
    zero_color: str = "tab:red",
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple = (7, 3.5),
    ax=None,
):
    """Résidus de la régression (reg["residuals"]) en fonction de x.
    Nécessite reg : les résidus n'existent que si une régression a
    effectivement été calculée (cf. show_fit de plot_regression)."""
    _validate_data(x_data=x_data, residuals=reg["residuals"], u_y=u_y)
    fig, ax = _new_ax(ax, figsize)
    ax.errorbar(
        x_data, reg["residuals"], yerr=u_y,
        fmt="o", color=data_color, ecolor=data_color,
        elinewidth=1, capsize=3, markersize=5, zorder=3,
    )
    ax.axhline(0, color=zero_color, linewidth=1, linestyle="--", zorder=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if xscale != "linear":
        ax.set_xscale(xscale)
    if yscale != "linear":
        ax.set_yscale(yscale)
    fig.tight_layout()
    return fig, ax


# Palette par défaut pour les séries qui ne précisent pas de couleur —
# reprend l'ordre du cycle tab10 de matplotlib, bouclée via modulo si plus
# de 10 séries (cf. plot_multi_series).
DEFAULT_SERIES_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_multi_series(
    x_data: list,
    series: list,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
    marker: str = "o",
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple = (7, 5),
    ax=None,
):
    """Plusieurs courbes sur les MÊMES axes, à la même échelle — remplace
    add_twin_series pour le cas réel du TP : comparer l'évolution d'une
    grandeur selon plusieurs conditions (a, b, c, d...) partageant le même
    x, pas deux grandeurs de nature différente sur deux échelles.

    x_data : abscisse commune à toutes les séries.

    series : liste de dicts, un par courbe, avec les clés :
        - "y"      : liste des ordonnées (obligatoire, même longueur que x_data)
        - "label"  : légende de la courbe (obligatoire)
        - "u_y"    : incertitude-type sur y, float (constante) ou liste
                     (une valeur par point) — optionnel, None par défaut
                     (pas de barre d'erreur)
        - "color"  : couleur de la courbe — optionnel ; si absente, prise
                     dans DEFAULT_SERIES_PALETTE (bouclée par modulo selon
                     la position de la série dans la liste)
        - "connect": bool — relie les points par une ligne si True. Par
                     défaut False (nuage de points seul), qui reste le
                     rendu le plus lisible avec plusieurs courbes
                     superposées ; à activer courbe par courbe si une
                     tendance continue est plus parlante qu'un nuage.

    Chaque série est validée indépendamment via _validate_data (longueur
    vs x_data, valeurs finies), pour qu'une série mal renseignée ne fasse
    pas échouer le tracé des autres avec un message ambigu.
    """
    fig, ax = _new_ax(ax, figsize)

    for i, s in enumerate(series):
        y = s["y"]
        label = s["label"]
        u_y = s.get("u_y")
        color = s.get("color") or DEFAULT_SERIES_PALETTE[i % len(DEFAULT_SERIES_PALETTE)]
        connect = s.get("connect", False)

        _validate_data(x_data=x_data, y=y, u_y=u_y)

        linestyle = "-" if connect else "None"
        ax.errorbar(
            x_data, y, yerr=u_y,
            fmt=marker, linestyle=linestyle, color=color, ecolor=color,
            elinewidth=1, capsize=3, markersize=5,
            label=label, zorder=3,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if xscale != "linear":
        ax.set_xscale(xscale)
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def annotate_point(
    ax,
    x: float,
    y: float,
    text: str,
    offset: tuple = (10, 10),
    color: str = "black",
    arrow: bool = True,
):
    """Annote un point remarquable (valeur exclue de la régression, point
    aberrant, valeur particulière) sur un axe existant.

    offset : décalage du texte par rapport au point, en points d'écran
    (pas en unités de données) — l'annotation ne dépend pas de l'échelle
    du graphe (utile en particulier avec xscale/yscale="log")."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=color, lw=1) if arrow else None,
    )
    return ax