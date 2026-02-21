"""
voronoi_frontier/plot.py
========================
Funcoes de visualizacao para os resultados da fronteira de Voronoi.

Todas as figuras sao geradas em qualidade de publicacao (300 dpi)
e podem ser salvas diretamente em PNG para uso no artigo.

Funcoes disponiveis
-------------------
plot_frontier         -- figura principal com a fronteira e pontos auxiliares
plot_convergence      -- grafico de convergencia do Newton-Raphson
plot_multiple_configs -- comparacao entre multiplas configuracoes (a, b, c)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from typing import Optional


# =============================================================================
# FUNCOES AUXILIARES INTERNAS
# =============================================================================

def _draw_circle(ax: plt.Axes, c: float, r: float = 1.0,
                 color: str = "steelblue", lw: float = 1.5,
                 label: str = "Obstaculo circular (r=1)") -> None:
    """Desenha o obstaculo circular no eixo fornecido."""
    circle = patches.Circle(
        (c, 0), r,
        fill=False, edgecolor=color, linewidth=lw,
        label=label, zorder=2
    )
    ax.add_patch(circle)


def _scatter_point(ax: plt.Axes, xy: np.ndarray, label: str,
                   color: str, marker: str = "o",
                   size: int = 60, zorder: int = 5) -> None:
    """Plota um ponto com rotulo no eixo fornecido."""
    ax.scatter(*xy, color=color, marker=marker, s=size, zorder=zorder)
    ax.annotate(
        label, xy,
        textcoords="offset points", xytext=(6, 4),
        fontsize=9, color=color
    )


# =============================================================================
# FIGURA PRINCIPAL DA FRONTEIRA
# =============================================================================

def plot_frontier(xs: np.ndarray, ys: np.ndarray,
                  a: float, b: float, c: float,
                  key_points: dict,
                  title: str = "Fronteira de Voronoi - Dois Sites e Obstaculo Circular",
                  figsize: tuple = (7, 7),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Gera a figura principal com a fronteira de Voronoi calculada.

    Exibe os pontos geradores A e B, o obstaculo circular, os pontos
    auxiliares Q2, Q5 e D, a bissetriz de referencia e a fronteira calculada.

    Parametros
    ----------
    xs, ys      : np.ndarray  -- coordenadas dos pontos da fronteira.
    a, b, c     : float       -- parametros de configuracao.
    key_points  : dict        -- pontos auxiliares retornados por compute_frontier.
    title       : str         -- titulo da figura.
    figsize     : tuple       -- tamanho da figura em polegadas (largura, altura).
    save_path   : str ou None -- se fornecido, salva a figura neste caminho (PNG, 300 dpi).

    Retorna
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Obstaculo circular
    _draw_circle(ax, c, color="steelblue", label="Obstaculo circular (r=1)")

    # Pontos geradores
    _scatter_point(ax, np.array([0.0, a]), "A = (0, a)", "crimson")
    _scatter_point(ax, np.array([0.0, b]), "B = (0, b)", "darkorange")

    # Centro do circulo
    ax.scatter(c, 0, color="steelblue", marker="+", s=80, zorder=5)
    ax.annotate("C = (c, 0)", (c, 0),
                textcoords="offset points", xytext=(4, -12),
                fontsize=8, color="steelblue")

    # Pontos auxiliares principais
    for name, color in [("Q2", "purple"), ("Q5", "darkcyan"), ("D", "forestgreen")]:
        pt = key_points.get(name)
        if pt is not None:
            _scatter_point(ax, pt, name, color, marker="s", size=45)

    # Bissetriz de referencia (sem obstaculo)
    bisector_y = (a + b) / 2.0
    ax.axhline(bisector_y, linestyle="--", color="gray", lw=0.8,
               label=f"Bissetriz  y = {bisector_y:.2f}")

    # Fronteira calculada
    ax.plot(xs, ys, "-", color="darkred", lw=2.0,
            label="Fronteira de Voronoi", zorder=6)
    ax.scatter(xs[0],  ys[0],  color="darkred", s=50, zorder=7)
    ax.scatter(xs[-1], ys[-1], color="black",   s=50, zorder=7,
               marker="x", label="Fim da fronteira")

    # Ajuste dos eixos
    margin  = 0.4
    all_x   = np.concatenate([xs, [0.0, 0.0, c]])
    all_y   = np.concatenate([ys, [a,   b,   0.0]])
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    return fig


# =============================================================================
# FIGURA DE CONVERGENCIA DO NEWTON-RAPHSON
# =============================================================================

def plot_convergence(info: dict,
                     figsize: tuple = (10, 4),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Gera dois graficos de convergencia do metodo de Newton-Raphson.

    Grafico esquerdo : numero de iteracoes por ponto da fronteira.
    Grafico direito  : linha do tempo das Situacoes 1 e 2.

    Este grafico e importante para demonstrar a convergencia rapida
    do algoritmo, conforme afirmado na Secao 6.5 do artigo.

    Parametros
    ----------
    info      : dict    -- dicionario retornado por compute_frontier,
                           com chaves 'iter_counts' e 'situation_flags'.
    figsize   : tuple   -- tamanho da figura.
    save_path : str     -- caminho para salvar a figura (opcional).

    Retorna
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Grafico de iteracoes ---
    iters = info["iter_counts"]
    axes[0].bar(range(len(iters)), iters,
                color="steelblue", edgecolor="white", width=0.8)
    axes[0].set_xlabel("Indice do ponto da fronteira", fontsize=10)
    axes[0].set_ylabel("Iteracoes do Newton-Raphson", fontsize=10)
    axes[0].set_title("Convergencia: iteracoes por ponto", fontsize=11)
    axes[0].set_ylim(bottom=0)

    # --- Linha do tempo das situacoes ---
    flags      = info["situation_flags"]
    color_map  = {1: "steelblue", 2: "darkorange"}
    bar_colors = [color_map[f] for f in flags]
    axes[1].bar(range(len(flags)), [1] * len(flags),
                color=bar_colors, width=1.0, edgecolor="none")

    legend_handles = [
        Line2D([0], [0], color="steelblue",   lw=8,
               label="Situacao 1: visada direta a B"),
        Line2D([0], [0], color="darkorange",  lw=8,
               label="Situacao 2: obstaculo bloqueia B"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)
    axes[1].set_xlabel("Indice do ponto da fronteira", fontsize=10)
    axes[1].set_title("Transicao entre situacoes", fontsize=11)
    axes[1].set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura de convergencia salva em: {save_path}")

    return fig


# =============================================================================
# FIGURA COM MULTIPLAS CONFIGURACOES
# =============================================================================

def plot_multiple_configs(configs: list,
                          figsize: tuple = (14, 5),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plota varias configuracoes (a, b, c) lado a lado para comparacao.

    Reproduz o estilo da Figura 1 do artigo, mostrando como a fronteira
    varia conforme os parametros sao alterados.

    Parametros
    ----------
    configs   : list de dicts, cada um com as chaves:
                  'a', 'b', 'c'        -- parametros de configuracao
                  'xs', 'ys'           -- pontos da fronteira calculados
                  'key_points'         -- pontos auxiliares
                  'label' (opcional)   -- titulo do subgrafico
    figsize   : tuple   -- tamanho total da figura.
    save_path : str     -- caminho para salvar a figura (opcional).

    Retorna
    -------
    fig : matplotlib.figure.Figure
    """
    n   = len(configs)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for idx, cfg in enumerate(configs):
        ax      = axes[0][idx]
        xs, ys  = cfg["xs"], cfg["ys"]
        a, b, c = cfg["a"], cfg["b"], cfg["c"]
        kp      = cfg.get("key_points", {})

        # Obstaculo
        circle = patches.Circle(
            (c, 0), 1.0,
            fill=False, edgecolor="steelblue", linewidth=1.5
        )
        ax.add_patch(circle)

        # Pontos geradores
        ax.scatter(0, a, color="crimson",    s=50, zorder=5)
        ax.annotate("A", (0, a), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color="crimson")
        ax.scatter(0, b, color="darkorange", s=50, zorder=5)
        ax.annotate("B", (0, b), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color="darkorange")

        # Fronteira
        ax.plot(xs, ys, "-", color="darkred", lw=2.0)

        # Ponto D
        D = kp.get("D")
        if D is not None:
            ax.scatter(*D, color="forestgreen", s=45, marker="s", zorder=6)
            ax.annotate("D", D, xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color="forestgreen")

        # Eixos e grade
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_title(cfg.get("label", f"a={a}, b={b}, c={c}"), fontsize=9)

        margin = 0.5
        all_x  = np.concatenate([xs, [0.0, c]])
        all_y  = np.concatenate([ys, [a,   b]])
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    fig.suptitle(
        "Fronteira de Voronoi - Multiplas Configuracoes",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figura de multiplas configuracoes salva em: {save_path}")

    return fig
