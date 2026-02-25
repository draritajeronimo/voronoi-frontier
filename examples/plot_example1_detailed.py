"""
examples/plot_example1_detailed.py
===================================
Figura detalhada e didatica do Exemplo 1 - versao com anotacoes aprimoradas.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from voronoi_frontier import compute_frontier
from voronoi_frontier.core import M_functions

# =============================================================================
# DADOS
# =============================================================================
a, b, c = 2.0, -3.0, 0.5
xs, ys, key_pts, info = compute_frontier(a, b, c, n_points=200)

Q2 = key_pts["Q2"]
Q5 = key_pts["Q5"]
D  = key_pts["D"]
A  = np.array([0.0, a])
B  = np.array([0.0, b])
C  = np.array([c,   0.0])

# Ponto de demonstracao E: 1/3 da fronteira
idx  = len(xs) // 3
E    = np.array([xs[idx], ys[idx]])
M    = M_functions(E[0], E[1], c)
Q3   = np.array([c + M[6]/M[7], M[9]/M[8]])

# =============================================================================
# FIGURA
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 11))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

# --- Obstaculo ---
circulo = patches.Circle(
    (c, 0), 1.0,
    fill=True, facecolor="#dbeafe", edgecolor="#2563eb",
    linewidth=2.0, zorder=2
)
ax.add_patch(circulo)

# --- Bissetriz ---
ax.axhline((a+b)/2, linestyle="--", color="#9ca3af", lw=1.2, zorder=1,
           label=f"Bissetriz sem obstaculo  y = {(a+b)/2:.1f}")

# --- Linhas de tangencia (tracejadas finas) ---
ax.plot([A[0], Q2[0]], [A[1], Q2[1]], "--", color="#6d28d9", lw=0.9, alpha=0.4, zorder=3)
ax.plot([B[0], Q5[0]], [B[1], Q5[1]], "--", color="#0891b2", lw=0.9, alpha=0.4, zorder=3)

# --- Geodesica E -> Q3 ---
ax.annotate("", xy=Q3, xytext=E,
    arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.8))

# --- Arco Q3 -> Q2 ---
theta3 = np.arctan2(Q3[1], Q3[0] - c)
theta2 = np.arctan2(Q2[1], Q2[0] - c)
thetas = np.linspace(theta3, theta2, 80)
ax.plot(c + np.cos(thetas), np.sin(thetas), "-", color="#7c3aed", lw=2.0, zorder=5)
ax.annotate("", xy=Q2, xytext=(c + np.cos(thetas[-2]), np.sin(thetas[-2])),
    arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.8))

# --- Geodesica Q2 -> A ---
ax.annotate("", xy=A, xytext=Q2,
    arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.8))

# --- Geodesica E -> B (reta direta, Situacao 1) ---
ax.annotate("", xy=B, xytext=E,
    arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.8))

# --- Rotulos das geodesicas (longe dos pontos, com seta fina) ---
mid_geo_A = (E + Q3) / 2
ax.annotate(
    "d(A, E)\nvia geodesica\n(Q2→Q3→E)",
    xy=mid_geo_A,
    xytext=(mid_geo_A[0] - 1.2, mid_geo_A[1] + 0.5),
    fontsize=8, color="#7c3aed",
    arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#7c3aed", alpha=0.9)
)

mid_EB = (E + B) / 2
ax.annotate(
    "d(E, B)\ndireto\n(Situacao 1)",
    xy=mid_EB,
    xytext=(mid_EB[0] + 0.5, mid_EB[1] - 0.3),
    fontsize=8, color="#059669",
    arrowprops=dict(arrowstyle="->", color="#059669", lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#059669", alpha=0.9)
)

# --- Fronteira ---
ax.plot(xs, ys, "-", color="#dc2626", lw=2.5,
        label="Fronteira de Voronoi (41 pontos)", zorder=6)
ax.scatter(xs[0],  ys[0],  color="#dc2626", s=40, zorder=7)
ax.scatter(xs[-1], ys[-1], color="black",   s=40, zorder=7, marker="x")

# =============================================================================
# PONTOS COM ANOTACOES BEM ESPAÇADAS
# Cada anotacao tem: posicao do texto (xytext) bem afastada do ponto,
# seta fina apontando para o ponto exato.
# =============================================================================

def anotar(ax, pt, texto, cor, xytext, marker="o", size=70):
    ax.scatter(*pt, color=cor, marker=marker, s=size, zorder=10,
               edgecolors="white", linewidths=1.0)
    ax.annotate(
        texto, xy=pt, xytext=xytext,
        fontsize=8.5, color=cor, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=cor, lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=cor, alpha=0.95),
        zorder=11
    )

# A -- acima e a esquerda
anotar(ax, A,
       "A = (0, 2)\nPonto gerador",
       "#dc2626", (-1.1, 2.3))

# B -- abaixo e a esquerda
anotar(ax, B,
       "B = (0, -3)\nPonto gerador",
       "#ea580c", (-1.1, -3.4))

# C -- a direita, dentro do circulo
anotar(ax, C,
       "C = (0.5, 0)\nCentro do\nobstaculo",
       "#2563eb", (0.9, 0.6), marker="o", size=50)

# Q2 -- a esquerda, acima
anotar(ax, Q2,
       f"Q2 = ({Q2[0]:.2f}, {Q2[1]:.2f})\nTangencia de A",
       "#6d28d9", (-1.5, 0.8))

# Q3 -- a esquerda, abaixo de Q2
anotar(ax, Q3,
       f"Q3 = ({Q3[0]:.2f}, {Q3[1]:.2f})\nTangencia de E",
       "#7c3aed", (-1.6, -0.3))

# Q5 -- a esquerda, abaixo
anotar(ax, Q5,
       f"Q5 = ({Q5[0]:.2f}, {Q5[1]:.2f})\nTangencia de B",
       "#0891b2", (-1.5, -0.8))

# D -- a esquerda
anotar(ax, D,
       f"D = ({D[0]:.2f}, {D[1]:.2f})\nInicio da fronteira\n(ultimo ponto bissetriz)",
       "#16a34a", (-1.7, -1.3))

# E -- a direita
anotar(ax, E,
       f"E = ({E[0]:.2f}, {E[1]:.2f})\nPonto da fronteira\n(demonstracao)",
       "#7c3aed", (0.3, -0.1))

# =============================================================================
# CAIXA DE INFORMACOES
# =============================================================================
ax.text(
    0.015, 0.015,
    "Equidistancia:  d(A, E) = d(E, B)  para todo E\n"
    "Newton-Raphson: max 2 iteracoes por ponto\n"
    "41 pontos gerados  |  Situacao 1 em todos os pontos",
    transform=ax.transAxes,
    fontsize=8, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff",
              edgecolor="#2563eb", alpha=0.95)
)

# --- Rotulo do obstaculo ---
ax.text(c, -0.55, "Obstaculo\n(r = 1)", fontsize=8, color="#2563eb",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#2563eb", alpha=0.8))

# --- Eixos ---
margin = 0.4
all_x  = np.concatenate([xs, [-1.8, 1.0]])
all_y  = np.concatenate([ys, [2.5, -3.6]])
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_aspect("equal")
ax.axhline(0, color="#6b7280", lw=0.6, zorder=1)
ax.axvline(0, color="#6b7280", lw=0.6, zorder=1)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(
    "Fronteira de Voronoi — Dois Sites e Obstáculo Circular\n"
    "A = (0, 2),   B = (0, -3),   C = (0.5, 0),   r = 1",
    fontsize=12, pad=14
)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
ax.grid(True, linestyle=":", alpha=0.3, color="#9ca3af")

plt.tight_layout()
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontier_detailed.png")
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"Figura salva em: {OUTPUT}")
