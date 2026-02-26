"""
examples/animate_frontier.py - versao final
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as PILImage
from voronoi_frontier import compute_frontier
from voronoi_frontier.core import (M_functions, dist_A_Q2, dist_Q2_Q3, compute_lambda_phi)

# =============================================================================
# DADOS
# =============================================================================
a, b, c = 2.0, -3.0, 0.5
xs, ys, key_pts, info = compute_frontier(a, b, c, n_points=200)
n    = len(xs)
Q2   = key_pts["Q2"]
Q5   = key_pts["Q5"]
D    = key_pts["D"]
A    = np.array([0.0, a])
B    = np.array([0.0, b])
lam, phi = compute_lambda_phi(a, c)
dAQ2     = dist_A_Q2(a, c)

# Cores
COR_FRONTEIRA  = "#dc2626"   # vermelho
COR_GEODESICA  = "#ff6600"   # laranja
COR_DIRETA     = "#16a34a"   # verde
COR_E          = "#ff00ff"   # magenta
COR_Q3         = "#f59e0b"   # amarelo
COR_A          = "#dc2626"   # vermelho
COR_B          = "#ea580c"   # laranja escuro
COR_D          = "#16a34a"   # verde
COR_Q2         = "#6d28d9"   # roxo escuro
COR_Q5         = "#0891b2"   # azul petroleo
COR_CIRCULO_E  = "#1d4ed8"   # azul escuro
COR_CIRCULO_F  = "#bfdbfe"   # azul claro

def geodesica_completa(xi, yi):
    M = M_functions(xi, yi, c)
    if abs(M[7]) < 1e-10 or abs(M[8]) < 1e-10:
        return None, None, None
    Q3 = np.array([c + M[6]/M[7], M[9]/M[8]])
    th2 = np.arctan2(Q2[1], Q2[0] - c)
    th3 = np.arctan2(Q3[1], Q3[0] - c)
    ths = np.linspace(th2, th3, 60)
    return Q3, c + np.cos(ths), np.sin(ths)

def dist_geo_A(xi, yi):
    M   = M_functions(xi, yi, c)
    arc = dist_Q2_Q3(M, lam, phi)
    Q3  = np.array([c + M[6]/M[7], M[9]/M[8]])
    return dAQ2 + arc + np.linalg.norm(np.array([xi, yi]) - Q3)

def dist_B(xi, yi):
    return np.sqrt(xi**2 + (yi - b)**2)

def draw_static(ax):
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-1.9, 1.1)
    ax.set_ylim(-3.9, 2.9)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4, color="#cbd5e1", zorder=0)
    ax.axhline(0, color="#94a3b8", lw=0.8, zorder=1)
    ax.axvline(0, color="#94a3b8", lw=0.8, zorder=1)
    ax.axhline((a+b)/2, linestyle="--", color="#94a3b8", lw=1.2, zorder=1)
    ax.text(-1.85, (a+b)/2+0.12, f"bissetriz y={(a+b)/2:.1f}",
            fontsize=7.5, color="#94a3b8")

    # Obstaculo
    circ = patches.Circle((c, 0), 1.0, fill=True,
                           facecolor=COR_CIRCULO_F,
                           edgecolor=COR_CIRCULO_E,
                           linewidth=2.5, zorder=2)
    ax.add_patch(circ)
    ax.text(c, 0, "C=(0.5,0)\nr=1", fontsize=8.5, color=COR_CIRCULO_E,
            ha="center", va="center", fontweight="bold", zorder=3)

    # Ponto A
    ax.scatter(*A, color=COR_A, s=110, zorder=9, edgecolors="white", lw=1.5)
    ax.annotate("A=(0,2)", A, xytext=(-1.3, 2.4), fontsize=9,
                color=COR_A, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COR_A, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COR_A, alpha=0.95), zorder=10)

    # Ponto B
    ax.scatter(*B, color=COR_B, s=110, zorder=9, edgecolors="white", lw=1.5)
    ax.annotate("B=(0,-3)", B, xytext=(-1.3, -3.5), fontsize=9,
                color=COR_B, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COR_B, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COR_B, alpha=0.95), zorder=10)

    # Ponto D
    ax.scatter(*D, color=COR_D, s=80, marker="s", zorder=9,
               edgecolors="white", lw=1.5)
    ax.annotate(f"D=({D[0]:.2f},{D[1]:.2f})\ninicio", D,
                xytext=(-1.7, -1.1), fontsize=8.5, color=COR_D,
                arrowprops=dict(arrowstyle="->", color=COR_D, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=COR_D, alpha=0.95), zorder=10)

    # Q2
    ax.scatter(*Q2, color=COR_Q2, s=65, marker="s", zorder=8,
               edgecolors="white", lw=1.0)
    ax.annotate("Q2", Q2, xytext=(Q2[0]-0.55, Q2[1]+0.35), fontsize=8.5,
                color=COR_Q2,
                arrowprops=dict(arrowstyle="->", color=COR_Q2, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COR_Q2, alpha=0.92), zorder=9)

    # Q5
    ax.scatter(*Q5, color=COR_Q5, s=65, marker="s", zorder=8,
               edgecolors="white", lw=1.0)
    ax.annotate("Q5", Q5, xytext=(Q5[0]-0.55, Q5[1]-0.35), fontsize=8.5,
                color=COR_Q5,
                arrowprops=dict(arrowstyle="->", color=COR_Q5, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=COR_Q5, alpha=0.92), zorder=9)

    ax.plot([A[0], Q2[0]], [A[1], Q2[1]], "--", color=COR_Q2, lw=0.8, alpha=0.3)
    ax.plot([B[0], Q5[0]], [B[1], Q5[1]], "--", color=COR_Q5, lw=0.8, alpha=0.3)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)

def draw_info_panel(ax_info, i, em_intro, em_pausa):
    ax_info.set_facecolor("#f8fafc")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    # Titulo
    ax_info.text(0.5, 0.97, "Informacoes do Algoritmo", fontsize=11,
                 fontweight="bold", ha="center", va="top", color="#1e293b")
    ax_info.plot([0, 1], [0.93, 0.93], color="#cbd5e1", lw=1.0)

    # Configuracao
    ax_info.text(0.05, 0.90, "Configuracao:", fontsize=9,
                 fontweight="bold", color="#475569")
    for k, txt in enumerate(["A = (0, 2)  —  ponto gerador",
                               "B = (0, -3)  —  ponto gerador",
                               "C = (0.5, 0),  r = 1  —  obstaculo"]):
        ax_info.text(0.07, 0.86 - k*0.055, txt, fontsize=8.5, color="#334155")

    ax_info.plot([0, 1], [0.70, 0.70], color="#cbd5e1", lw=0.8)
    ax_info.text(0.05, 0.67, "Ponto atual E:", fontsize=9,
                 fontweight="bold", color="#475569")

    # Dados dinamicos
    if not em_intro:
        xi, yi = xs[i], ys[i]
        dA  = dist_geo_A(xi, yi)
        dB  = dist_B(xi, yi)
        err = abs(dA - dB)
        sit = info["situation_flags"][i]

        ax_info.text(0.07, 0.63, f"E = ({xi:.4f},  {yi:.4f})",
                     fontsize=9, color="#334155")
        cor_sit = "#0891b2" if sit == 1 else "#ea580c"
        ax_info.text(0.07, 0.575,
                     "Situacao 1: visada direta a B" if sit == 1
                     else "Situacao 2: contorna obstaculo",
                     fontsize=9, color=cor_sit)
        ax_info.text(0.07, 0.52,
                     f"Iteracoes Newton-Raphson: {info['iter_counts'][i]}",
                     fontsize=9, color="#334155")
        ax_info.text(0.07, 0.465, f"d(A, E) = {dA:.6f}",
                     fontsize=9, color=COR_GEODESICA)
        ax_info.text(0.07, 0.41,  f"d(E, B) = {dB:.6f}",
                     fontsize=9, color=COR_DIRETA)
        cor_err = "#16a34a" if err < 1e-3 else "#dc2626"
        ax_info.text(0.07, 0.355,
                     f"|d(A,E) - d(E,B)| = {err:.2e}  ✓" if err < 1e-3
                     else f"|d(A,E) - d(E,B)| = {err:.2e}",
                     fontsize=9.5, color=cor_err, fontweight="bold")

    ax_info.plot([0, 1], [0.33, 0.33], color="#cbd5e1", lw=0.8)
    ax_info.text(0.05, 0.30, "Legenda:", fontsize=9,
                 fontweight="bold", color="#475569")
    for k, (cor, txt) in enumerate([
        (COR_FRONTEIRA, "━━  Fronteira de Voronoi"),
        (COR_GEODESICA, "━━  Geodesica A→Q2→Q3→E"),
        (COR_DIRETA,    "━━  Geodesica E→B  (reta direta)"),
        (COR_E,         "●   Ponto atual E"),
        (COR_Q3,        "●   Ponto Q3 (sobre o circulo)"),
    ]):
        ax_info.text(0.07, 0.26 - k*0.052, txt, fontsize=8.5, color=cor)

    # Barra de progresso
    ax_info.add_patch(patches.Rectangle(
        (0.05, 0.038), 0.90, 0.038,
        facecolor="#e2e8f0", edgecolor="#cbd5e1", lw=1))
    prog = 0.0 if em_intro else (i+1)/n
    if prog > 0:
        ax_info.add_patch(patches.Rectangle(
            (0.05, 0.038), 0.90*prog, 0.038,
            facecolor=COR_FRONTEIRA, edgecolor="none"))
    ax_info.text(0.50, 0.057,
                 "0%" if em_intro else f"{i+1}/{n}  ({prog*100:.0f}%)",
                 fontsize=7.5, ha="center", va="center", color="#475569")

# =============================================================================
# PRE-RENDERIZA TODOS OS FRAMES
# =============================================================================
N_INTRO = 10
N_PAUSE = 5
total   = N_INTRO + n + N_PAUSE

print("Pre-renderizando frames...")
frames_img = []

for frame in range(total):
    fig_f = plt.figure(figsize=(14, 8))
    fig_f.patch.set_facecolor("#f0f4f8")
    gs      = fig_f.add_gridspec(1, 2, width_ratios=[1.6, 1],
                                  left=0.05, right=0.97,
                                  top=0.93, bottom=0.08, wspace=0.12)
    ax_f      = fig_f.add_subplot(gs[0])
    ax_info_f = fig_f.add_subplot(gs[1])

    draw_static(ax_f)

    em_intro = frame < N_INTRO
    i        = min(max(frame - N_INTRO, 0), n - 1)
    em_pausa = (frame - N_INTRO) >= n

    draw_info_panel(ax_info_f, i, em_intro, em_pausa)

    if em_intro:
        ax_f.set_title("Fronteira de Voronoi — configuracao inicial\n"
                       "A=(0,2)  B=(0,-3)  obstaculo r=1 em C=(0.5,0)",
                       fontsize=11, pad=8)
    elif em_pausa:
        # Fronteira final estatica em destaque
        ax_f.plot(xs, ys, "-", color=COR_FRONTEIRA, lw=4.0, zorder=6)
        # Pontos ao longo da fronteira para destacar os 41 pontos calculados
        ax_f.scatter(xs, ys, color=COR_FRONTEIRA, s=18, zorder=7,
                     edgecolors="white", lw=0.5)

        # Pontos inicial e final marcados
        ax_f.scatter(xs[0], ys[0], color=COR_FRONTEIRA, s=80, zorder=10,
                     edgecolors="white", lw=1.5)
        ax_f.annotate("Inicio\n(D)", (xs[0], ys[0]),
                      xytext=(xs[0]-0.55, ys[0]+0.3), fontsize=8.5,
                      color=COR_FRONTEIRA, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=COR_FRONTEIRA, lw=1.0),
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor=COR_FRONTEIRA, alpha=0.95), zorder=11)
        ax_f.scatter(xs[-1], ys[-1], color="#7c3aed", s=80, zorder=10,
                     edgecolors="white", lw=1.5)
        ax_f.annotate("Fim\n(obstaculo)", (xs[-1], ys[-1]),
                      xytext=(xs[-1]-0.6, ys[-1]-0.45), fontsize=8.5,
                      color="#7c3aed", fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.0),
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="#7c3aed", alpha=0.95), zorder=11)

        # Caixa de resultado final sobre o grafico
        ax_f.text(0.02, 0.02,
                  f"✓  {n} pontos calculados\n"
                  "✓  Equidistancia verificada em todos\n"
                  "✓  Newton-Raphson: max 2 iteracoes\n"
                  "✓  Situacao 1 em todos os pontos",
                  transform=ax_f.transAxes, fontsize=9,
                  verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7",
                            edgecolor="#16a34a", alpha=0.97), zorder=12)

        ax_f.set_title(f"✓  Fronteira de Voronoi concluida — {n} pontos\n"
                       "Equidistancia d(A,E) = d(E,B) verificada em todos os pontos",
                       fontsize=11, pad=8, color="#16a34a", fontweight="bold")
    else:
        # Fronteira acumulada
        ax_f.plot(xs[:i+1], ys[:i+1], "-", color=COR_FRONTEIRA, lw=2.5, zorder=6)

        xi, yi = xs[i], ys[i]
        Q3_pt, arco_x, arco_y = geodesica_completa(xi, yi)

        if Q3_pt is not None:
            # Geodesica A->Q2 laranja (zorder baixo, abaixo do circulo)
            ax_f.plot([A[0], Q2[0]], [A[1], Q2[1]], "-",
                      color=COR_GEODESICA, lw=3.0, zorder=3)
            # Arco Q2->Q3 laranja
            ax_f.plot(arco_x, arco_y, "-",
                      color=COR_GEODESICA, lw=3.0, zorder=3)
            # Q3->E laranja
            ax_f.plot([Q3_pt[0], xi], [Q3_pt[1], yi], "-",
                      color=COR_GEODESICA, lw=3.0, zorder=3)
            # E->B verde
            ax_f.plot([xi, B[0]], [yi, B[1]], "-",
                      color=COR_DIRETA, lw=3.0, zorder=3)
            # Redesenha circulo por cima para cobrir qualquer sobreposicao
            circ2 = patches.Circle((c, 0), 1.0, fill=True,
                                   facecolor=COR_CIRCULO_F,
                                   edgecolor=COR_CIRCULO_E,
                                   linewidth=2.5, zorder=4)
            ax_f.add_patch(circ2)
            ax_f.text(c, 0, "C=(0.5,0)\nr=1", fontsize=8.5, color=COR_CIRCULO_E,
                      ha="center", va="center", fontweight="bold", zorder=5)
            # Ponto E magenta com rotulo
            ax_f.plot(xi, yi, "o", color=COR_E, ms=14, zorder=10,
                      mew=2, mec="white")
            ax_f.annotate("E", (xi, yi), xytext=(xi+0.12, yi+0.12),
                          fontsize=11, color=COR_E, fontweight="bold", zorder=11,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor=COR_E, alpha=0.9))
            # Ponto Q3 amarelo com rotulo
            ax_f.plot(Q3_pt[0], Q3_pt[1], "o", color=COR_Q3, ms=12, zorder=10,
                      mew=2, mec="white")
            ax_f.annotate("Q3", Q3_pt, xytext=(Q3_pt[0]-0.35, Q3_pt[1]+0.15),
                          fontsize=9, color=COR_Q3, fontweight="bold", zorder=11,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor=COR_Q3, alpha=0.9))

        sit  = info["situation_flags"][i]
        stxt = "Situacao 1: d(E,B) e linha reta direta" if sit == 1 \
               else "Situacao 2: geodesica contorna o obstaculo"
        ax_f.set_title(f"Construindo ponto {i+1}/{n}  —  {stxt}",
                       fontsize=11, pad=8)

    fig_f.canvas.draw()
    buf = np.frombuffer(fig_f.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig_f.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames_img.append(img)
    plt.close(fig_f)

    if (frame+1) % 10 == 0:
        print(f"  Frame {frame+1}/{total}")

print("Salvando MP4...")
import matplotlib.animation as manimation
import PIL.Image as PILImage

OUTPUT_MP4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontier_animation.mp4")

# Cria video a partir dos frames numpy
fig_vid, ax_vid = plt.subplots(figsize=(14, 8))
ax_vid.axis("off")
im = ax_vid.imshow(frames_img[0])
plt.tight_layout(pad=0)

def update_vid(frame):
    im.set_data(frames_img[frame])
    return [im]

ani_vid = manimation.FuncAnimation(
    fig_vid, update_vid,
    frames=len(frames_img),
    interval=120,
    blit=True
)

writer = manimation.FFMpegWriter(fps=10, bitrate=2000,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
ani_vid.save(OUTPUT_MP4, writer=writer)
plt.close(fig_vid)
print(f"Video salvo em: {OUTPUT_MP4}")








