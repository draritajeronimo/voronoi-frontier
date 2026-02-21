"""
examples/run_example.py
=======================
Exemplo completo de uso do pacote voronoi_frontier.

Reproduce todas as figuras do artigo e exibe informacoes
diagnosticas sobre a convergencia do algoritmo.

Execute a partir da raiz do repositorio:
    python examples/run_example.py

Figuras geradas
---------------
examples/frontier_single.png   -- fronteira para a configuracao padrao
examples/convergence.png       -- convergencia do Newton-Raphson
examples/frontier_multiple.png -- comparacao entre multiplas configuracoes
"""

import sys
import os

# Garante que o pacote e encontrado independente de onde o script e executado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # modo sem interface grafica para salvar figuras

from voronoi_frontier import (
    compute_frontier,
    plot_frontier,
    plot_convergence,
    plot_multiple_configs,
)

# Garante que a pasta examples existe para salvar as figuras
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# EXEMPLO 1 -- Configuracao padrao do artigo
# =============================================================================

print("=" * 60)
print("Exemplo 1: Configuracao padrao (a=2.0, b=-3.0, c=0.5)")
print("=" * 60)

a, b, c = 2.0, -3.0, 0.5

xs, ys, key_pts, info = compute_frontier(
    a=a, b=b, c=c,
    n_points=200,
    max_iter=50,
    tol=1e-10,
)

print(f"  Pontos da fronteira gerados : {len(xs)}")
print(f"  Pontos em Situacao 1        : {info['situation_flags'].count(1)}")
print(f"  Pontos em Situacao 2        : {info['situation_flags'].count(2)}")
print(f"  Max. iteracoes Newton-Raphson: {max(info['iter_counts'])}")
print(f"  Pontos auxiliares calculados:")
for nome, pt in key_pts.items():
    print(f"    {nome} = ({pt[0]:.5f}, {pt[1]:.5f})")

# Figura principal da fronteira
fig1 = plot_frontier(
    xs, ys,
    a=a, b=b, c=c,
    key_points=key_pts,
    title=f"Fronteira de Voronoi  (a={a}, b={b}, c={c})",
    save_path=os.path.join(OUTPUT_DIR, "frontier_single.png"),
)

# Figura de convergencia
fig2 = plot_convergence(
    info,
    save_path=os.path.join(OUTPUT_DIR, "convergence.png"),
)


# =============================================================================
# EXEMPLO 2 -- Multiplas configuracoes para comparacao
# =============================================================================

print()
print("=" * 60)
print("Exemplo 2: Multiplas configuracoes")
print("=" * 60)

configs_params = [
    dict(a=2.0, b=-3.0, c=0.5, label="a=2, b=-3, c=0.5"),
    dict(a=1.5, b=-4.0, c=0.3, label="a=1.5, b=-4, c=0.3"),
    dict(a=3.0, b=-5.0, c=0.7, label="a=3, b=-5, c=0.7"),
]

configs_com_dados = []
for cfg in configs_params:
    a_i, b_i, c_i = cfg["a"], cfg["b"], cfg["c"]
    print(f"  Calculando: a={a_i}, b={b_i}, c={c_i} ...", end=" ", flush=True)
    xs_i, ys_i, kp_i, _ = compute_frontier(a_i, b_i, c_i, n_points=200)
    print(f"{len(xs_i)} pontos gerados")
    configs_com_dados.append({**cfg, "xs": xs_i, "ys": ys_i, "key_points": kp_i})

fig3 = plot_multiple_configs(
    configs_com_dados,
    save_path=os.path.join(OUTPUT_DIR, "frontier_multiple.png"),
)

print()
print("=" * 60)
print("Concluido! Figuras salvas em examples/")
print("  frontier_single.png   -- fronteira configuracao padrao")
print("  convergence.png       -- convergencia Newton-Raphson")
print("  frontier_multiple.png -- multiplas configuracoes")
print("=" * 60)
