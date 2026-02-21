"""
voronoi_frontier/__init__.py
============================
Ponto de entrada do pacote voronoi_frontier.

Exporta todas as funcoes publicas de core.py e plot.py para que
os pesquisadores possam importar diretamente do pacote:

    from voronoi_frontier import compute_frontier, plot_frontier

Modulos
-------
core.py  -- matematica: pontos geometricos, distancias, derivadas, algoritmo
plot.py  -- visualizacao: figuras de fronteira, convergencia e comparacoes
"""

from voronoi_frontier.core import (
    # Validacao
    validate_params,
    # Pontos geometricos principais (Proposicoes 1-4)
    compute_Q1_Q2,
    compute_Q5_Q6,
    compute_D,
    # Constantes lambda e phi (Equacoes 3-4)
    compute_lambda_phi,
    compute_lambda1_phi1,
    # Funcoes auxiliares Mi e derivadas (Secao 4.1)
    M_functions,
    dM_dy,
    # Distancias individuais (Proposicoes 5-10)
    dist_A_Q2,
    dist_Q5_B,
    dist_Q2_Q3,
    ddist_Q2_Q3_dy,
    # Residuos e derivadas para Newton-Raphson (Equacoes 19-22)
    F_situation1,
    dF_situation1_dy,
    F_situation2,
    dF_situation2_dy,
    # Testes de pertencimento (Secao 6.4)
    is_below_line_BQ5,
    is_inside_circle,
    # Algoritmo principal
    compute_frontier,
)

from voronoi_frontier.plot import (
    plot_frontier,
    plot_convergence,
    plot_multiple_configs,
)

__all__ = [
    "validate_params",
    "compute_Q1_Q2", "compute_Q5_Q6", "compute_D",
    "compute_lambda_phi", "compute_lambda1_phi1",
    "M_functions", "dM_dy",
    "dist_A_Q2", "dist_Q5_B",
    "dist_Q2_Q3", "ddist_Q2_Q3_dy",
    "F_situation1", "dF_situation1_dy",
    "F_situation2", "dF_situation2_dy",
    "is_below_line_BQ5", "is_inside_circle",
    "compute_frontier",
    "plot_frontier", "plot_convergence", "plot_multiple_configs",
]
