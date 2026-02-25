"""
voronoi_frontier/core.py
========================
Implementacao analitica e computacional da fronteira do diagrama de Voronoi
para dois pontos geradores A = (0, a) e B = (0, b) e um obstaculo circular
centrado em C = (c, 0) com raio unitario r = 1.

Referencia matematica:
    "Solving a path planning problem using the frontier of Voronoi diagram
     with two sites and a circular obstacle"
     Submetido ao Computer-Aided Design, fevereiro de 2023.

Restricoes dos parametros (Secao 3 do artigo):
    a > 0
    b < 0
    -b > a
    0 < c < 1
    a^2 + c^2 >= 1  (ponto A fora ou sobre o circulo)
    b^2 + c^2 >= 1  (ponto B fora ou sobre o circulo)
"""

import numpy as np
from typing import Tuple


# =============================================================================
# VALIDACAO DOS PARAMETROS
# =============================================================================

def validate_params(a: float, b: float, c: float) -> None:
    """
    Verifica se os parametros de entrada satisfazem as restricoes do artigo.
    Lanca ValueError com mensagem explicativa se alguma restricao for violada.

    Parametros
    ----------
    a : float  -- coordenada y do ponto gerador A = (0, a). Deve ser positivo.
    b : float  -- coordenada y do ponto gerador B = (0, b). Deve ser negativo.
    c : float  -- coordenada x do centro do obstaculo C = (c, 0). Deve satisfazer 0 < c < 1.
    """
    if a <= 0:
        raise ValueError(f"a deve ser positivo. Recebido: a = {a}")
    if b >= 0:
        raise ValueError(f"b deve ser negativo. Recebido: b = {b}")
    if -b <= a:
        raise ValueError(f"-b deve ser maior que a. Recebido: a = {a}, b = {b}")
    if not (0 < c < 1):
        raise ValueError(f"c deve satisfazer 0 < c < 1. Recebido: c = {c}")
    if a**2 + c**2 < 1:
        raise ValueError(
            f"O ponto A = (0, {a}) esta dentro do circulo centrado em ({c}, 0). "
            f"Necessario: a^2 + c^2 >= 1."
        )
    if b**2 + c**2 < 1:
        raise ValueError(
            f"O ponto B = (0, {b}) esta dentro do circulo centrado em ({c}, 0). "
            f"Necessario: b^2 + c^2 >= 1."
        )


# =============================================================================
# PONTOS GEOMETRICOS PRINCIPAIS (Proposicoes 1 a 4)
# =============================================================================

def compute_Q1_Q2(a: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proposicao 1 -- Pontos de tangencia de A = (0, a) ao circulo unitario.

    Q2 e o ponto de tangencia usado no calculo da fronteira do lado esquerdo.
    Q1 e reservado para a fronteira do lado direito (trabalho futuro).

    Condicao geometrica: o segmento CQ2 e ortogonal ao segmento AQ2.

    Parametros
    ----------
    a : float  -- coordenada y do ponto A.
    c : float  -- coordenada x do centro C do obstaculo.

    Retorna
    -------
    Q1, Q2 : np.ndarray de shape (2,)
    """
    r2 = a**2 + c**2
    sq = np.sqrt(r2 - 1.0)

    Q1_x = c + (-c + abs(a) * sq) / r2
    Q1_y = (-c**2 + c * abs(a) * sq) / (a * r2) + 1.0 / a

    Q2_x = c + (-c - abs(a) * sq) / r2
    Q2_y = (-c**2 - c * abs(a) * sq) / (a * r2) + 1.0 / a

    return np.array([Q1_x, Q1_y]), np.array([Q2_x, Q2_y])


def compute_Q5_Q6(b: float, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proposicao 4 -- Pontos de tangencia de B = (0, b) ao circulo unitario.

    Condicao geometrica: o segmento CQ5 e ortogonal ao segmento BQ5.

    Parametros
    ----------
    b : float  -- coordenada y do ponto B.
    c : float  -- coordenada x do centro C do obstaculo.

    Retorna
    -------
    Q5, Q6 : np.ndarray de shape (2,)
    """
    r2 = b**2 + c**2
    sq = np.sqrt(r2 - 1.0)

    Q5_x = c + (-c - abs(b) * sq) / r2
    Q5_y = (-c**2 - c * abs(b) * sq) / (b * r2) + 1.0 / b

    Q6_x = c + (-c + abs(b) * sq) / r2
    Q6_y = (-c**2 + c * abs(b) * sq) / (b * r2) + 1.0 / b

    return np.array([Q5_x, Q5_y]), np.array([Q6_x, Q6_y])


def compute_D(a: float, b: float, Q2: np.ndarray) -> np.ndarray:
    """
    Proposicao 2 -- Ultimo ponto da fronteira com visada direta para A e B.

    D pertence a bissetriz y = (a + b) / 2 e e o ponto de transicao a partir
    do qual o obstaculo passa a interferir na visada entre o ponto movel e B.

    Parametros
    ----------
    a  : float       -- coordenada y do ponto A.
    b  : float       -- coordenada y do ponto B.
    Q2 : np.ndarray  -- ponto de tangencia Q2 (calculado por compute_Q1_Q2).

    Retorna
    -------
    D : np.ndarray de shape (2,)
    """
    xQ2, yQ2 = Q2
    D_x = ((a - b) / 2.0 * xQ2) / (a - yQ2)
    D_y = (a + b) / 2.0
    return np.array([D_x, D_y])


# =============================================================================
# CONSTANTES LAMBDA E PHI (Equacoes 3 e 4)
# =============================================================================

def compute_lambda_phi(a: float, c: float) -> Tuple[float, float]:
    """
    Equacoes (3) e (4) -- Constantes derivadas das coordenadas de Q2.

    Interpretacao geometrica:
        lambda = cos(theta_2)  -- componente x do vetor unitario em Q2
        phi    = sin(theta_2)  -- componente y do vetor unitario em Q2

    Usadas no calculo da distancia geodesica Q2->Q3 (Proposicao 6).

    Parametros
    ----------
    a : float  -- coordenada y do ponto A.
    c : float  -- coordenada x do centro C.

    Retorna
    -------
    lam, phi : float
    """
    r2  = a**2 + c**2
    sq  = np.sqrt(r2 - 1.0)
    lam = (-c - abs(a) * sq) / r2
    phi = (-c**2 - c * abs(a) * sq) / (a * r2) + 1.0 / a
    return lam, phi


def compute_lambda1_phi1(b: float, c: float) -> Tuple[float, float]:
    """
    Analogas a lambda e phi, porem derivadas das coordenadas de Q5.

    Interpretacao geometrica:
        lambda1 = cos(theta_5)
        phi1    = sin(theta_5)

    Usadas no calculo da distancia geodesica Q4->Q5 (Proposicao 9).

    Parametros
    ----------
    b : float  -- coordenada y do ponto B.
    c : float  -- coordenada x do centro C.

    Retorna
    -------
    lam1, phi1 : float
    """
    r2   = b**2 + c**2
    sq   = np.sqrt(r2 - 1.0)
    lam1 = (-c - abs(b) * sq) / r2
    phi1 = (-c**2 - c * abs(b) * sq) / (b * r2) + 1.0 / b
    return lam1, phi1


# =============================================================================
# FUNCOES AUXILIARES Mi E SUAS DERIVADAS (Secao 4.1)
# =============================================================================

def M_functions(x: float, y: float, c: float) -> dict:
    """
    Calcula as funcoes auxiliares Mi para um ponto E = (x, y).

    Estas funcoes foram introduzidas no artigo para simplificar as expressoes
    das distancias e suas derivadas. Cada Mi representa uma subexpressao
    que aparece repetidamente nas formulas das Proposicoes 6 a 9.

    Nota: M[5] = max(y^2 + (x-c)^2 - 1, 0) e truncado em zero para evitar
    erros numericos quando E esta muito proximo da borda do circulo.

    Parametros
    ----------
    x, y : float  -- coordenadas do ponto E.
    c    : float  -- coordenada x do centro do obstaculo.

    Retorna
    -------
    dict com chaves 0..11
    """
    M = {}
    M[0]  = y
    M[1]  = y * y
    M[2]  = x - c
    M[3]  = c - x
    M[4]  = abs(y)
    M[5]  = max(y**2 + (x - c)**2 - 1.0, 0.0)
    sq5   = np.sqrt(M[5])
    M[6]  = M[2] - M[4] * sq5
    M[7]  = M[1] + M[2]**2
    M[8]  = M[0] * M[7]
    M[9]  = M[1] + M[2] * M[4] * sq5
    M[10] = M[2] + M[4] * sq5
    M[11] = M[1] - M[2] * M[4] * sq5
    return M


def dM_dy(x: float, y: float, M: dict) -> dict:
    """
    Derivadas das funcoes Mi em relacao a y.

    Estas derivadas nao estao explicitadas no artigo original. Foram derivadas
    aqui aplicando a regra da cadeia diretamente sobre cada Mi.

    Caso especial: quando M[5] = 0 (ponto E dentro ou sobre o circulo),
    o truncamento torna a derivada de M[5] igual a zero nessa regiao,
    e todas as expressoes que envolvem sqrt(M[5]) tambem se anulam.

    Parametros
    ----------
    x, y : float  -- coordenadas do ponto E.
    M    : dict   -- saida de M_functions(x, y, c).

    Retorna
    -------
    dict com chaves 0..11 (derivadas dMi/dy)
    """
    dM     = {}
    sgn_y  = np.sign(y) if y != 0.0 else 0.0
    clamped = (M[5] == 0.0)
    sq5     = np.sqrt(M[5]) if not clamped else 0.0
    sq5     = max(sq5, 1e-14) if not clamped else 0.0

    dM[0] = 1.0
    dM[1] = 2.0 * y
    dM[2] = 0.0
    dM[4] = sgn_y
    dM[5] = 0.0 if clamped else 2.0 * y

    if clamped:
        _d = 0.0
    else:
        _d = sgn_y * sq5 + M[4] * y / sq5

    dM[6]  = -_d
    dM[7]  = 2.0 * y
    dM[8]  = M[7] + 2.0 * y**2
    dM[9]  = 2.0 * y + M[2] * _d
    dM[10] = _d
    dM[11] = 2.0 * y - M[2] * _d

    return dM


# =============================================================================
# DISTANCIAS INDIVIDUAIS (Proposicoes 5 a 10)
# =============================================================================

def dist_A_Q2(a: float, c: float) -> float:
    """
    Proposicao 5 -- Distancia euclidiana de A = (0, a) a Q2.

    Derivada pelo Teorema de Pitagoras: d(A,Q2)^2 = d(A,C)^2 - r^2
    onde d(A,C) = sqrt(a^2 + c^2) e r = 1.

    Retorna
    -------
    float : sqrt(a^2 + c^2 - 1)
    """
    return np.sqrt(a**2 + c**2 - 1.0)


def dist_Q5_B(b: float, c: float) -> float:
    """
    Proposicao 10 -- Distancia euclidiana de Q5 a B = (0, b).

    Analogia direta com a Proposicao 5.

    Retorna
    -------
    float : sqrt(b^2 + c^2 - 1)
    """
    return np.sqrt(b**2 + c**2 - 1.0)


def dist_Q2_Q3(M: dict, lam: float, phi: float) -> float:
    """
    Proposicao 6 -- Distancia geodesica (comprimento do arco) de Q2 a Q3.

    Calculada como o angulo entre os vetores unitarios em Q2 e Q3:
        ||Q2Q3|| = arccos(cos(theta3)*cos(theta2) + sin(theta3)*sin(theta2))

    Parametros
    ----------
    M   : dict  -- funcoes auxiliares Mi.
    lam : float -- cos(theta_2).
    phi : float -- sin(theta_2).

    Retorna
    -------
    float : comprimento do arco Q2Q3 (em radianos, pois r = 1)
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    u = (M[6] / M[7]) * lam + (M[9] / M[8]) * phi
    return np.arccos(np.clip(u, -1.0, 1.0))


def ddist_Q2_Q3_dy(M: dict, dM: dict, lam: float, phi: float) -> float:
    """
    Derivada de ||Q2Q3|| em relacao a y.

    Aplicacao da regra da cadeia sobre arccos(u(y)):
        d/dy arccos(u) = -u'(y) / sqrt(1 - u^2)

    Parametros
    ----------
    M, dM : dict  -- funcoes Mi e suas derivadas.
    lam, phi : float -- constantes de Q2.

    Retorna
    -------
    float
    """
    M6, M7, M8, M9   = M[6], M[7], M[8], M[9]
    dM6, dM7, dM8, dM9 = dM[6], dM[7], dM[8], dM[9]

    if abs(M7) < 1e-14 or abs(M8) < 1e-14:
        return 0.0

    u   = (M6 / M7) * lam + (M9 / M8) * phi
    u_c = np.clip(u, -1.0 + 1e-10, 1.0 - 1e-10)
    du  = lam * (dM6 * M7 - M6 * dM7) / M7**2 + phi * (dM9 * M8 - M9 * dM8) / M8**2
    return -du / np.sqrt(1.0 - u_c**2)


def _dist_Q3_E(M: dict, x: float, y: float, c: float) -> float:
    """
    Proposicao 7 -- Distancia euclidiana de Q3 a E = (x, y).

    Q3 = (c + M6/M7, M9/M8), portanto:
        ||Q3E|| = sqrt((M6/M7 + c - x)^2 + (M9/M8 - y)^2)

    Parametros
    ----------
    M    : dict  -- funcoes auxiliares Mi.
    x, y : float -- coordenadas do ponto E.
    c    : float -- coordenada x do centro.
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    dx = M[6] / M[7] + c - x
    dy = M[9] / M[8] - y
    return np.sqrt(dx**2 + dy**2)


def _ddist_Q3_E_dy(M: dict, dM: dict, x: float, y: float, c: float) -> float:
    """
    Derivada de ||Q3E|| em relacao a y.
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    M6, M7, M8, M9   = M[6], M[7], M[8], M[9]
    dM6, dM7, dM8, dM9 = dM[6], dM[7], dM[8], dM[9]

    P  = M6 / M7 + c - x
    R  = M9 / M8 - y
    dP = (dM6 * M7 - M6 * dM7) / M7**2
    dR = (dM9 * M8 - M9 * dM8) / M8**2 - 1.0

    dist = np.sqrt(P**2 + R**2)
    if dist < 1e-14:
        return 0.0
    return (P * dP + R * dR) / dist


def _dist_E_B(x: float, y: float, b: float) -> float:
    """
    Distancia euclidiana de E = (x, y) a B = (0, b).
    """
    return np.sqrt(x**2 + (y - b)**2)


def _ddist_E_B_dy(x: float, y: float, b: float) -> float:
    """
    Derivada de d(E, B) em relacao a y.
    """
    d = _dist_E_B(x, y, b)
    return (y - b) / d if d > 1e-14 else 0.0


def _dist_E_Q4(M: dict, x: float, y: float, c: float) -> float:
    """
    Proposicao 8 -- Distancia euclidiana de E = (x, y) a Q4.

    Q4 = (c + M10/M7, M11/M8), portanto:
        d(E, Q4) = sqrt((x - c - M10/M7)^2 + (y - M11/M8)^2)
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    S = x - c - M[10] / M[7]
    T = y - M[11] / M[8]
    return np.sqrt(S**2 + T**2)


def _ddist_E_Q4_dy(M: dict, dM: dict, x: float, y: float, c: float) -> float:
    """
    Derivada de d(E, Q4) em relacao a y.
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    M7, M8, M10, M11     = M[7], M[8], M[10], M[11]
    dM7, dM8, dM10, dM11 = dM[7], dM[8], dM[10], dM[11]

    S  = x - c - M10 / M7
    T  = y - M11 / M8
    dS = -(dM10 * M7 - M10 * dM7) / M7**2
    dT = 1.0 - (dM11 * M8 - M11 * dM8) / M8**2

    dist = np.sqrt(S**2 + T**2)
    if dist < 1e-14:
        return 0.0
    return (S * dS + T * dT) / dist


def _dist_Q4_Q5(M: dict, lam1: float, phi1: float) -> float:
    """
    Proposicao 9 -- Distancia geodesica (comprimento do arco) de Q4 a Q5.

        ||Q4Q5|| = arccos(cos(theta4)*cos(theta5) + sin(theta4)*sin(theta5))
    """
    if abs(M[7]) < 1e-14 or abs(M[8]) < 1e-14:
        return 0.0
    u = (M[10] / M[7]) * lam1 + (M[11] / M[8]) * phi1
    return np.arccos(np.clip(u, -1.0, 1.0))


def _ddist_Q4_Q5_dy(M: dict, dM: dict, lam1: float, phi1: float) -> float:
    """
    Derivada de ||Q4Q5|| em relacao a y.
    """
    M7, M8, M10, M11     = M[7], M[8], M[10], M[11]
    dM7, dM8, dM10, dM11 = dM[7], dM[8], dM[10], dM[11]

    if abs(M7) < 1e-14 or abs(M8) < 1e-14:
        return 0.0

    u   = (M10 / M7) * lam1 + (M11 / M8) * phi1
    u_c = np.clip(u, -1.0 + 1e-10, 1.0 - 1e-10)
    du  = lam1 * (dM10 * M7 - M10 * dM7) / M7**2 + phi1 * (dM11 * M8 - M11 * dM8) / M8**2
    return -du / np.sqrt(1.0 - u_c**2)


# =============================================================================
# RESIDUOS F(y) E DERIVADAS F'(y) PARA O NEWTON-RAPHSON (Equacoes 19-22)
# =============================================================================

def F_situation1(x: float, y: float,
                 a: float, b: float, c: float,
                 lam: float, phi: float,
                 dAQ2: float) -> float:
    """
    Equacao (19) -- Residuo para a Situacao 1.

    Situacao 1: Pi tem visada direta a B (obstaculo nao bloqueia B).

    A fronteira e definida pela condicao de equidistancia:
        d(A, Pi) = d(Pi, B)

    onde d(A, Pi) e calculado via geodesica:
        d(A, Pi) = d(A, Q2) + ||Q2Q3|| + ||Q3Pi||

    Portanto: F = d(A,Q2) + ||Q2Q3|| + ||Q3Pi|| - d(Pi, B) = 0

    Parametros
    ----------
    x, y : float  -- coordenadas do ponto Pi = (x, y).
    a, b, c : float  -- parametros de configuracao.
    lam, phi : float -- constantes de Q2.
    dAQ2 : float     -- d(A, Q2), precalculado (constante para todo i).
    """
    M = M_functions(x, y, c)
    return (dAQ2
            + dist_Q2_Q3(M, lam, phi)
            + _dist_Q3_E(M, x, y, c)
            - _dist_E_B(x, y, b))


def dF_situation1_dy(x: float, y: float,
                     a: float, b: float, c: float,
                     lam: float, phi: float) -> float:
    """
    Equacao (21) -- Derivada do residuo F em relacao a y (Situacao 1).

    Usada no passo de Newton-Raphson:
        y_{j+1} = y_j - F(y_j) / F'(y_j)
    """
    M  = M_functions(x, y, c)
    dM = dM_dy(x, y, M)
    return (ddist_Q2_Q3_dy(M, dM, lam, phi)
            + _ddist_Q3_E_dy(M, dM, x, y, c)
            - _ddist_E_B_dy(x, y, b))


def F_situation2(x: float, y: float,
                 a: float, b: float, c: float,
                 lam: float, phi: float,
                 lam1: float, phi1: float,
                 dAQ2: float, dQ5B: float) -> float:
    """
    Equacao (20) -- Residuo para a Situacao 2.

    Situacao 2: Obstaculo bloqueia a visada para AMBOS A e B.

    Neste caso, a geodesica de Pi ate B tambem contorna o obstaculo:
        d(Pi, B) = ||PiQ4|| + ||Q4Q5|| + d(Q5, B)

    Portanto:
        F = d(A,Q2) + ||Q2Q3|| + ||Q3Pi|| - ||PiQ4|| - ||Q4Q5|| - d(Q5,B) = 0

    Parametros
    ----------
    lam1, phi1 : float -- constantes de Q5.
    dQ5B : float       -- d(Q5, B), precalculado (constante para todo i).
    """
    M = M_functions(x, y, c)
    return (dAQ2
            + dist_Q2_Q3(M, lam, phi)
            + _dist_Q3_E(M, x, y, c)
            - _dist_E_Q4(M, x, y, c)
            - _dist_Q4_Q5(M, lam1, phi1)
            - dQ5B)


def dF_situation2_dy(x: float, y: float,
                     a: float, b: float, c: float,
                     lam: float, phi: float,
                     lam1: float, phi1: float) -> float:
    """
    Equacao (22) -- Derivada do residuo F em relacao a y (Situacao 2).
    """
    M  = M_functions(x, y, c)
    dM = dM_dy(x, y, M)
    return (ddist_Q2_Q3_dy(M, dM, lam, phi)
            + _ddist_Q3_E_dy(M, dM, x, y, c)
            - _ddist_E_Q4_dy(M, dM, x, y, c)
            - _ddist_Q4_Q5_dy(M, dM, lam1, phi1))


# =============================================================================
# TESTES DE PERTENCIMENTO (Secao 6.4 do algoritmo)
# =============================================================================

def is_below_line_BQ5(x: float, y: float,
                      b: float, Q5: np.ndarray) -> bool:
    """
    Retorna True se o ponto (x, y) esta estritamente abaixo da reta BQ5.

    Este teste determina a transicao da Situacao 1 para a Situacao 2.
    A reta e definida pelos pontos B = (0, b) e Q5.
    O teste usa o produto vetorial: se negativo, o ponto esta abaixo.

    Parametros
    ----------
    x, y : float       -- coordenadas do ponto a testar.
    b    : float       -- coordenada y de B.
    Q5   : np.ndarray  -- ponto de tangencia Q5.
    """
    B  = np.array([0.0, b])
    dx = Q5[0] - B[0]
    dy = Q5[1] - B[1]
    cross = dx * (y - B[1]) - dy * (x - B[0])
    return cross < 0.0


def is_inside_circle(x: float, y: float, c: float, r: float = 1.0) -> bool:
    """
    Retorna True se o ponto (x, y) esta estritamente dentro do circulo.

    O circulo tem centro (c, 0) e raio r (padrao r = 1).

    Parametros
    ----------
    x, y : float  -- coordenadas do ponto a testar.
    c    : float  -- coordenada x do centro do obstaculo.
    r    : float  -- raio do circulo (padrao 1.0).
    """
    return (x - c)**2 + y**2 < r**2


# =============================================================================
# ALGORITMO PRINCIPAL (Algoritmo 1 do artigo, versao corrigida)
# =============================================================================

def compute_frontier(a: float, b: float, c: float,
                     n_points: int = 200,
                     max_iter: int = 50,
                     tol: float = 1e-10
                     ) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Calcula a fronteira de Voronoi do lado esquerdo para a configuracao:
        A = (0, a),  B = (0, b),  obstaculo = circulo unitario em (c, 0).

    O algoritmo discretiza o eixo x a partir do ponto D (ultimo ponto da
    bissetriz) em direcao ao obstaculo, e para cada xi usa Newton-Raphson
    para encontrar yi tal que (xi, yi) pertenca a fronteira.

    CORRECAO aplicada em relacao ao artigo original:
        O criterio de parada do Newton-Raphson usava OR (incorreto).
        A versao correta usa AND:
            while j < max_iter AND |F(y)| > tol

    Parametros
    ----------
    a, b, c   : float  -- parametros de configuracao.
    n_points  : int    -- numero maximo de pontos a gerar (padrao 200).
    max_iter  : int    -- iteracoes maximas do Newton-Raphson (padrao 50).
    tol       : float  -- criterio de convergencia |F(y)| < tol (padrao 1e-10).

    Retorna
    -------
    xs, ys      : np.ndarray  -- coordenadas dos pontos da fronteira.
    key_points  : dict        -- pontos auxiliares {Q1, Q2, Q5, Q6, D}.
    info        : dict        -- informacoes diagnosticas por ponto:
                                 'situation_flags': lista de 1 ou 2 por ponto,
                                 'iter_counts': numero de iteracoes N-R por ponto.
    """
    validate_params(a, b, c)

    # --- Constantes precalculadas ---
    lam,  phi  = compute_lambda_phi(a, c)
    lam1, phi1 = compute_lambda1_phi1(b, c)
    Q1, Q2     = compute_Q1_Q2(a, c)
    Q5, Q6     = compute_Q5_Q6(b, c)
    D          = compute_D(a, b, Q2)
    dAQ2       = dist_A_Q2(a, c)
    dQ5B       = dist_Q5_B(b, c)

    key_points = {"Q1": Q1, "Q2": Q2, "Q5": Q5, "Q6": Q6, "D": D}

    # --- Discretizacao do eixo x (linhas 7-9 do Algoritmo 1) ---
    x0 = D[0]
    h  = (c - x0) / (n_points + 1)

    xs, ys           = [], []
    situation_flags  = []
    iter_counts      = []

    y_guess        = D[1]
    flag_test_line = True
    flag_circle    = False

    for i in range(1, n_points + 1):
        if flag_circle:
            break

        xi = x0 + i * h
        yi = y_guess

        # Seleciona residuo e derivada conforme a situacao atual
        if flag_test_line:
            def F(y):
                return F_situation1(xi, y, a, b, c, lam, phi, dAQ2)
            def dF(y):
                return dF_situation1_dy(xi, y, a, b, c, lam, phi)
        else:
            def F(y):
                return F_situation2(xi, y, a, b, c, lam, phi, lam1, phi1, dAQ2, dQ5B)
            def dF(y):
                return dF_situation2_dy(xi, y, a, b, c, lam, phi, lam1, phi1)

        # Newton-Raphson com criterio de parada CORRIGIDO (AND em vez de OR)
        iters = 0
        for j in range(max_iter):
            f_val = F(yi)
            if abs(f_val) < tol:
                break
            df_val = dF(yi)
            if abs(df_val) < 1e-14:
                break
            yi   -= f_val / df_val
            iters += 1

        # Registra ponto e atualiza flags
        xs.append(xi)
        ys.append(yi)
        situation_flags.append(1 if flag_test_line else 2)
        iter_counts.append(iters)
        y_guess = yi

        if flag_test_line and is_below_line_BQ5(xi, yi, b, Q5):
            flag_test_line = False

        if is_inside_circle(xi, yi, c):
            xs.pop(); ys.pop()
            situation_flags.pop(); iter_counts.pop()
            flag_circle = True

    info = {
        "situation_flags": situation_flags,
        "iter_counts":     iter_counts,
    }

    return np.array(xs), np.array(ys), key_points, info


# =============================================================================
# GENERALIZACAO PARA RAIO ARBITRARIO r
# =============================================================================

def compute_frontier_r(a: float, b: float, c: float, r: float,
                       n_points: int = 200,
                       max_iter: int = 50,
                       tol: float = 1e-10
                       ) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Calcula a fronteira de Voronoi para um obstaculo circular de raio r arbitrario.

    Esta funcao generaliza compute_frontier (que assume r=1) usando o principio
    de invariancia por escala: um circulo de raio r centrado em (c, 0) e
    equivalente ao circulo unitario apos escalar todas as coordenadas por 1/r.

    Procedimento:
        1. Escalar os parametros: a -> a/r, b -> b/r, c -> c/r
        2. Chamar compute_frontier com o problema normalizado (r=1)
        3. Reescalar os resultados multiplicando por r

    Restricoes dos parametros:
        a > 0
        b < 0
        -b > a
        0 < c < r    (centro do obstaculo dentro do intervalo)
        a^2 + c^2 >= r^2  (A fora do circulo)
        b^2 + c^2 >= r^2  (B fora do circulo)

    Parametros
    ----------
    a, b, c : float -- mesmos parametros de compute_frontier.
    r       : float -- raio do obstaculo circular (deve ser positivo).
    n_points, max_iter, tol : mesmos parametros de compute_frontier.

    Retorna
    -------
    xs, ys     : np.ndarray -- coordenadas dos pontos da fronteira (escala original).
    key_points : dict       -- pontos auxiliares Q1, Q2, Q5, Q6, D (escala original).
    info       : dict       -- informacoes diagnosticas (igual a compute_frontier).

    Exemplo
    -------
    >>> xs, ys, kp, info = compute_frontier_r(a=4.0, b=-6.0, c=1.0, r=2.0)
    >>> # Equivalente a compute_frontier(a=2.0, b=-3.0, c=0.5) escalado por 2
    """
    if r <= 0:
        raise ValueError(f"r deve ser positivo. Recebido: r = {r}")

    # Validacao das restricoes no espaco original
    if a <= 0:
        raise ValueError(f"a deve ser positivo. Recebido: a = {a}")
    if b >= 0:
        raise ValueError(f"b deve ser negativo. Recebido: b = {b}")
    if -b <= a:
        raise ValueError(f"-b deve ser maior que a. Recebido: a = {a}, b = {b}")
    if not (0 < c < r):
        raise ValueError(f"c deve satisfazer 0 < c < r = {r}. Recebido: c = {c}")
    if a**2 + c**2 < r**2:
        raise ValueError(
            f"O ponto A = (0, {a}) esta dentro do circulo de raio {r}. "
            f"Necessario: a^2 + c^2 >= r^2."
        )
    if b**2 + c**2 < r**2:
        raise ValueError(
            f"O ponto B = (0, {b}) esta dentro do circulo de raio {r}. "
            f"Necessario: b^2 + c^2 >= r^2."
        )

    # Escalar para o problema unitario (r=1)
    a_n = a / r
    b_n = b / r
    c_n = c / r

    # Resolver no espaco normalizado
    xs_n, ys_n, kp_n, info = compute_frontier(
        a_n, b_n, c_n,
        n_points=n_points,
        max_iter=max_iter,
        tol=tol
    )

    # Reescalar os resultados para o espaco original
    xs = xs_n * r
    ys = ys_n * r
    key_points = {nome: ponto * r for nome, ponto in kp_n.items()}

    return xs, ys, key_points, info
