"""
tests/test_voronoi.py
=====================
Testes unitarios para o pacote voronoi_frontier.

Cobertura: 30 testes organizados em 7 classes.
Execute com:
    pytest tests/ -v

Classes de teste
----------------
TestValidation      -- restricoes dos parametros de entrada
TestKeyPoints       -- pontos Q1-Q5 sobre o circulo, tangencia, D na bissetriz
TestDistances       -- Proposicoes 5 e 10 vs. distancia euclidiana direta
TestDerivatives     -- dF/dy analitica vs. diferencas finitas centradas
TestNewtonRaphson   -- equidistancia, monotonicidade, convergencia
TestMembership      -- testes de pertencimento ao circulo e a reta BQ5
TestRobustness      -- multiplas configuracoes (a, b, c)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voronoi_frontier.core import (
    validate_params,
    compute_Q1_Q2,
    compute_Q5_Q6,
    compute_D,
    compute_lambda_phi,
    compute_lambda1_phi1,
    M_functions,
    dM_dy,
    dist_A_Q2,
    dist_Q5_B,
    dist_Q2_Q3,
    ddist_Q2_Q3_dy,
    F_situation1,
    dF_situation1_dy,
    F_situation2,
    dF_situation2_dy,
    is_below_line_BQ5,
    is_inside_circle,
    compute_frontier,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_params():
    """Configuracao padrao de teste: a=2, b=-3, c=0.5."""
    return dict(a=2.0, b=-3.0, c=0.5)


@pytest.fixture
def tight_params():
    """Configuracao onde A esta quase sobre o circulo."""
    return dict(a=1.0, b=-2.0, c=0.0001)


# =============================================================================
# 1. VALIDACAO DOS PARAMETROS
# =============================================================================

class TestValidation:
    """Verifica que validate_params aceita entradas validas e rejeita invalidas."""

    def test_params_validos_nao_lancam_excecao(self, default_params):
        validate_params(**default_params)

    def test_a_deve_ser_positivo(self):
        with pytest.raises(ValueError, match="a deve ser positivo"):
            validate_params(a=-1.0, b=-3.0, c=0.5)

    def test_b_deve_ser_negativo(self):
        with pytest.raises(ValueError, match="b deve ser negativo"):
            validate_params(a=2.0, b=1.0, c=0.5)

    def test_menos_b_deve_superar_a(self):
        with pytest.raises(ValueError, match="-b deve ser maior que a"):
            validate_params(a=3.0, b=-2.0, c=0.5)

    def test_c_deve_estar_entre_0_e_1(self):
        with pytest.raises(ValueError, match="0 < c < 1"):
            validate_params(a=2.0, b=-3.0, c=1.5)

    def test_A_deve_estar_fora_do_circulo(self):
        with pytest.raises(ValueError, match="dentro do circulo"):
            validate_params(a=0.5, b=-2.0, c=0.5)


# =============================================================================
# 2. PONTOS GEOMETRICOS PRINCIPAIS (Proposicoes 1, 2, 4)
# =============================================================================

class TestKeyPoints:
    """Verifica as propriedades geometricas dos pontos Q1, Q2, Q5, Q6 e D."""

    def test_Q2_sobre_o_circulo(self, default_params):
        """Q2 deve estar exatamente sobre o circulo unitario centrado em (c, 0)."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        raio = np.sqrt((Q2[0] - c)**2 + Q2[1]**2)
        assert abs(raio - 1.0) < 1e-10, f"Q2 fora do circulo: r = {raio}"

    def test_Q1_sobre_o_circulo(self, default_params):
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        Q1, _ = compute_Q1_Q2(a, c)
        raio = np.sqrt((Q1[0] - c)**2 + Q1[1]**2)
        assert abs(raio - 1.0) < 1e-10

    def test_Q5_sobre_o_circulo(self, default_params):
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        Q5, _ = compute_Q5_Q6(b, c)
        raio = np.sqrt((Q5[0] - c)**2 + Q5[1]**2)
        assert abs(raio - 1.0) < 1e-10

    def test_tangencia_AQ2(self, default_params):
        """CQ2 deve ser ortogonal a AQ2 (condicao de tangencia)."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        A = np.array([0.0, a])
        C = np.array([c, 0.0])
        produto_escalar = (Q2 - A) @ (Q2 - C)
        assert abs(produto_escalar) < 1e-8, f"AQ2 nao e tangente: dot = {produto_escalar}"

    def test_tangencia_BQ5(self, default_params):
        """CQ5 deve ser ortogonal a BQ5 (condicao de tangencia)."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        Q5, _ = compute_Q5_Q6(b, c)
        B = np.array([0.0, b])
        C = np.array([c, 0.0])
        produto_escalar = (Q5 - B) @ (Q5 - C)
        assert abs(produto_escalar) < 1e-8

    def test_D_sobre_a_bissetriz(self, default_params):
        """D deve estar na bissetriz y = (a + b) / 2."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        D = compute_D(a, b, Q2)
        assert abs(D[1] - (a + b) / 2.0) < 1e-10

    def test_D_equidistante_de_A_e_B(self, default_params):
        """D deve ser equidistante de A e de B."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        D = compute_D(a, b, Q2)
        dA = np.linalg.norm(D - np.array([0.0, a]))
        dB = np.linalg.norm(D - np.array([0.0, b]))
        assert abs(dA - dB) < 1e-8, f"D nao equidistante: dA={dA:.6f}, dB={dB:.6f}"


# =============================================================================
# 3. FORMULAS DE DISTANCIA (Proposicoes 5 e 10)
# =============================================================================

class TestDistances:
    """Verifica as formulas analiticas de distancia contra calculos diretos."""

    def test_dist_A_Q2_proposicao5(self, default_params):
        """d(A, Q2) = sqrt(a^2 + c^2 - 1) conforme Proposicao 5."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        esperado = np.sqrt(a**2 + c**2 - 1)
        assert abs(dist_A_Q2(a, c) - esperado) < 1e-12

    def test_dist_A_Q2_confere_com_euclidiana(self, default_params):
        """Formula da Proposicao 5 deve coincidir com a distancia euclidiana direta."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        A = np.array([0.0, a])
        euclidiana = np.linalg.norm(A - Q2)
        formula    = dist_A_Q2(a, c)
        assert abs(euclidiana - formula) < 1e-10

    def test_dist_Q5_B_confere_com_euclidiana(self, default_params):
        """Formula da Proposicao 10 deve coincidir com a distancia euclidiana direta."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        Q5, _ = compute_Q5_Q6(b, c)
        B = np.array([0.0, b])
        euclidiana = np.linalg.norm(Q5 - B)
        formula    = dist_Q5_B(b, c)
        assert abs(euclidiana - formula) < 1e-10

    def test_geodesica_Q2_Q3_e_arco_valido(self, default_params):
        """A distancia geodesica Q2->Q3 deve estar no intervalo [0, pi]."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        lam, phi = compute_lambda_phi(a, c)
        x, y = -2.0, (a + b) / 2.0
        M    = M_functions(x, y, c)
        arco = dist_Q2_Q3(M, lam, phi)
        assert 0.0 <= arco <= np.pi


# =============================================================================
# 4. DERIVADAS ANALITICAS (verificacao por diferencas finitas)
# =============================================================================

class TestDerivatives:
    """
    Valida as derivadas analiticas comparando com diferencas finitas centradas.

    Nota: os pontos de teste devem estar FORA do circulo (M[5] > 0)
    para que as derivadas sejam bem definidas.
    """

    EPS = 1e-6

    def _deriv_numerica(self, f, y, eps=None):
        eps = eps or self.EPS
        return (f(y + eps) - f(y - eps)) / (2 * eps)

    def test_dF_situacao1_vs_numerica(self, default_params):
        """Derivada analitica de F (Situacao 1) vs. diferenca finita."""
        a, b, c  = default_params["a"], default_params["b"], default_params["c"]
        lam, phi = compute_lambda_phi(a, c)
        dAQ2     = dist_A_Q2(a, c)

        # Ponto fora do circulo: (x-c)^2 + y^2 = 6.5 > 1
        x, y0 = -2.0, 0.5
        F        = lambda y: F_situation1(x, y, a, b, c, lam, phi, dAQ2)
        dF_anal  = dF_situation1_dy(x, y0, a, b, c, lam, phi)
        dF_num   = self._deriv_numerica(F, y0)

        assert abs(dF_anal - dF_num) < 1e-5, (
            f"Situacao1 - analitica={dF_anal:.6f}, numerica={dF_num:.6f}"
        )

    def test_dF_situacao2_vs_numerica(self, default_params):
        """Derivada analitica de F (Situacao 2) vs. diferenca finita."""
        a, b, c    = default_params["a"], default_params["b"], default_params["c"]
        lam,  phi  = compute_lambda_phi(a, c)
        lam1, phi1 = compute_lambda1_phi1(b, c)
        dAQ2 = dist_A_Q2(a, c)
        dQ5B = dist_Q5_B(b, c)

        x, y0 = -2.0, -0.8
        F       = lambda y: F_situation2(x, y, a, b, c, lam, phi, lam1, phi1, dAQ2, dQ5B)
        dF_anal = dF_situation2_dy(x, y0, a, b, c, lam, phi, lam1, phi1)
        dF_num  = self._deriv_numerica(F, y0)

        assert abs(dF_anal - dF_num) < 1e-5, (
            f"Situacao2 - analitica={dF_anal:.6f}, numerica={dF_num:.6f}"
        )

    def test_dM_dy_vs_numerica(self, default_params):
        """Cada dMi/dy analitica deve coincidir com a diferenca finita."""
        c     = default_params["c"]
        x, y0 = -2.0, 0.5   # ponto fora do circulo
        eps   = self.EPS

        for key in [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]:
            num = (M_functions(x, y0 + eps, c)[key]
                   - M_functions(x, y0 - eps, c)[key]) / (2 * eps)
            M   = M_functions(x, y0, c)
            ana = dM_dy(x, y0, M)[key]
            assert abs(ana - num) < 1e-5, (
                f"dM[{key}]/dy - analitica={ana:.6f}, numerica={num:.6f}"
            )


# =============================================================================
# 5. NEWTON-RAPHSON E QUALIDADE DA FRONTEIRA
# =============================================================================

class TestNewtonRaphson:
    """Verifica convergencia, monotonicidade e equidistancia dos pontos gerados."""

    def test_pontos_satisfazem_condicao_de_fronteira(self, default_params):
        """
        Todo ponto (xi, yi) calculado deve satisfazer |F(yi)| < tolerancia,
        ou seja, deve ser equidistante de A e B (via geodesica).
        """
        a, b, c    = default_params["a"], default_params["b"], default_params["c"]
        lam,  phi  = compute_lambda_phi(a, c)
        lam1, phi1 = compute_lambda1_phi1(b, c)
        dAQ2 = dist_A_Q2(a, c)
        dQ5B = dist_Q5_B(b, c)

        xs, ys, _, info = compute_frontier(a, b, c, n_points=100)
        tol = 1e-4

        for xi, yi, sit in zip(xs, ys, info["situation_flags"]):
            assert not is_inside_circle(xi, yi, c), (
                f"Ponto ({xi:.4f}, {yi:.4f}) dentro do obstaculo"
            )
            if sit == 1:
                residuo = abs(F_situation1(xi, yi, a, b, c, lam, phi, dAQ2))
            else:
                residuo = abs(F_situation2(xi, yi, a, b, c, lam, phi, lam1, phi1, dAQ2, dQ5B))
            assert residuo < tol, (
                f"Ponto ({xi:.4f}, {yi:.4f}) nao esta na fronteira: residuo = {residuo:.2e}"
            )

    def test_numero_minimo_de_pontos(self, default_params):
        """O algoritmo deve gerar ao menos 10 pontos para a configuracao padrao."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        xs, ys, _, _ = compute_frontier(a, b, c, n_points=50)
        assert len(xs) >= 10

    def test_coordenadas_x_sao_monotonicamente_crescentes(self, default_params):
        """Os valores de x da fronteira devem ser estritamente crescentes."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        xs, _, _, _ = compute_frontier(a, b, c, n_points=100)
        assert np.all(np.diff(xs) > 0), "Coordenadas x nao sao monotonicamente crescentes"

    def test_iteracoes_dentro_do_limite(self, default_params):
        """Newton-Raphson deve convergir dentro do numero maximo de iteracoes."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, _, _, info = compute_frontier(a, b, c, n_points=100, max_iter=50)
        assert max(info["iter_counts"]) <= 50


# =============================================================================
# 6. TESTES DE PERTENCIMENTO
# =============================================================================

class TestMembership:
    """Verifica os testes de pertencimento ao circulo e a reta BQ5."""

    def test_centro_esta_dentro_do_circulo(self, default_params):
        c = default_params["c"]
        assert is_inside_circle(c, 0.0, c) is True

    def test_ponto_distante_esta_fora_do_circulo(self, default_params):
        c = default_params["c"]
        assert is_inside_circle(c + 1.5, 0.0, c) is False

    def test_ponto_sobre_o_circulo_nao_esta_dentro(self, default_params):
        c = default_params["c"]
        assert is_inside_circle(c + 1.0, 0.0, c) is False

    def test_ponto_D_nao_esta_abaixo_da_reta_BQ5(self, default_params):
        """D e o ponto inicial da fronteira e deve estar na Situacao 1."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, Q2 = compute_Q1_Q2(a, c)
        Q5, _ = compute_Q5_Q6(b, c)
        D     = compute_D(a, b, Q2)
        assert not is_below_line_BQ5(D[0], D[1], b, Q5)

    def test_B_nao_esta_abaixo_da_reta_BQ5(self, default_params):
        """B esta sobre a reta, portanto nao esta abaixo dela."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        Q5, _ = compute_Q5_Q6(b, c)
        assert not is_below_line_BQ5(0.0, b, b, Q5)


# =============================================================================
# 7. ROBUSTEZ COM MULTIPLAS CONFIGURACOES
# =============================================================================

class TestRobustness:
    """Verifica que o algoritmo funciona corretamente para diferentes (a, b, c)."""

    def test_multiplas_configuracoes_geram_pontos(self):
        """O algoritmo deve gerar pontos para todas as configuracoes validas."""
        configs = [
            (2.0, -3.0, 0.5),
            (1.5, -4.0, 0.3),
            (3.0, -5.0, 0.7),
        ]
        for a, b, c in configs:
            xs, ys, _, _ = compute_frontier(a, b, c, n_points=80)
            assert len(xs) > 0, f"Nenhum ponto gerado para a={a}, b={b}, c={c}"

    def test_dimensoes_das_saidas_sao_consistentes(self, default_params):
        """xs, ys e os vetores de info devem ter o mesmo comprimento."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        xs, ys, kp, info = compute_frontier(a, b, c, n_points=60)
        assert xs.shape == ys.shape
        assert len(info["situation_flags"]) == len(xs)
        assert len(info["iter_counts"])     == len(xs)

    def test_todos_os_pontos_auxiliares_presentes(self, default_params):
        """O dicionario key_points deve conter Q1, Q2, Q5, Q6 e D."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        _, _, kp, _ = compute_frontier(a, b, c, n_points=10)
        for nome in ("Q1", "Q2", "Q5", "Q6", "D"):
            assert nome in kp, f"Ponto auxiliar ausente: {nome}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# 8. GENERALIZACAO PARA RAIO ARBITRARIO r
# =============================================================================

from voronoi_frontier.core import compute_frontier_r

class TestFrontierR:
    """
    Verifica a generalizacao para raio r arbitrario.

    Principio testado: compute_frontier_r(a, b, c, r) deve produzir
    exatamente os mesmos pontos que compute_frontier(a/r, b/r, c/r)
    escalados por r. Isso valida o principio de invariancia por escala.
    """

    def test_r1_identico_a_compute_frontier(self, default_params):
        """Para r=1, compute_frontier_r deve ser identico a compute_frontier."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        xs1, ys1, _, _ = compute_frontier(a, b, c, n_points=50)
        xs2, ys2, _, _ = compute_frontier_r(a, b, c, r=1.0, n_points=50)
        np.testing.assert_allclose(xs1, xs2, rtol=1e-10)
        np.testing.assert_allclose(ys1, ys2, rtol=1e-10)

    def test_escala_por_r(self, default_params):
        """
        compute_frontier_r(a, b, c, r) deve ser igual a
        compute_frontier(a/r, b/r, c/r) * r.
        """
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        r = 2.0
        # Resultado de referencia: compute_frontier(a,b,c) escalado por r
        # Pois compute_frontier_r(a*r, b*r, c*r, r) divide internamente por r,
        # recuperando exatamente (a, b, c) -> resultado = compute_frontier(a,b,c) * r
        xs_n, ys_n, _, _ = compute_frontier(a, b, c, n_points=50)
        xs_esperado = xs_n * r
        ys_esperado = ys_n * r
        # Resultado da funcao generalizada
        xs_r, ys_r, _, _ = compute_frontier_r(a*r, b*r, c*r, r=r, n_points=50)
        np.testing.assert_allclose(xs_r, xs_esperado, rtol=1e-8)
        np.testing.assert_allclose(ys_r, ys_esperado, rtol=1e-8)

    def test_pontos_auxiliares_escalados(self, default_params):
        """Os pontos auxiliares Q2 e D devem ser escalados por r."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        r = 3.0
        _, _, kp1, _ = compute_frontier(a, b, c, n_points=10)
        _, _, kp_r, _ = compute_frontier_r(a*r, b*r, c*r, r=r, n_points=10)
        for nome in ("Q2", "Q5", "D"):
            np.testing.assert_allclose(
                kp_r[nome], kp1[nome] * r, rtol=1e-8,
                err_msg=f"Ponto {nome} nao foi escalado corretamente"
            )

    def test_r_negativo_lanca_excecao(self, default_params):
        """r negativo deve lancar ValueError."""
        a, b, c = default_params["a"], default_params["b"], default_params["c"]
        with pytest.raises(ValueError, match="r deve ser positivo"):
            compute_frontier_r(a, b, c, r=-1.0)

    def test_multiplos_valores_de_r(self):
        """O algoritmo deve funcionar para varios valores de r."""
        configs = [
            (4.0, -6.0, 1.0,  2.0),
            (6.0, -9.0, 1.5,  3.0),
            (1.0, -1.5, 0.25, 0.5),
        ]
        for a, b, c, r in configs:
            xs, ys, _, _ = compute_frontier_r(a, b, c, r=r, n_points=50)
            assert len(xs) > 0, f"Nenhum ponto gerado para r={r}"

