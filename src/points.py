import numpy as np


def compute_Q1_Q2(a, c):
    """
    Calcula as coordenadas de Q1 e Q2.
    Q1 e Q2 são os pontos sobre o círculo tais que AQ1 e AQ2
    são tangentes ao círculo de centro C=(c,0) e raio 1.
    Restrições: a(a²+c²) != 0 e (a²+c²-1) >= 0
    """
    a2c2 = a**2 + c**2
    sqrt_term = np.sqrt(a2c2 - 1)

    x_Q1 = c + (-c + abs(a) * sqrt_term) / a2c2
    y_Q1 = (-c**2 + c * abs(a) * sqrt_term) / (a * a2c2) + 1/a

    x_Q2 = c + (-c - abs(a) * sqrt_term) / a2c2
    y_Q2 = (-c**2 - c * abs(a) * sqrt_term) / (a * a2c2) + 1/a

    return (x_Q1, y_Q1), (x_Q2, y_Q2)


def compute_Q3_Q4(x, y, c):
    """
    Calcula as coordenadas de Q3 e Q4 para um ponto E=(x,y).
    Q3 e Q4 são os pontos sobre o círculo tais que EQ3 e EQ4
    são tangentes ao círculo de centro C=(c,0) e raio 1.
    Restrições: y((c-x)²+y²) != 0 e y²+(c-x)²-1 >= 0
    """
    M1 = y**2
    M2 = x - c
    M4 = abs(y)
    M5 = M1 + M2**2 - 1
    M7 = M1 + M2**2
    M8 = y * M7

    sqrt_M5 = np.sqrt(M5)

    M6  = M2 - M4 * sqrt_M5
    M9  = M1 + M2 * M4 * sqrt_M5
    M10 = M2 + M4 * sqrt_M5
    M11 = M1 - M2 * M4 * sqrt_M5

    x_Q3 = c + M6 / M7
    y_Q3 = M9 / M8

    x_Q4 = c + M10 / M7
    y_Q4 = M11 / M8

    return (x_Q3, y_Q3), (x_Q4, y_Q4)


def compute_Q5_Q6(b, c):
    """
    Calcula as coordenadas de Q5 e Q6.
    Q5 e Q6 são os pontos sobre o círculo tais que BQ5 e BQ6
    são tangentes ao círculo de centro C=(c,0) e raio 1.
    Restrições: b(b²+c²) != 0 e (b²+c²-1) >= 0
    """
    b2c2 = b**2 + c**2
    sqrt_term = np.sqrt(b2c2 - 1)

    x_Q5 = c + (-c - abs(b) * sqrt_term) / b2c2
    y_Q5 = (-c**2 - c * abs(b) * sqrt_term) / (b * b2c2) + 1/b

    x_Q6 = c + (-c + abs(b) * sqrt_term) / b2c2
    y_Q6 = (-c**2 + c * abs(b) * sqrt_term) / (b * b2c2) + 1/b

    return (x_Q5, y_Q5), (x_Q6, y_Q6)


def compute_D(a, b, c, Q2):
    """
    Calcula as coordenadas do ponto D.
    D é o último ponto da mediatriz de AB que pertence à fronteira.
    """
    x_Q2, y_Q2 = Q2
    y_D = (a + b) / 2
    x_D = ((a - b) / 2) * x_Q2 / (a - y_Q2)

    return (x_D, y_D)