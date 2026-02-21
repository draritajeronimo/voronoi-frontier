# Voronoi Frontier — Dois Sites e um Obstáculo Circular

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)]()

> **Código de referência** para o manuscrito:
> *"Solving a path planning problem using the frontier of Voronoi diagram
> with two sites and a circular obstacle"*
> Submetido ao periódico *Computer-Aided Design*, fevereiro de 2023.
>
> **Autora:** Rita Jeronimo — UFPB
> **Contato:** draritajeronimo@github.com

---

## Sobre o Projeto

Este repositório implementa de forma **analítica e computacional** a fronteira
do diagrama de Voronoi para a seguinte configuração:

- Dois pontos geradores: **A = (0, a)** e **B = (0, b)**
- Um obstáculo circular centrado em **C = (c, 0)** com raio unitário (r = 1)

O algoritmo determina a fronteira **sem aproximações** (sem convex hull),
usando coordenadas exatas dos pontos auxiliares e o método de Newton-Raphson.

---

## Estrutura do Repositório
```
voronoi-frontier/
│
├── voronoi_frontier/          # Pacote principal
│   ├── __init__.py            # Exportações públicas do pacote
│   ├── core.py                # Algoritmo, geometria, distâncias e derivadas
│   └── plot.py                # Geração das figuras
│
├── tests/
│   └── test_voronoi.py        # 30 testes unitários (pytest)
│
├── examples/
│   └── run_example.py         # Exemplo completo que gera todas as figuras
│
├── requirements.txt           # Dependências Python
├── LICENSE                    # Licença MIT
└── README.md                  # Este arquivo
```

---

## Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip

### Passos
```bash
# 1. Clone o repositório
git clone https://github.com/draritajeronimo/voronoi-frontier.git
cd voronoi-frontier

# 2. Instale as dependências
pip install -r requirements.txt
```

---

## Uso Rápido
```python
from voronoi_frontier import compute_frontier, plot_frontier

# Parâmetros: a > 0, b < 0, -b > a, 0 < c < 1
xs, ys, key_pts, info = compute_frontier(a=2.0, b=-3.0, c=0.5)

fig = plot_frontier(xs, ys, a=2.0, b=-3.0, c=0.5, key_points=key_pts)
fig.show()
```

Para rodar o exemplo completo e gerar todas as figuras:
```bash
python examples/run_example.py
```

---

## Fundamentos Matemáticos

### Restrições dos Parâmetros (Seção 3 do artigo)

| Parâmetro | Significado                    | Restrição          |
|-----------|--------------------------------|--------------------|
| a         | Coordenada y do ponto A        | a > 0              |
| b         | Coordenada y do ponto B        | b < 0, -b > a      |
| c         | Coordenada x do centro C       | 0 < c < 1          |
| r         | Raio do obstáculo circular     | fixado em r = 1    |

### Pontos Auxiliares Principais

| Ponto   | Definição                                              |
|---------|--------------------------------------------------------|
| Q1, Q2  | Pontos de tangência de A ao círculo (Proposição 1)     |
| Q5, Q6  | Pontos de tangência de B ao círculo (Proposição 4)     |
| D       | Último ponto da bissetriz com visada direta a A e B    |
| E=(x,y) | Ponto móvel da fronteira (y calculado por Newton-Raphson)|

### Duas Situações Durante o Percurso da Fronteira

**Situação 1** — Pi tem visada direta a B:
```
D(Pi) = d(A,Q2) + ||Q2Q3|| + ||Q3Pi|| - d(Pi,B) = 0
```

**Situação 2** — Obstáculo bloqueia a visada para A e B:
```
D(Pi) = d(A,Q2) + ||Q2Q3|| + ||Q3Pi|| - [||PiQ4|| + ||Q4Q5|| + d(Q5,B)] = 0
```

### Correção do Algoritmo Original

O pseudocódigo original do artigo continha um erro lógico no critério de parada
do Newton-Raphson (uso de OR em vez de AND). Este repositório apresenta a versão corrigida:
```python
# ERRADO — continua iterando mesmo após convergência
while j < max_iter OR |F(y)| <= epsilon:

# CORRETO — para quando ambas as condições são satisfeitas
while j < max_iter AND |F(y)| > epsilon:
```

---

## Testes
```bash
pytest tests/ -v
```

Cobertura dos testes (30 testes, 7 classes):

| Classe de Teste          | O que é verificado                                  |
|--------------------------|-----------------------------------------------------|
| TestValidation           | Restrições dos parâmetros de entrada                |
| TestKeyPoints            | Q1–Q5 sobre o círculo, tangência, D na bissetriz    |
| TestDistances            | Proposições 5 e 10 vs. distância euclidiana direta  |
| TestDerivatives          | dF/dy analítica vs. diferenças finitas centradas    |
| TestNewtonRaphson        | Equidistância, monotonicidade, convergência         |
| TestMembership           | Testes de pertencimento ao círculo e à reta         |
| TestRobustness           | Múltiplas configurações (a, b, c)                   |

---

## Reprodutibilidade

Todas as figuras do manuscrito podem ser reproduzidas executando:
```bash
python examples/run_example.py
```

Figuras geradas:

| Arquivo                      | Conteúdo                                      |
|------------------------------|-----------------------------------------------|
| examples/frontier_single.png | Fronteira principal (configuração padrão)     |
| examples/convergence.png     | Contagem de iterações do Newton-Raphson       |
| examples/frontier_multiple.png | Comparação entre múltiplas configurações    |

---

## Citação

Se utilizar este código em trabalho acadêmico, cite:
```bibtex
@article{jeronimo2023voronoi,
  author  = {Jeronimo, Rita},
  title   = {Solving a path planning problem using the frontier of
             Voronoi diagram with two sites and a circular obstacle},
  journal = {Computer-Aided Design},
  year    = {2023},
  note    = {Preprint. Universidade Federal da Paraíba (UFPB)}
}
```

---

## Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.
