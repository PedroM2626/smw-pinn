# Formulação Matemática: PINN Discreta vs. Redes Estatísticas em Jogos 2D

## 1. Definição Formal do Problema

Seja um sistema dinâmico híbrido não-linear em tempo discreto indexado por $t \in \mathbb{N}_0$, com intervalo de amostragem constante $\Delta t = 1$ quadro ($1/60\text{ s}$):

$$s_{t+1} = \mathcal{F}(s_t, a_t)$$

onde:
*   $s_t \in \mathcal{S} \subseteq \mathbb{R}^d$ é o vetor de estado extraído da memória RAM do jogo no quadro $t$.
*   $a_t \in \mathcal{A} \subseteq \{0, 1\}^m$ é o vetor de entradas discretas do controle no quadro $t$.
*   $\mathcal{F}$ representa a função de transição de estados regida pela rotina de física da ROM do SNES.

O estado do jogador é parametrizado por:
$$s_t = \begin{bmatrix} X_t & Y_t & v_{x,t} & v_{y,t} & c_{t, \text{ground}} & c_{t, \text{ceiling}} & c_{t, \text{left}} & c_{t, \text{right}} \end{bmatrix}^T$$
onde $X_t, Y_t$ são coordenadas contínuas normalizadas em pixels ($X = x_{\text{pix}} + \frac{sx}{16}$), $v_{x,t}, v_{y,t}$ são velocidades normalizadas, e $c_t$ indica contato com superfícies sólidas.

O objetivo do modelo preditivo $f_\theta$ (parametrizado por pesos $\theta$) é mapear $(s_t, a_t)$ para o estado seguinte $\hat{s}_{t+1}$.

---

## 2. Modelos Estatísticos Puros (Baselines Sem Física)

### 2.1 Perceptron Multicamadas (MLP Estatística)
A MLP atua como um aproximador universal de funções caixa-preta:

$$\hat{s}_{t+1} = f_{\theta_{\text{MLP}}}(s_t, a_t)$$

A otimização minimiza a perda empírica supervisionada sobre um conjunto de transições observadas $\mathcal{D} = \{(s_t^{(i)}, a_t^{(i)}, s_{t+1}^{(i)})\}_{i=1}^N$:

$$\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N} \sum_{i=1}^N \|\hat{s}_{t+1}^{(i)} - s_{t+1}^{(i)}\|_2^2$$

**Limitação Teórica:** A rede não possui qualquer garantia de que a relação cinemática $\Delta X = v_x \Delta t$ seja respeitada, nem de que as leis de conservação ou limites físicos de aceleração/velocidade sejam cumpridos.

### 2.2 Rede Recorrente (LSTM Temporal)
Para capturar inércia ou histórico de aceleração implicitamente:

$$h_t, c_t = \text{LSTM}( [s_t, a_t], (h_{t-1}, c_{t-1}) )$$
$$\hat{s}_{t+1} = W_o h_t + b_o$$

O treinamento utiliza a mesma perda empírica $\mathcal{L}_{\text{data}}$.

---

## 3. Rede Neural Informada pela Física (PINN com Restrição Suave)

Inspirado no paradigma clássico de Raissi et al. (2019) e adaptado para sistemas dinâmicos discretos (Karniadakis et al., 2021; Banerjee et al., 2023), a PINN formula o aprendizado através de um problema de otimização multiobjetivo regularizado:

$$\min_{\theta} \quad \mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda_{\text{kin}} \mathcal{L}_{\text{kin}}(\theta) + \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(\theta) + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}}(\theta)$$

### 3.1 Perda Cinemática Euleriana ($\mathcal{L}_{\text{kin}}$)
A física do jogo prescreve que o deslocamento por quadro é a velocidade dividida por 16 (subpixels por pixel):

$$R_x(s_t, \hat{s}_{t+1}) = \left(\hat{X}_{t+1} - X_t\right) - \frac{v_{x,t}}{16}$$
$$R_y(s_t, \hat{s}_{t+1}) = \left(\hat{Y}_{t+1} - Y_t\right) - \frac{v_{y,t}}{16}$$

$$\mathcal{L}_{\text{kin}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left( R_x^{(i)2} + R_y^{(i)2} \right)$$

Esta penalidade força a rede a aprender que a posição no quadro seguinte é um integrador exato da velocidade atual.

### 3.2 Perda de Limites de Velocidade Física ($\mathcal{L}_{\text{bound}}$)
O motor do jogo restringe estritamente as velocidades a limites operacionais:
*   Velocidade terminal de queda: $v_{y} \le v_{y,\text{term}} = 64$ subpixels/frame.
*   Velocidade máxima horizontal: $|v_{x}| \le v_{x,\max} = 72$ subpixels/frame.

$$\mathcal{L}_{\text{bound}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[ \max(0, |\hat{v}_{x,t+1}^{(i)}| - v_{x,\max})^2 + \max(0, \hat{v}_{y,t+1}^{(i)} - v_{y,\text{term}})^2 \right]$$

### 3.3 Perda de Invariância de Contato e Suporte ($\mathcal{L}_{\text{contact}}$)
Se o jogador está firmemente em contato com o chão ($c_{t,\text{ground}} = 1$) e nenhuma ação de salto é comandada ($a_{t, \text{jump}} = 0$):

$$\mathcal{L}_{\text{contact}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(c_{t,\text{ground}}^{(i)} = 1 \land a_{t,\text{jump}}^{(i)} = 0) \cdot \left( \hat{v}_{y,t+1}^{(i)} \right)^2$$

---

## 4. Rede com Viés Indutivo Forte (PINN Residual / Arquitetura Híbrida)

Em vez de permitir que a rede preveja a posição e depois penalizar desvios (restrição suave), a arquitetura com viés indutivo forte constrói a cinemática diretamente na computação gráfica do grafo de tensores:

$$\hat{\alpha}_{x,t}, \hat{\alpha}_{y,t} = \text{NN}_{\text{force}}(s_t, a_t)$$

$$\hat{v}_{x,t+1} = \text{clamp}(v_{x,t} + \hat{\alpha}_{x,t}, -v_{x,\max}, v_{x,\max})$$
$$\hat{v}_{y,t+1} = \begin{cases}
  0 & \text{se } c_{t,\text{ground}} = 1 \text{ e } a_{t,\text{jump}} = 0 \\
  \min(v_{y,t} + \hat{\alpha}_{y,t}, v_{y,\text{term}}) & \text{caso contrário}
\end{cases}$$

$$\hat{X}_{t+1} = X_t + \frac{\hat{v}_{x,t+1}}{16}$$
$$\hat{Y}_{t+1} = Y_t + \frac{\hat{v}_{y,t+1}}{16}$$

**Propriedade Teórica:** Por construção estrutural, o resíduo cinemático $\|R_x\|_2 + \|R_y\|_2 \equiv 0$ para qualquer configuração de pesos $\theta$. A rede neural é responsável unicamente por aprender as acelerações e forças resultantes não-triviais (atrito de solo, derrapagem, colisão com blocos).

---

## 5. Formalização das Métricas de Comparação

1. **Erro de Predição de 1 Passo (One-step MSE / MAE):**
   $$\text{MSE}(k) = \frac{1}{N} \sum_{i=1}^N (\hat{s}_{t+1, k}^{(i)} - s_{t+1, k}^{(i)})^2$$

2. **Divergência de Trajetória Multi-passo (Rollout Drift em Horizonte $H$):**
   Dada a sequência de ações reais $(a_t, a_{t+1}, \dots, a_{t+H-1})$ e o estado inicial real $s_t$:
   $$\hat{s}_{t+\tau+1} = f_\theta(\hat{s}_{t+\tau}, a_{t+\tau}), \quad \tau = 0, \dots, H-1$$
   $$\text{Drift}(H) = \frac{1}{H} \sum_{\tau=1}^H \sqrt{ (\hat{X}_{t+\tau} - X_{t+\tau})^2 + (\hat{Y}_{t+\tau} - Y_{t+\tau})^2 }$$

3. **Taxa de Violação Física ($\nu_{\text{phys}}$):**
   $$\nu_{\text{phys}} = \frac{1}{N_{\text{eval}}} \sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\left( |\hat{v}_x| > v_{x,\max} \lor \hat{v}_y > v_{y,\text{term}} \lor \text{Penetração Inadmissível} \right)$$

4. **Curva de Eficiência Amostral (Sample Efficiency Pareto):**
   $$\text{Desempenho}(N) \quad \text{para} \quad N \in \{200, 500, 1000, 5000, 20000\}$$
