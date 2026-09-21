# Protocolo Experimental Acadêmico e Metodologia de Benchmark

## 1. Integridade Científica e Reprodutibilidade
Em conformidade com os mais rigorosos padrões de pesquisa em Machine Learning e as diretrizes do projeto:
*   **Dados Reais:** Nenhum dado sintético fictício é gerado; todas as trajetórias procedem da execução real do jogo Super Mario World.
*   **Fixação de Sementes (*Seed Reproducibility*):** Todos os experimentos utilizam sementes pseudo-aleatórias fixadas (`seed=42`, `1337`, `2026`) para inicialização de pesos em PyTorch, geração de minibatches e particionamento de conjuntos de dados.
*   **Hardware de Treinamento e Teste:**
    *   GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM, Driver 556.29, CUDA 12.5).
    *   CPU: AMD Ryzen / Intel x86_64 sob Windows 11.
    *   Framework: PyTorch 2.5.1 + CUDA 12.1.

---

## 2. Divisão e Particionamento dos Dados
As trajetórias completas extraídas do jogo serão particionadas em três subconjuntos mutualmente exclusivos:
1. **Conjunto de Treinamento (Train Split - 70%):** Utilizado para otimização dos parâmetros $\theta$ via gradiente descendente (otimizador AdamW).
2. **Conjunto de Validação (Validation Split - 15%):** Utilizado para *early stopping* e monitoramento de *overfitting*.
3. **Conjunto de Teste / Generalização (Test Split - 15%):**
   *   **In-Distribution (ID):** Trajetórias na mesma fase (ex: Yoshi's Island 2) não vistas durante o treino.
   *   **Out-of-Distribution (OOD):** Trechos com velocidades extremas, manobras de corrida contínua com P-meter ativo e saltos repetidos contra tetos.

---

## 3. Protocolo de Treinamento e Hiperparâmetros

| Hiperparâmetro | Valor Padrão | Observação |
| :--- | :--- | :--- |
| Otimizador | AdamW | Taxa de decaimento de peso $\text{weight\_decay} = 10^{-4}$ |
| Learning Rate ($\eta$) | $1 \times 10^{-3}$ | Com escalonador *Cosine Annealing* |
| Batch Size ($B$) | 128 / 256 | Ajustado para saturação ótima da GPU |
| Épocas Máximas | 100 | Com critério de parada prematura (*patience* = 15) |
| Coeficiente $\lambda_{\text{kin}}$ | 1.0 a 5.0 | Peso da penalidade cinemática na PINN |
| Coeficiente $\lambda_{\text{bound}}$ | 0.5 a 2.0 | Peso da violação de velocidades limites |
| Coeficiente $\lambda_{\text{contact}}$| 0.5 | Peso da condição de contato no solo |

---

## 4. Análise Estatística e Significância
Cada modelo será treinado através de múltiplas inicializações aleatórias (mínimo de 3 *runs* independentes). Os resultados serão reportados no formato:
$$\mu \pm \sigma$$
(Média $\pm$ Desvio Padrão) acompanhados de testes estatísticos de significância (teste t de Student pareado ou teste de postos sinalizados de Wilcoxon para verificar se a diferença entre PINN e MLP/LSTM é estatisticamente significativa com $p < 0.05$).
