# Especificação Técnica da Física e Memória RAM de Super Mario World (SNES)

## 1. Introdução e Arquitetura do Sistema
O jogo *Super Mario World* (Nintendo, 1990) roda sobre a CPU Ricoh 5A22 (baseada no microprocessador WDC 65C816 de 16-bits), operando a uma taxa de atualização nominal de aproximadamente 60 Hz (59.94 Hz NTSC). A cada quadro (*frame* $\Delta t \approx 16.67\text{ ms}$), a rotina de controle do jogador executa atualizações cinemáticas discretas baseadas em acumuladores de ponto fixo (*fixed-point*), contadores de estado e tabelas pré-computadas (*lookup tables*).

Para aplicações de Machine Learning sem visão computacional (*computer vision*), todas as grandezas de estado necessárias para descrição cinemática e dinâmica encontram-se mapeadas diretamente no espaço de endereçamento da memória RAM WRAM (`$7E:0000` a `$7F:FFFF`).

---

## 2. Mapa de Registradores de Estado do Jogador na RAM

| Endereço RAM | Tamanho / Formato | Nome Técnico | Descrição e Interpretação |
| :--- | :--- | :--- | :--- |
| `$7E:0094` - `$7E:0095` | 16-bit Unsigned Little-Endian | `Mario_X_Pos` | Posição horizontal do Mario em pixels no espaço de coordenadas da fase. Byte baixo em `$94`, byte alto em `$95`. |
| `$7E:0096` - `$7E:0097` | 16-bit Unsigned Little-Endian | `Mario_Y_Pos` | Posição vertical do Mario em pixels (cresce de cima para baixo). Byte baixo em `$96`, byte alto em `$97`. |
| `$7E:13DA` | 8-bit Unsigned | `Mario_X_Sub` | Subpixel horizontal. Varia de `0` a `15` ($0\text{x}0$ a $0\text{x}F$). Cada unidade representa $1/16$ de pixel. |
| `$7E:13DC` | 8-bit Unsigned | `Mario_Y_Sub` | Subpixel vertical. Varia de `0` a `15` ($0\text{x}0$ a $0\text{x}F$). Cada unidade representa $1/16$ de pixel. |
| `$7E:007B` | 8-bit Signed (Complemento de 2) | `Mario_X_Speed` | Velocidade horizontal em subpixels por quadro. Positivo indica movimento para a direita; negativo para a esquerda. |
| `$7E:007D` | 8-bit Signed (Complemento de 2) | `Mario_Y_Speed` | Velocidade vertical em subpixels por quadro. Positivo indica queda (para baixo); negativo indica ascensão/salto (para cima). |
| `$7E:0077` | 8-bit Flag Mask | `Mario_Blocked_Status` | Máscara de contato físico com o cenário. Formato de bits: `SxxMUDLR`. Bit 0 (R): parede direita; Bit 1 (L): parede esquerda; Bit 2 (D): chão/solo; Bit 3 (U): teto/bloqueio superior. |
| `$7E:0072` | 8-bit Enum | `Mario_Air_State` | Estado no ar: `0` = no chão, `1` = caindo, `2` = saltando. |
| `$7E:0071` | 8-bit Enum | `Mario_Animation_State`| Estado de animação do jogador: `0` = normal, `9` = animação de morte, etc. |
| `$7E:0019` | 8-bit Enum | `Mario_Powerup` | `0` = Pequeno, `1` = Super Mario, `2` = Capa, `3` = Flor de Fogo. |
| `$7E:0015` | 8-bit Flag Mask | `Controller_Hold` | Botões mantidos pressionados: `BYsSUDLR` (B, Y, Select, Start, Cima, Baixo, Esquerda, Direita). |
| `$7E:0016` | 8-bit Flag Mask | `Controller_Press`| Botões recém-pressionados no quadro atual (*frame trigger*). |
| `$7E:0017` | 8-bit Flag Mask | `Controller_Hold_2`| Botões complementares: `AXLR----` (A, X, L, R). |

---

## 3. Mecânica Cinemática e Equações de Movimento

### 3.1 Posição em Ponto Fixo (Subpixels)
Em Super Mario World, 1 pixel equivale a 16 subpixels:
$$X_{\text{total}} = \text{Mario\_X\_Pos} + \frac{\text{Mario\_X\_Sub}}{16}$$
$$Y_{\text{total}} = \text{Mario\_Y\_Pos} + \frac{\text{Mario\_Y\_Sub}}{16}$$

A cada quadro, o deslocamento é calculado pela adição da velocidade correspondente:
$$16 \cdot X_{\text{total}, t+1} = 16 \cdot X_{\text{total}, t} + \text{Mario\_X\_Speed}_t$$
$$X_{\text{total}, t+1} = X_{\text{total}, t} + \frac{\text{Mario\_X\_Speed}_t}{16}$$

Isso estabelece uma **relação analítica exata e invariante** entre velocidade e posição.

### 3.2 Dinâmica de Movimento Horizontal
1. **Velocidade Máxima de Caminhada:** $\pm 20$ subpixels/frame ($\approx 1.25\text{ pixels/frame}$).
2. **Velocidade Máxima de Corrida (Segurando Botão Y ou X):** $\pm 48$ subpixels/frame ($3.0\text{ pixels/frame}$).
3. **Velocidade Máxima de Sprint (P-Meter cheio):** $\pm 72$ subpixels/frame ($4.5\text{ pixels/frame}$).
4. **Atrito e Derrapagem (*Skidding*):**
   * Ao soltar o direcional no solo, o atrito desacelera gradualmente o Mario até zero.
   * Se o jogador inverter a direção enquanto corre, entra no estado de derrapagem (*skid*), aplicando uma desaceleração superior.

### 3.3 Dinâmica de Salto e Gravidade
1. **Impulso Inicial de Salto:**
   * Botão B (Salto Padrão): $v_{y,0} \approx -64$ a $-80$ subpixels/frame (dependendo da velocidade horizontal prévia).
   * Botão A (Salto Giratório / *Spin Jump*): menor impulso vertical inicial.
2. **Aceleração Gravitacional Modulada por Entrada:**
   * Enquanto o botão de salto (A ou B) for mantido pressionado na subida:
     $$g_{\text{held}} \approx +3\text{ subpixels/frame}^2$$
   * Quando o botão de salto for solto prematuramente ou na descida:
     $$g_{\text{released}} \approx +6\text{ subpixels/frame}^2$$
   * Essa assimetria intencional confere a dinâmica de salto variável característica da jogabilidade de plataforma da Nintendo.
3. **Velocidade Terminal de Queda:**
   A velocidade vertical descendente é limitada superiormente (*terminal velocity*):
   $$v_{y} \le +64\text{ subpixels/frame } (+4.0\text{ pixels/frame})$$
   (ou $+72$ em certas condições de voo com capa).

### 3.4 Condições de Contorno e Colisão
1. **Contato com o Solo ($c_{\text{ground}} = 1$):**
   Se o pé do Mario atinge um bloco sólido:
   * $v_y$ é forçada a $0$.
   * $Y_{\text{pos}}$ é alinhada com a superfície do bloco.
2. **Contato com o Teto ($c_{\text{ceiling}} = 1$):**
   Se a cabeça atinge um bloco sólido por baixo com $v_y < 0$:
   * $v_y$ é imediatamente anulada ou invertida para valor descendente positivo.
3. **Paredes Laterais ($c_{\text{left}} = 1$ ou $c_{\text{right}} = 1$):**
   * Se $v_x$ for em direção à parede sólida, $v_x$ é zerada.

---

## 4. Implicações para Modelagem por Redes Neurais (PIML vs. Estatística)
Modelos puramente empíricos (como MLPs ou LSTMs sem física) precisam inferir todas essas regras estritamente a partir de dados amostrados. Quando colocados em previsão multi-passo autorregressiva:
* Pequenos erros numéricos em $\hat{v}_{x}$ e $\hat{v}_{y}$ acumulam-se em divergência rápida de $\hat{X}$ e $\hat{Y}$ (*trajectory drift*).
* Modelos estatísticos podem predizer que o Mario ultrapassa a velocidade terminal de 64 subpixels/frame ou "afunde" através do solo sólido.
* Em regimes com poucos dados de treinamento (*low-data regime*), redes estatísticas falham em inferir que $\Delta X$ é rigorosamente linear em relação a $v_x$.
* Uma Rede Neural Informada pela Física (PINN) ou com restrições arquiteturais indutivas elimina essa classe de erros por projeto.
