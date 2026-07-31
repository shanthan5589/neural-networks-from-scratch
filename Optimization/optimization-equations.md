# Optimization Algorithms — Equations Reference

## Exponentially Weighted Average (EWA)

```math
v_t = \beta \, v_{t-1} + (1 - \beta) \, \theta_t
```

Unrolled:

```math
v_t = (1-\beta)\theta_t + (1-\beta)\beta\,\theta_{t-1} + (1-\beta)\beta^2\,\theta_{t-2} + \cdots
```

Weight of an observation $k$ steps back from the current step: $(1-\beta)\beta^k$ 

Weight of $\theta_{t-k}$: $(1-\beta)\beta^k$

Per-step Decay Ratio between consecutive observations: $\beta$

Effective window: $\approx \dfrac{1}{1-\beta}$

Initialize: $v_0 = 0$

---

## Bias Correction

At step $t$, weights on real observations sum to $1 - \beta^t$, not $1$.

Corrected estimate:

```math
\hat{v}_t = \frac{v_t}{1 - \beta^t}
```

As $t \to \infty$: $\beta^t \to 0$, correction $\to 1$.

---

## Momentum (without Bias Correction)

Complete info: [Momentum and Bias Correction](./momentum_and_bias_correction.md)

Refer: [Momentum based optimizers - GFG](https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/)

The idea is we don't use simple average like $(w1 + w2 + w3)/3$ because in that case we are giving equal weight to all the previous gradients. Instead, we use exponentially weighted average which gives more weight to the recent gradients and less weight to the older gradients. 

Suppose average gradient is moving in +ve direction and suppose the next batch unexpectedly produces the opposite gradient, -1, The model still moves in the original direction, but more slowly. If several subsequent gradients point the opposite way, the direction (velocity) will eventually reverse.

EWA applied to gradients.

```math
v_{dW} = \beta \, v_{dW} + (1 - \beta) \, dW
```

```math
v_{db} = \beta \, v_{db} + (1 - \beta) \, db
```

Where $v_{dW}$ and $v_{db}$ are the running exponentially weighted averages of the gradients $dW$ and $db$ respectively.

Example (with $\beta = 0.9$, $v_{dW_0}=0$):

```math
v_{dW_1} = 0.9 \, v_{dW_0} + 0.1 \, dW_{0}
```

```math
v_{dW_2} = 0.9 \, v_{dW_1} + 0.1 \, dW_{1}
```

```math
v_{dW_3} = 0.9 \, v_{dW_2} + 0.1 \, dW_{2}
```

All the above equations result in:

```math
v_{dW_3} = 0.1 \, dW_{2} + 0.09 \, dW_{1} + 0.081 \, dW_{0}
```

The weights will sum to $0.1 + 0.09 + 0.081 = 0.271$, which is less than $1$. This is because we are not considering the initial value of $v_{dW_0}$, which is $0$. The weights on the real observations sum to $1 - \beta^t$, not $1$.

Update:

```math
W = W - \alpha \, v_{dW}
```

```math
b = b - \alpha \, v_{db}
```

Initialize: $v_{dW} = 0$, $v_{db} = 0$

Typical: $\beta = 0.9$

---

## RMSprop (without Bias Correction)

EWA applied to squared gradients.

```math
s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2) \, dW^2
```

```math
s_{db} = \beta_2 \, s_{db} + (1 - \beta_2) \, db^2
```

Update:

```math
W = W - \alpha \, \frac{dW}{\sqrt{s_{dW}} + \epsilon}
```

```math
b = b - \alpha \, \frac{db}{\sqrt{s_{db}} + \epsilon}
```

$dW^2$ is element-wise squaring. Division and square root are also element-wise.

Initialize: $s_{dW} = 0$, $s_{db} = 0$

Typical: $\epsilon = 10^{-8}$

---

## Adam (Momentum + RMSprop + Bias Correction)

First moment (momentum):

```math
v_{dW} = \beta_1 \, v_{dW} + (1 - \beta_1) \, dW
```

```math
v_{db} = \beta_1 \, v_{db} + (1 - \beta_1) \, db
```

Second moment (RMSprop):

```math
s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2) \, dW^2
```

```math
s_{db} = \beta_2 \, s_{db} + (1 - \beta_2) \, db^2
```

Bias correction:

```math
\hat{v}_{dW} = \frac{v_{dW}}{1 - \beta_1^t}, \quad \hat{v}_{db} = \frac{v_{db}}{1 - \beta_1^t}
```

```math
\hat{s}_{dW} = \frac{s_{dW}}{1 - \beta_2^t}, \quad \hat{s}_{db} = \frac{s_{db}}{1 - \beta_2^t}
```

Update:

```math
W = W - \alpha \, \frac{\hat{v}_{dW}}{\sqrt{\hat{s}_{dW}} + \epsilon}
```

```math
b = b - \alpha \, \frac{\hat{v}_{db}}{\sqrt{\hat{s}_{db}} + \epsilon}
```

Typical: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## Learning Rate Decay

```math
\alpha = \frac{1}{1 + \text{decay rate} \times \text{epoch}} \cdot \alpha_0
```

Other forms:

```math
\alpha = 0.95^{\text{epoch}} \cdot \alpha_0 \quad \text{(exponential decay)}
```

```math
\alpha = \frac{k}{\sqrt{\text{epoch}}} \cdot \alpha_0
```
