# Optimization Algorithms — Equations Reference

## Exponentially Weighted Average (EWA)

$$v_t = \beta \, v_{t-1} + (1 - \beta) \, \theta_t$$

Unrolled:

$$v_t = (1-\beta)\theta_t + (1-\beta)\beta\,\theta_{t-1} + (1-\beta)\beta^2\,\theta_{t-2} + \cdots$$

Weight of an observation $k$ steps back from the current step: $(1-\beta)\beta^k$ 

Weight of $\theta_{t-k}$: $(1-\beta)\beta^k$

Per-step Decay Ratio between consecutive observations: $\beta$

Effective window: $\approx \dfrac{1}{1-\beta}$

Initialize: $v_0 = 0$

---

## Bias Correction

At step $t$, weights on real observations sum to $1 - \beta^t$, not $1$.

Corrected estimate:

$$\hat{v}_t = \frac{v_t}{1 - \beta^t}$$

As $t \to \infty$: $\beta^t \to 0$, correction $\to 1$.

---

## Momentum (without Bias Correction)

EWA applied to gradients.

$$v_{dW} = \beta \, v_{dW} + (1 - \beta) \, dW$$

$$v_{db} = \beta \, v_{db} + (1 - \beta) \, db$$

Update:

$$W = W - \alpha \, v_{dW}$$

$$b = b - \alpha \, v_{db}$$

Initialize: $v_{dW} = 0$, $v_{db} = 0$

Typical: $\beta = 0.9$

---

## RMSprop (without Bias Correction)

EWA applied to squared gradients.

$$s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2) \, dW^2$$

$$s_{db} = \beta_2 \, s_{db} + (1 - \beta_2) \, db^2$$

Update:

$$W = W - \alpha \, \frac{dW}{\sqrt{s_{dW}} + \epsilon}$$

$$b = b - \alpha \, \frac{db}{\sqrt{s_{db}} + \epsilon}$$

$dW^2$ is element-wise squaring. Division and square root are also element-wise.

Initialize: $s_{dW} = 0$, $s_{db} = 0$

Typical: $\epsilon = 10^{-8}$

---

## Adam (Momentum + RMSprop + Bias Correction)

First moment (momentum):

$$v_{dW} = \beta_1 \, v_{dW} + (1 - \beta_1) \, dW$$

$$v_{db} = \beta_1 \, v_{db} + (1 - \beta_1) \, db$$

Second moment (RMSprop):

$$s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2) \, dW^2$$

$$s_{db} = \beta_2 \, s_{db} + (1 - \beta_2) \, db^2$$

Bias correction:

$$\hat{v}_{dW} = \frac{v_{dW}}{1 - \beta_1^t}, \quad \hat{v}_{db} = \frac{v_{db}}{1 - \beta_1^t}$$

$$\hat{s}_{dW} = \frac{s_{dW}}{1 - \beta_2^t}, \quad \hat{s}_{db} = \frac{s_{db}}{1 - \beta_2^t}$$

Update:

$$W = W - \alpha \, \frac{\hat{v}_{dW}}{\sqrt{\hat{s}_{dW}} + \epsilon}$$

$$b = b - \alpha \, \frac{\hat{v}_{db}}{\sqrt{\hat{s}_{db}} + \epsilon}$$

Typical: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## Learning Rate Decay

$$\alpha = \frac{1}{1 + \text{decay rate} \times \text{epoch}} \cdot \alpha_0$$

Other forms:

$$\alpha = 0.95^{\text{epoch}} \cdot \alpha_0 \quad \text{(exponential decay)}$$

$$\alpha = \frac{k}{\sqrt{\text{epoch}}} \cdot \alpha_0$$
