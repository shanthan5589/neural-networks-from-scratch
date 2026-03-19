# Batch Normalization — Equations Reference

## Forward Pass

Applied per layer, per neuron, across all $m$ examples in a mini-batch. Applied to $z$ (before activation function).

**Step 1 — Mean:**

$$\mu = \frac{1}{m} \sum_{i=1}^{m} z^{(i)}$$

**Step 2 — Variance:**

$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (z^{(i)} - \mu)^2$$

**Step 3 — Normalize:**

$$\hat{z}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

**Step 4 — Scale and shift:**

$$\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$$

$\gamma$ and $\beta$ are learnable parameters (per neuron), updated through backprop.

$\epsilon$ is a small constant (typically $10^{-8}$) to avoid division by zero.

Bias $b$ is dropped from the layer because batch norm subtracts the mean, which cancels $b$. The $\beta$ parameter replaces it.

Per-layer learnable parameters: $(W, \gamma, \beta)$ instead of $(W, b)$.

---

## Why $b$ Disappears

We have $z^{(i)} = W a^{(i)} + b$. When batch norm subtracts the mean:

$$\mu = \frac{1}{m}\sum(W a^{(i)} + b) = \frac{1}{m}\sum W a^{(i)} + b$$

$b$ is a constant across all examples, so it passes straight through the average. Now subtract:

$$z^{(i)} - \mu = (W a^{(i)} + b) - \left(\frac{1}{m}\sum W a^{(i)} + b\right)$$

The $+b$ and $+b$ cancel. No matter what value $b$ has, subtracting the mean removes it. So there's no point learning $b$ when batch norm erases it.

Batch norm's $\beta$ in step 4 replaces $b$ — it adds a learnable shift after normalization, where it won't get cancelled.

---

## Shorthand

$$s = \frac{1}{\sqrt{\sigma^2 + \epsilon}} = (\sigma^2 + \epsilon)^{-1/2}$$

---

## Computation Graph for Backward Pass

$$\gamma \hat{z}^{(i)} + \beta = \tilde{z}^{(i)}$$

$$g(\tilde{z}^{(i)}) = a^{(i)}$$

$$z^{(i)} \rightarrow \mu, \; \sigma^2 \rightarrow \hat{z}^{(i)} \rightarrow \gamma \hat{z}^{(i)} + \beta = \tilde{z}^{(i)} \rightarrow g(\tilde{z}^{(i)}) = a^{(i)} \rightarrow \mathcal{L}$$

Intermediate variables (each one is a simple node):

$$\mu = \frac{1}{m}\sum z^{(i)}$$

$$v = \sigma^2 = \frac{1}{m}\sum (z^{(i)} - \mu)^2$$

$$s = (v + \epsilon)^{-1/2}$$

$$\hat{z}^{(i)} = (z^{(i)} - \mu) \cdot s$$

$$\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$$

---

## Backward Pass — Step by Step

### Step 1: Start from the loss

We have $\frac{\partial \mathcal{L}}{\partial \tilde{z}^{(i)}} = d\tilde{z}^{(i)}$ from upstream.

For the output layer with sigmoid + binary cross entropy, this simplifies to $A_2^{(i)} - y^{(i)}$.

---

### Step 2: Gradient for $\gamma$

$\gamma$ has no direct connection to loss. $\gamma$ affects $\tilde{z}$, and $\tilde{z}$ affects loss. So chain rule:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \frac{\partial \mathcal{L}}{\partial \tilde{z}^{(i)}} \cdot \frac{\partial \tilde{z}^{(i)}}{\partial \gamma}$$

We already have $\frac{\partial \mathcal{L}}{\partial \tilde{z}^{(i)}} = d\tilde{z}^{(i)}$ from Step 1.

Now find $\frac{\partial \tilde{z}^{(i)}}{\partial \gamma}$. From $\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$:

$$\frac{\partial \tilde{z}^{(i)}}{\partial \gamma} = \hat{z}^{(i)}$$

So for a single example: $\frac{\partial \mathcal{L}}{\partial \gamma} = d\tilde{z}^{(i)} \cdot \hat{z}^{(i)}$

But $\gamma$ is a single parameter shared across all $m$ examples. Each example contributes a gradient, so we sum:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} d\tilde{z}^{(i)} \cdot \hat{z}^{(i)}$$

---

### Step 3: Gradient for $\beta$

Same logic. $\beta$ affects $\tilde{z}$, $\tilde{z}$ affects loss:

$$\frac{\partial \mathcal{L}}{\partial \beta} = \frac{\partial \mathcal{L}}{\partial \tilde{z}^{(i)}} \cdot \frac{\partial \tilde{z}^{(i)}}{\partial \beta}$$

From $\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$:

$$\frac{\partial \tilde{z}^{(i)}}{\partial \beta} = 1$$

So for a single example: $\frac{\partial \mathcal{L}}{\partial \beta} = d\tilde{z}^{(i)} \cdot 1 = d\tilde{z}^{(i)}$

Again $\beta$ is shared across all $m$ examples, so sum:

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} d\tilde{z}^{(i)}$$

---

### Step 4: Gradient for $\hat{z}^{(i)}$

$\hat{z}$ affects $\tilde{z}$, $\tilde{z}$ affects loss:

$$\frac{\partial \mathcal{L}}{\partial \hat{z}^{(i)}} = \frac{\partial \mathcal{L}}{\partial \tilde{z}^{(i)}} \cdot \frac{\partial \tilde{z}^{(i)}}{\partial \hat{z}^{(i)}}$$

From $\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$:

$$\frac{\partial \tilde{z}^{(i)}}{\partial \hat{z}^{(i)}} = \gamma$$

Multiply:

$$\frac{\partial \mathcal{L}}{\partial \hat{z}^{(i)}} = d\tilde{z}^{(i)} \cdot \gamma$$

---

### Step 5: Gradient $d\hat{z}/dz$ — the hard part

We need $\frac{\partial \mathcal{L}}{\partial z^{(i)}}$. Loss has no direct connection to $z^{(i)}$ — it goes through $\hat{z}^{(i)}$. So chain rule:

$$\frac{\partial \mathcal{L}}{\partial z^{(i)}} = \frac{\partial \mathcal{L}}{\partial \hat{z}^{(i)}} \cdot \frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}}$$

We already have $\frac{\partial \mathcal{L}}{\partial \hat{z}^{(i)}} = d\tilde{z}^{(i)} \cdot \gamma$ from Step 4.

Now we need $\frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}}$. Look at the equation:

$$\hat{z}^{(i)} = (z^{(i)} - \mu) \cdot s$$

$z^{(i)}$ affects $\hat{z}^{(i)}$ through **three paths** because $z^{(i)}$ appears:

1. Directly in $(z^{(i)} - \mu)$
2. Inside $\mu = \frac{1}{m}\sum z^{(j)}$
3. Inside $v = \frac{1}{m}\sum(z^{(j)} - \mu)^2$, which feeds into $s$

---

#### Path 1 — Direct

Freeze $\mu$ and $s$ as constants.

$\hat{z}^{(i)} = s \cdot z^{(i)} - s \cdot \mu$

$$\frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} = s$$

---

#### Path 2 — Through $\mu$

**Link 1:** How does $z^{(i)}$ affect $\mu$?

$$\mu = \frac{1}{m}\sum z^{(j)} \implies \frac{\partial \mu}{\partial z^{(i)}} = \frac{1}{m}$$

**Link 2:** How does $\mu$ affect $\hat{z}^{(i)}$?

$\hat{z}^{(i)} = (z^{(i)} - \mu) \cdot s$, so $\mu$ is subtracted:

$$\frac{\partial \hat{z}^{(i)}}{\partial \mu} = -s$$

**Chaining Link 1 and Link 2:**

$$\frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} \bigg|_{\text{path 2}} = \frac{\partial \hat{z}^{(i)}}{\partial \mu} \cdot \frac{\partial \mu}{\partial z^{(i)}} = (-s) \cdot \frac{1}{m} = -\frac{s}{m}$$

---

#### Path 3 — Through $v$ through $s$

Three links in the chain:

**Link 1:** How does $z^{(i)}$ affect $v$? (with $\mu$ frozen — its dependency is handled by path 2)

$$v = \frac{1}{m}\sum(z^{(j)} - \mu)^2$$

Only the $j = i$ term contains $z^{(i)}$. Power rule:

$$\frac{\partial v}{\partial z^{(i)}} = \frac{1}{m} \cdot 2(z^{(i)} - \mu) = \frac{2}{m}(z^{(i)} - \mu)$$

**Link 2:** How does $v$ affect $s$?

$$s = (v + \epsilon)^{-1/2}$$

Power rule:

$$\frac{\partial s}{\partial v} = -\frac{1}{2}(v + \epsilon)^{-3/2}$$

**Link 3:** How does $s$ affect $\hat{z}^{(i)}$?

$\hat{z}^{(i)} = (z^{(i)} - \mu) \cdot s$. With everything except $s$ frozen:

$$\frac{\partial \hat{z}^{(i)}}{\partial s} = (z^{(i)} - \mu)$$

**Chaining Link 1, Link 2, and Link 3:**

$$\frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} \bigg|_{\text{path 3}} = \frac{\partial \hat{z}^{(i)}}{\partial s} \cdot \frac{\partial s}{\partial v} \cdot \frac{\partial v}{\partial z^{(i)}} = (z^{(i)} - \mu) \cdot \left(-\frac{1}{2}\right)(v + \epsilon)^{-3/2} \cdot \frac{2}{m}(z^{(i)} - \mu)$$

The 2 and $\frac{1}{2}$ cancel:

$$= -\frac{1}{m}(z^{(i)} - \mu)^2 (v + \epsilon)^{-3/2}$$

---

#### Sum all three paths

$$\frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} = \frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} \bigg|_{\text{path 1}} + \frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} \bigg|_{\text{path 2}} + \frac{\partial \hat{z}^{(i)}}{\partial z^{(i)}} \bigg|_{\text{path 3}}$$

$$= s + \left(-\frac{s}{m}\right) + \left(-\frac{1}{m}(z^{(i)} - \mu)^2 (v + \epsilon)^{-3/2}\right)$$

$$= s - \frac{s}{m} - \frac{1}{m}(z^{(i)} - \mu)^2 (v + \epsilon)^{-3/2}$$

---

### Step 6: Full gradient $dL/dz$

$$\frac{\partial \mathcal{L}}{\partial z^{(i)}} = d\hat{z}^{(i)} \cdot \left(s - \frac{s}{m} - \frac{1}{m}(z^{(i)} - \mu)^2 (v + \epsilon)^{-3/2}\right)$$

where $d\hat{z}^{(i)} = d\tilde{z}^{(i)} \cdot \gamma$

Same formula applies at every layer — only the upstream gradient and layer-specific $z$, $\mu$, $v$, $\gamma$ change.

---

## Inference

During training: $\mu$ and $\sigma^2$ come from the current mini-batch.

During inference: use running EWA of $\mu$ and $\sigma^2$ accumulated across mini-batches during training.

$$\mu_{\text{running}} = \beta_{\text{bn}} \, \mu_{\text{running}} + (1 - \beta_{\text{bn}}) \, \mu_{\text{batch}}$$

$$\sigma^2_{\text{running}} = \beta_{\text{bn}} \, \sigma^2_{\text{running}} + (1 - \beta_{\text{bn}}) \, \sigma^2_{\text{batch}}$$

At inference, normalize using $\mu_{\text{running}}$ and $\sigma^2_{\text{running}}$ instead of batch statistics.
