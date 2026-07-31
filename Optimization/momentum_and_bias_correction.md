# Momentum (without Bias Correction)

Refer: [Momentum based optimizers - GFG](https://www.geeksforgeeks.org/machine-learning/ml-momentum-based-gradient-optimizer-introduction/)

The idea is we don't use simple average like $(w1 + w2 + w3)/3$ because in that case we are giving equal weight to all the previous gradients. Instead, we use exponentially weighted average which gives more weight to the recent gradients and less weight to the older gradients. 

Suppose average gradient is moving in +ve direction and suppose the next batch unexpectedly produces the opposite gradient, -1, The model still moves in the original direction, but more slowly. If several subsequent gradients point the opposite way, the direction (velocity) will eventually reverse.

EWA applied to gradients.

$$v_{dW} = \beta \, v_{dW} + (1 - \beta) \, dW$$

$$v_{db} = \beta \, v_{db} + (1 - \beta) \, db$$
Where $v_{dW}$ and $v_{db}$ are the running exponentially weighted averages of the gradients $dW$ and $db$ respectively.

Example (with $\beta = 0.9$, $v_{dW_0}=0$):
$$v_{dW_1} = 0.9 \, v_{dW_0} + 0.1 \, dW_{1}$$
$$v_{dW_2} = 0.9 \, v_{dW_1} + 0.1 \, dW_{2}$$
$$v_{dW_3} = 0.9 \, v_{dW_2} + 0.1 \, dW_{3}$$

All the above equations result in:

$$v_{dW_3} = 0.1 \, dW_{3} + 0.09 \, dW_{2} + 0.081 \, dW_{1}$$

The weights will sum to $0.1 + 0.09 + 0.081 = 0.271$, which is less than $1$. This is because we are not considering the initial value of $v_{dW_0}$, which is $0$. The weights on the real observations sum to $1 - \beta^t$, not $1$.

Update:

$$W = W - \alpha \, v_{dW}$$

$$b = b - \alpha \, v_{db}$$

Initialize: $v_{dW} = 0$, $v_{db} = 0$

Typical: $\beta = 0.9$


## Why Do the Gradient Weights Add Up to Only 0.271?

Let us focus only on why the gradient weights add up to $0.271$.

The key idea is that there is another coefficient, $0.729$, attached to the initial value:

$$
v_{dW_1} = 0
$$

We use the following update rule:

$$
v_{\text{new}}
=
0.9v_{\text{old}}
+
0.1(\text{current gradient})
$$

### First Update

We begin with:

$$
v_{dW_2}
=
0.9v_{dW_1}
+
0.1dW_1
$$

At this point, the coefficients add up to $1$:

$$
0.9 + 0.1 = 1
$$

Now substitute $v_{dW_1}=0$:

$$
v_{dW_2}
=
0.9(0)
+
0.1dW_1
$$

Therefore:

$$
v_{dW_2}
=
0.1dW_1
$$

The coefficient $0.9$ has not mathematically disappeared. It is attached to the initial value, which is zero:

$$
0.9 \times 0 = 0
$$

Therefore, the real gradient $dW_1$ currently has a total weight of only $0.1$.

---

### Second Update

The second update is:

$$
v_{dW_3}
=
0.9v_{dW_2}
+
0.1dW_2
$$

From the first update:

$$
v_{dW_2}
=
0.9v_{dW_1}
+
0.1dW_1
$$

Substitute this complete expression into the second update:

$$
v_{dW_3}
=
0.9(0.9v_{dW_1}+0.1dW_1)
+
0.1dW_2
$$

Distribute the outer $0.9$:

$$
v_{dW_3}
=
(0.9 \times 0.9)v_{dW_1}
+
(0.9 \times 0.1)dW_1
+
0.1dW_2
$$

Calculate the multiplications:

$$
v_{dW_3}
=
0.81v_{dW_1}
+
0.09dW_1
+
0.1dW_2
$$

All the coefficients still add up to $1$:

$$
0.81+0.09+0.1=1
$$

However, because $v_{dW_1}=0$:

$$
0.81v_{dW_1}
=
0.81(0)
=
0
$$

The expression becomes:

$$
v_{dW_3}
=
0.09dW_1
+
0.1dW_2
$$

The weights of the actual gradients now add up to:

$$
0.09+0.1=0.19
$$

The remaining weight, $0.81$, is attached to the initial value of zero.

---

### Third Update

The third update is:

$$
v_{dW_4}
=
0.9v_{dW_3}
+
0.1dW_3
$$

From the second update:

$$
v_{dW_3}
=
0.81v_{dW_1}
+
0.09dW_1
+
0.1dW_2
$$

Substitute this expression into the third update:

$$
v_{dW_4}
=
0.9
\left(
0.81v_{dW_1}
+
0.09dW_1
+
0.1dW_2
\right)
+
0.1dW_3
$$

Distribute $0.9$ across every term:

$$
v_{dW_4}
=
(0.9 \times 0.81)v_{dW_1}
+
(0.9 \times 0.09)dW_1
+
(0.9 \times 0.1)dW_2
+
0.1dW_3
$$

Calculate each multiplication:

$$
v_{dW_4}
=
0.729v_{dW_1}
+
0.081dW_1
+
0.09dW_2
+
0.1dW_3
$$

Now add all the coefficients:

$$
0.729+0.081+0.09+0.1=1
$$

Therefore, the total weight is still $1$.

However, because:

$$
v_{dW_1}=0
$$

we get:

$$
0.729v_{dW_1}
=
0.729(0)
=
0
$$

The expression therefore becomes:

$$
v_{dW_4}
=
0.081dW_1
+
0.09dW_2
+
0.1dW_3
$$

The weights of the actual gradients add up to:

$$
0.081+0.09+0.1=0.271
$$

Therefore:

$$
\boxed{0.271+0.729=1}
$$

- $0.271$ is the total weight assigned to the three real gradients.
- $0.729$ is the weight still attached to the initial value of zero.

Because the initial value is zero:

$$
0.729 \times 0 = 0
$$

That is why only $0.271$ appears in the final expression.

The exponentially weighted average is initially influenced heavily by its initialization. Because it was initialized to zero, that influence is invisible in the final numerical result.

As more gradients are processed, the weight attached to the initial zero becomes progressively smaller.

# Bias Correction

This note explains the direct connection between exponentially weighted averages, zero initialization, bias correction, and normalization.

## 1. Exponentially Weighted Average of Gradients

Let $g_t$ be the gradient calculated at step $t$. An exponentially weighted average is updated using:

$$
v_t = \beta v_{t-1} + (1-\beta)g_t
$$

where:

- $v_t$ is the running exponentially weighted average;
- $g_t$ is the current gradient;
- $\beta$ controls how much of the previous average is retained;
- $1-\beta$ controls how much weight the current gradient receives.

For $\beta=0.9$, the equation becomes:

$$
v_t = 0.9v_{t-1}+0.1g_t
$$

We initialize the average as:

$$
v_0=0
$$

## 2. Expanding the First Three Updates

### First update

$$
v_1=0.9v_0+0.1g_1
$$

Because $v_0=0$:

$$
v_1=0.9(0)+0.1g_1
$$

Therefore:

$$
v_1=0.1g_1
$$

The coefficient $0.9$ has not mathematically disappeared. It is attached to the initial value, which happens to be zero:

$$
0.9v_0=0.9(0)=0
$$

### Second update

$$
v_2=0.9v_1+0.1g_2
$$

Substitute the complete expression for $v_1$:

$$
v_2=0.9(0.9v_0+0.1g_1)+0.1g_2
$$

Distribute the outer $0.9$:

$$
v_2=(0.9\times0.9)v_0+(0.9\times0.1)g_1+0.1g_2
$$

Therefore:

$$
v_2=0.81v_0+0.09g_1+0.1g_2
$$

All the coefficients still add up to $1$:

$$
0.81+0.09+0.1=1
$$

Because $v_0=0$:

$$
0.81v_0=0.81(0)=0
$$

So the visible expression is:

$$
v_2=0.09g_1+0.1g_2
$$

The weights of the real gradients currently add up to:

$$
0.09+0.1=0.19
$$

The remaining weight, $0.81$, is attached to the initial zero.

### Third update

$$
v_3=0.9v_2+0.1g_3
$$

Substitute the complete expression for $v_2$:

$$
v_3=0.9(0.81v_0+0.09g_1+0.1g_2)+0.1g_3
$$

Distribute $0.9$ across every term:

$$
v_3
=(0.9\times0.81)v_0
+(0.9\times0.09)g_1
+(0.9\times0.1)g_2
+0.1g_3
$$

Calculate the products:

$$
v_3=0.729v_0+0.081g_1+0.09g_2+0.1g_3
$$

All the coefficients add up to $1$:

$$
0.729+0.081+0.09+0.1=1
$$

But because $v_0=0$:

$$
0.729v_0=0.729(0)=0
$$

The visible expression becomes:

$$
v_3=0.081g_1+0.09g_2+0.1g_3
$$

The weights of the three real gradients add up to only:

$$
0.081+0.09+0.1=0.271
$$

The complete weight is therefore:

$$
\underbrace{0.729}_{\text{initial value}}
+
\underbrace{0.271}_{\text{real gradients}}
=1
$$

We did not ignore the initial value. We included it, but its numerical contribution is zero because it is multiplied by $v_0=0$.

## 3. Why This Creates a Bias Toward Zero

The three observed gradients occupy only $0.271$ of the total weight. The remaining $0.729$ is assigned to the artificial initial value $v_0=0$.

Consequently, the early exponentially weighted averages are pulled toward zero. This is called **initialization bias** or **bias toward zero**.

Consider a simple example in which all three gradients are $10$:

$$
g_1=g_2=g_3=10
$$

Their average should intuitively be $10$. Let us calculate the uncorrected exponentially weighted average.

### First update

$$
v_1=0.9v_0+0.1g_1
$$

$$
v_1=0.9(0)+0.1(10)=1
$$

Even though the gradient is $10$, the result is only $1$.

### Second update

$$
v_2=0.9v_1+0.1g_2
$$

$$
v_2=0.9(1)+0.1(10)=0.9+1=1.9
$$

### Third update

$$
v_3=0.9v_2+0.1g_3
$$

$$
v_3=0.9(1.9)+0.1(10)=1.71+1=2.71
$$

We supplied three gradients of $10$, but the uncorrected estimate is only:

$$
v_3=2.71
$$

The expanded expression shows why:

$$
v_3=0.081(10)+0.09(10)+0.1(10)
$$

$$
v_3=10(0.081+0.09+0.1)
$$

$$
v_3=10(0.271)=2.71
$$

The estimate is too small because the real gradients possess only $0.271$ of the total weight at this point.

## 4. Normalization from First Principles

Normalization converts actual amounts into their shares of a chosen total.

Suppose three people have the amounts:

$$
2,\qquad3,\qquad5
$$

The total is:

$$
2+3+5=10
$$

Their respective shares of the total are:

$$
\frac{2}{10}=0.2
$$

$$
\frac{3}{10}=0.3
$$

$$
\frac{5}{10}=0.5
$$

Adding the shares gives:

$$
0.2+0.3+0.5=1
$$

Here, $1$ represents one complete whole, or $100\%$:

$$
20\%+30\%+50\%=100\%
$$

| Actual amount | Share of total |
| ---: | ---: |
| $2$ | $2/10=0.2$ |
| $3$ | $3/10=0.3$ |
| $5$ | $5/10=0.5$ |
| **Total: $10$** | **Total: $1$** |

### General proof

Let the values be $a$, $b$, and $c$, and define their total as:

$$
S=a+b+c
$$

Divide each value by the total:

$$
\frac{a}{S}+\frac{b}{S}+\frac{c}{S}
$$

Because the denominators are identical, combine the fractions:

$$
\frac{a+b+c}{S}
$$

But $a+b+c=S$, so:

$$
\frac{a+b+c}{S}=\frac{S}{S}=1
$$

Therefore:

$$
\boxed{
\frac{a}{S}+\frac{b}{S}+\frac{c}{S}=1
}
$$

The sum becomes $1$ because the numerator contains the complete total and the denominator is that same total:

$$
\frac{\text{complete total}}{\text{complete total}}=1
$$

### Does normalization change the contribution of each value?

Normalization changes each value's **absolute magnitude and scale**, but it does not change the values' **relative proportions**.

Before normalization, the ratio is:

$$
2:3:5
$$

After normalization, it is:

$$
0.2:0.3:0.5
$$

The relationship between the first two numbers is unchanged:

$$
\frac{3}{2}=1.5
$$

$$
\frac{0.3}{0.2}=1.5
$$

Likewise:

$$
\frac{5}{2}=2.5
$$

$$
\frac{0.5}{0.2}=2.5
$$

Every number was scaled by the same factor, $1/10$. The process can be reversed by multiplying by the original total:

$$
0.2(10)=2,\qquad0.3(10)=3,\qquad0.5(10)=5
$$

Normalization does not destroy the original proportions. It expresses each value as a fraction of the chosen whole.

## 5. Applying Normalization to the Gradient Weights

Before bias correction, the real-gradient weights are:

$$
0.081,\qquad0.09,\qquad0.1
$$

Their total is:

$$
0.081+0.09+0.1=0.271
$$

If we want to know each gradient's share of this available $0.271$ weight, we divide every weight by $0.271$:

$$
\frac{0.081}{0.271}\approx0.299
$$

$$
\frac{0.09}{0.271}\approx0.332
$$

$$
\frac{0.1}{0.271}\approx0.369
$$

The normalized weights add up to $1$:

$$
0.299+0.332+0.369\approx1
$$

Using the exact fractions rather than rounded decimals:

$$
\frac{0.081}{0.271}
+\frac{0.09}{0.271}
+\frac{0.1}{0.271}
$$

$$
=\frac{0.081+0.09+0.1}{0.271}
$$

$$
=\frac{0.271}{0.271}
$$

$$
=1
$$

This is not a coincidence. We deliberately divide by $0.271$ because it is the sum of the real-gradient weights.

The order and relative importance of the gradients remain unchanged:

$$
0.081<0.09<0.1
$$

and:

$$
0.299<0.332<0.369
$$

The most recent gradient still contributes the most. Normalization only changes the overall scale from a total weight of $0.271$ to a total weight of $1$.

## 6. What Bias Correction Changes

Before correction, the complete expression is:

$$
v_3
=
\underbrace{0.729v_0}_{\text{initial value}}
+
\underbrace{0.081g_1+0.09g_2+0.1g_3}_{\text{observed gradients}}
$$

The coefficients are relative to the original total of $1$, which includes the artificial initial value.

Bias correction effectively says:

> The initial zero is not an observed gradient. Remove its share and treat the observed gradients as the complete available information.

The observed gradients have a combined weight of $0.271$, so bias correction treats that $0.271$ as the new whole:

$$
\frac{0.271}{0.271}=1
$$

It divides the uncorrected estimate by $0.271$:

$$
\hat v_3=\frac{v_3}{0.271}
$$

Using the expanded expression:

$$
\hat v_3
=
\frac{0.081g_1+0.09g_2+0.1g_3}{0.271}
$$

This is equivalent to dividing every coefficient individually:

$$
\hat v_3
=
\frac{0.081}{0.271}g_1
+
\frac{0.09}{0.271}g_2
+
\frac{0.1}{0.271}g_3
$$

Therefore:

$$
\hat v_3
\approx
0.299g_1+0.332g_2+0.369g_3
$$

Bias correction changes the scale and interpretation of the coefficients:

- **Before correction:** the coefficients represent contributions relative to the original total of $1$, including the initial zero.
- **After correction:** the coefficients represent contributions relative only to the weight occupied by actual observed gradients.

One way to imagine this is as a pie chart:

- $72.9\%$ of the original pie belongs to the blank initialization $v_0=0$;
- $27.1\%$ contains real gradient information.

Bias correction removes the blank portion and proportionally enlarges the real slices until they fill the entire pie. Their sizes change, but their proportions relative to one another do not.

## 7. Correcting the Numerical Example

Earlier, the three gradients were all $10$, and the uncorrected estimate was:

$$
v_3=2.71
$$

The real-gradient weights sum to:

$$
0.271
$$

Therefore, the bias-corrected value is:

$$
\hat v_3=\frac{v_3}{0.271}
$$

$$
\hat v_3=\frac{2.71}{0.271}=10
$$

Thus:

$$
\boxed{\hat v_3=10}
$$

The corrected estimate agrees with the constant gradient value.

## 8. The General Bias-Correction Formula

After $t$ updates, the exponentially weighted average can be expanded as:

$$
v_t
=
(1-\beta)
\left(
g_t+\beta g_{t-1}+\beta^2g_{t-2}+\cdots+\beta^{t-1}g_1
\right)
$$

The sum of the real-gradient weights is:

$$
(1-\beta)
\left(
1+\beta+\beta^2+\cdots+\beta^{t-1}
\right)
$$

The expression inside the parentheses is a geometric series:

$$
1+\beta+\beta^2+\cdots+\beta^{t-1}
=
\frac{1-\beta^t}{1-\beta}
$$

Therefore, the total gradient weight is:

$$
(1-\beta)\frac{1-\beta^t}{1-\beta}
$$

Cancel $1-\beta$:

$$
\boxed{1-\beta^t}
$$

This gives the general bias-correction formula:

$$
\boxed{
\hat v_t=\frac{v_t}{1-\beta^t}
}
$$

The denominator is not arbitrary: $1-\beta^t$ is precisely the sum of the weights assigned to the real gradients after $t$ updates.

## 9. Does the Uncorrected Weight Ever Reach One?

Without bias correction, the real-gradient weights sum to:

$$
1-\beta^t
$$

For $0<\beta<1$, we have:

$$
\beta^t>0
$$

at every finite value of $t$. Consequently:

$$
1-\beta^t<1
$$

For $\beta=0.9$:

| Updates $t$ | Real-gradient weight $1-0.9^t$ |
| ---: | ---: |
| $1$ | $0.1$ |
| $2$ | $0.19$ |
| $3$ | $0.271$ |
| $10$ | approximately $0.6513$ |
| $50$ | approximately $0.9948$ |
| $100$ | approximately $0.99997$ |

The uncorrected real-gradient weight becomes extremely close to $1$, but it does not equal $1$ at any finite step. It reaches $1$ only as a limit:

$$
\lim_{t\to\infty}(1-\beta^t)=1
$$

With bias correction, however, the normalized real-gradient weights sum to exactly $1$ at every step:

$$
\frac{1-\beta^t}{1-\beta^t}=1
$$

Therefore:

$$
\boxed{
\begin{aligned}
\text{Without bias correction:}&\quad
\text{real-gradient weights approach }1 \\
\text{With bias correction:}&\quad
\text{real-gradient weights equal }1
\end{aligned}
}
$$

In computer calculations, floating-point rounding may display a value such as $0.9999999$, but mathematically the corrected weights sum to exactly $1$.

## 10. The Complete Connection

The entire idea can be summarized as follows:

1. Initialize the exponentially weighted average with $v_0=0$.
2. During the early updates, part of the total weight remains attached to this artificial zero.
3. After $t$ updates, the real-gradient weights sum to only $1-\beta^t$.
4. The estimate is therefore biased toward zero.
5. Divide by $1-\beta^t$, the current total weight of the observed gradients.
6. The corrected gradient weights now sum to $1$ while retaining their relative proportions.

In equation form:

$$
v_0=0
$$

$$
\Downarrow
$$

$$
\text{Real-gradient weights sum to }1-\beta^t<1
$$

$$
\Downarrow
$$

$$
v_t\text{ is biased toward zero during the early updates}
$$

$$
\Downarrow
$$

$$
\text{Divide by }1-\beta^t
$$

$$
\Downarrow
$$

$$
\boxed{\hat v_t=\frac{v_t}{1-\beta^t}}
$$

## 11. Momentum and Adam

Classical stochastic gradient descent with momentum often does **not** apply bias correction because the initialization bias diminishes during a long training run.

Adam, however, explicitly applies bias correction to both its first-moment and second-moment estimates. The mathematical motivation is the same: both running estimates are initialized to zero and would otherwise be artificially small during the early optimization steps.

The central idea is:

$$
\boxed{
\text{Normalized share}
=
\frac{\text{individual amount}}{\text{total amount}}
}
$$

Bias correction finds the current total weight of the observed gradients, divides the estimate by that total, and thereby removes the artificial influence of zero initialization while preserving the relative emphasis on recent gradients.
