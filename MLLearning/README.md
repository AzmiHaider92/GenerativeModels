# Maximum Likelihood Learning

---

# 1. Maximum Likelihood Estimation (MLE)

## Main Idea

We define a probabilistic model:

$$
P_\theta(x)
$$

where:
- $x$ is the data
- $\theta$ are the learnable parameters

The goal of learning is:

> Find parameters that make the observed data most likely.

Given a dataset:

$$
\mathcal{D} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}
$$

we want:

$$
\theta^* =
\arg\max_\theta P_\theta(\mathcal{D})
$$

---

## Likelihood of a Dataset

Assuming samples are independent:

$$
P_\theta(\mathcal{D}) =
\prod_{i=1}^N P_\theta(x^{(i)})
$$

This is called the **likelihood**.

Important distinction:

- $P_\theta(x)$ → probability of data
- Likelihood → same expression viewed as a function of $\theta$

---

## Why Use Log Likelihood?

Products of many probabilities become numerically unstable.

So we take the logarithm:

$$
\log P_\theta(\mathcal{D}) =
\log \prod_{i=1}^N P_\theta(x^{(i)})
$$

Using:

$$
\log(ab)=\log a+\log b
$$

we get:

$$
\log P_\theta(\mathcal{D}) =
\sum_{i=1}^N \log P_\theta(x^{(i)})
$$

This makes optimization much easier.

---

## Final Objective

Maximum likelihood learning becomes:

$$
\theta^* =
\arg\max_\theta
\sum_{i=1}^N
\log P_\theta(x^{(i)})
$$

---

## Negative Log Likelihood (NLL)

Deep learning usually minimizes losses instead of maximizing objectives.

So we define:

$$
\mathcal{L}_{NLL} =
-
\sum_{i=1}^N
\log P_\theta(x^{(i)})
$$

This is the **Negative Log Likelihood (NLL)**.

---

## Connection to Cross Entropy

For classification:

$$
P_\theta(y|x)
$$

with softmax outputs, minimizing cross entropy is exactly maximizing likelihood.

Cross entropy is a special case of NLL.

---

# 2. Bernoulli Example → Binary Cross Entropy (BCE)

Suppose:

$$
x \in \{0,1\}
$$

and:

$$
P_\theta(x=1)=p_\theta
$$

Then:

$$
P_\theta(x) =
p_\theta^x(1-p_\theta)^{1-x}
$$

The log likelihood becomes:

$$
\log P_\theta(x) =
x\log p_\theta
+
(1-x)\log(1-p_\theta)
$$

Negative log likelihood:

$$
\mathcal{L} = - x\log p_\theta - (1-x)\log(1-p_\theta)
$$

This is exactly the **Binary Cross Entropy (BCE)** loss.

---

# 3. Gaussian Example → Mean Squared Error (MSE)

Suppose:

$$
y = f_\theta(x) + \epsilon
$$

with Gaussian noise:

$$
\epsilon \sim \mathcal{N}(0,\sigma^2)
$$

Then:

$$
P_\theta(y|x) =
\mathcal{N}(f_\theta(x),\sigma^2)
$$

Maximizing likelihood becomes minimizing:

$$
\|y-f_\theta(x)\|^2
$$

Therefore:

> Mean Squared Error (MSE) is Maximum Likelihood under Gaussian noise assumptions.

---

# 4. KL-Divergence

## Why Do We Need KL-Divergence?

MLE is one specific optimization rule.

KL-divergence is a more general mathematical tool for comparing probability distributions.

MLE actually comes from KL-divergence.

---

## Main Idea

Suppose:
- real data distribution:

$$
P_{data}(x)
$$

- model distribution:

$$
P_\theta(x)
$$

We want:

$$
P_\theta(x)
\approx
P_{data}(x)
$$

So we need a way to measure how close two distributions are.

---

## Definition

The KL-divergence from $P$ to $Q$ is:

$$
D_{KL}(P||Q) =
\sum_x
P(x)
\log
\frac{P(x)}{Q(x)}
$$

Continuous version:

$$
D_{KL}(P||Q) =
\int
P(x)
\log
\frac{P(x)}{Q(x)}
dx
$$

---

## Intuition

KL-divergence measures:

> How different one probability distribution is from another.

If:

$$
D_{KL}(P||Q)=0
$$

then the distributions are identical.

---

## Important Interpretation

KL-divergence measures:

> Extra information cost when using the wrong distribution.

---

# 5. Connection Between KL and Maximum Likelihood

Start from:

$$
D_{KL}(P_{data}||P_\theta) =
\sum_x
P_{data}(x)
\log
\frac{P_{data}(x)}{P_\theta(x)}
$$

Split the logarithm:

$$
= \sum_x P_{data}(x)\log P_{data}(x) - \sum_x P_{data}(x)\log P_\theta(x)
$$

Rewrite:

$$
=
H(P_{data})
-
\mathbb{E}_{x\sim P_{data}}
[\log P_\theta(x)]
$$

where:

$$
H(P_{data}) = - \sum_x
P_{data}(x)\log P_{data}(x)
$$

is the entropy.

Since entropy does not depend on $\theta$:

$$
\arg\min_\theta
D_{KL}(P_{data}||P_\theta) =
\arg\max_\theta
\mathbb{E}_{x\sim P_{data}}
[\log P_\theta(x)]
$$

This is exactly Maximum Likelihood Learning.

---

# 6. KL vs Maximum Likelihood

MLE initially looks like:

$$
\max_\theta
\sum_i
\log P_\theta(x_i)
$$

which feels like:

> Increase probability on training samples.

KL-divergence reveals the deeper meaning:

$$
\min_\theta
D_{KL}(P_{data}||P_\theta)
$$

Meaning:

> Reshape the model distribution to match the real data distribution.

---

## Important Insight

MLE is:
- a training rule

KL-divergence is:
- a general mathematical concept about distributions

MLE comes from KL-divergence.

---

# 7. Monte Carlo Estimation

## Expectations

An expectation means:

> The average value over a probability distribution.

Example:

$$
\mathbb{E}_{x\sim P(x)}[f(x)]
$$

means:

> Average value of $f(x)$ when sampling from $P(x)$.

---

## Problem

The true objective is:

$$
\mathbb{E}_{x\sim P_{data}}
[\log P_\theta(x)]
$$

But:
- we do not know $P_{data}$ explicitly
- only samples from it

---

## Monte Carlo Estimation

Suppose:

$$
x^{(1)}, \dots, x^{(N)} \sim P(x)
$$

Then:

$$
\mathbb{E}_{x\sim P}[f(x)]
\approx
\frac{1}{N}
\sum_{i=1}^N f(x^{(i)})
$$

Monte Carlo estimation is literally averaging samples.

---

## Connection to MLE

Using Monte Carlo:

$$
\mathbb{E}_{x\sim P_{data}}
[\log P_\theta(x)]
\approx
\frac{1}{N}
\sum_{i=1}^N
\log P_\theta(x^{(i)})
$$

which becomes Maximum Likelihood Learning.

---

# 8. Monte Carlo Bias and Variance

## Bias

Bias means:

> Does the estimator systematically miss the true value?

Monte Carlo estimation is unbiased:

$$
\mathbb{E}[\hat{\mu}] =
\mathbb{E}[f(x)]
$$

---

## Variance

Different sample sets produce different estimates.

Monte Carlo variance is:

$$
\text{Var}(\hat{\mu}) =
\frac{\sigma^2}{N}
$$

Meaning:
- more samples
- lower variance

---

# 9. Coin Toss MLE Example

Suppose dataset:

$$
\mathcal{D} =
\{H,H,T,H,T,H,H,H,T,H\}
$$

Counts:
- Heads = 7
- Tails = 3

Define:

$$
\theta=P(H)
$$

Then:

$$
P(T)=1-\theta
$$

Likelihood:

$$
P(\mathcal{D}|\theta) =
\theta^7(1-\theta)^3
$$

Log likelihood:

$$
L(\theta) =
7\log\theta
+
3\log(1-\theta)
$$

Take derivative:

$$
\frac{dL}{d\theta} =
\frac{7}{\theta} -
\frac{3}{1-\theta}
$$

Set equal to zero:

$$
\frac{7}{\theta} =
\frac{3}{1-\theta}
$$

Solve:

$$
7(1-\theta)=3\theta
$$

$$
7=10\theta
$$

Final estimate:

$$
\theta_{MLE}=0.7
$$

which equals:

$$
\frac{7}{10}
$$

---

# 10. Gradient Descent and SGD

## Why We Need Gradient Descent

For simple models like coin tosses, we can solve analytically.

But autoregressive neural networks are extremely complicated:

$$
P_\theta(x) =
\prod_{i=1}^N
P_\theta(x_i|x_{<i})
$$

No closed-form solution exists.

So we optimize iteratively.

---

## Gradient Descent

Suppose loss:

$$
\mathcal{L}(\theta)
$$

The gradient:

$$
\nabla_\theta \mathcal{L}
$$

points toward increasing loss.

So we update:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

where:
- $\eta$ is the learning rate

---

## Connection to MLE

We minimize:

$$
\mathcal{L} = - \sum_i
\log P_\theta(x_i)
$$

which is the Negative Log Likelihood.

---

## Stochastic Gradient Descent (SGD)

Computing gradients over huge datasets is expensive.

So we use mini-batches:

$$
\hat{\mathcal{L}} = -
\frac{1}{B}
\sum_{i=1}^B
\log P_\theta(x^{(i)})
$$

This is a Monte Carlo estimate of the true loss.

---

# 11. Bias–Variance Tradeoff

## Bias

Bias means:

> The model is systematically too simple to represent the true distribution.

This causes:
- underfitting
- poor performance even on training data

High bias usually comes from:
- overly simple models
- restrictive assumptions

---

## Variance

Variance means:

> The model changes too much depending on the training dataset.

This causes:
- overfitting
- sensitivity to noise

High variance usually comes from:
- overly expressive models
- small datasets

---

## Relationship to Model Complexity

Simple models:
- high bias
- low variance

Complex models:
- low bias
- high variance

---

## Relationship to Dataset Size

Small datasets:
- noisy Monte Carlo estimates
- unstable parameter estimation
- high variance

Large datasets:
- more stable expectations
- lower variance

---

# Final Big Picture

The full probabilistic learning story is:

We want:

$$
P_\theta(x)
\approx
P_{data}(x)
$$

Measure similarity using:

$$
D_{KL}(P_{data}||P_\theta)
$$

This becomes:

$$
\max_\theta
\mathbb{E}_{x\sim P_{data}}
[\log P_\theta(x)]
$$

We approximate expectations using Monte Carlo:

$$
\frac{1}{N}
\sum_i
\log P_\theta(x_i)
$$

Then optimize using:
- gradient descent
- stochastic gradient descent

This leads to:
- NLL
- Cross Entropy
- BCE
- MSE

which are all connected through Maximum Likelihood Learning.
