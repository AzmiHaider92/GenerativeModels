
# Energy-Based Models (EBMs)

## 1. Motivation

So far, many probabilistic models define an explicit normalized probability distribution:

Bernoulli:

$$
P(x)=p^x(1-p)^{1-x}
$$

Gaussian:

$$
P(x)=\mathcal N(x;\mu,\Sigma)
$$

Categorical:

$$
P(x)=\text{softmax outputs}
$$

These are valid probability distributions because:

$$
\sum_x P(x)=1
$$

or for continuous variables:

$$
\int P(x)\,dx=1
$$

But what if instead of directly modeling probability, we simply learn a function that tells us how *good* or *realistic* a sample is?

This leads to Energy-Based Models.

---

# 2. Energy Formulation

Instead of predicting probability directly, a neural network predicts an energy:

$$
E_\theta(x)
$$

where:

- low energy = good / realistic sample
- high energy = unlikely / unrealistic sample

Probability is then defined as:

$$
P_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta}
$$

where:

$$
Z_\theta=\int e^{-E_\theta(x)}dx
$$

is called the **partition function**.

Interpretation:

- lower energy $\rightarrow$ higher probability
- higher energy $\rightarrow$ lower probability

This comes from statistical physics.

---

# 3. The Main Problem

The partition function:

$$
Z_\theta=\int e^{-E_\theta(x)}dx
$$

requires integrating over the entire data space.

For images:

- millions of dimensions
- impossible to compute exactly

So while the probability is mathematically defined, exact likelihood computation is intractable.

---

# 4. Maximum Likelihood Training

We still want Maximum Likelihood Estimation.

For one data sample:

$$
\log P_\theta(x) =
\log \left(\frac{e^{-E_\theta(x)}}{Z_\theta}\right)
$$

Expanding:

$$
\log P_\theta(x) =
-E_\theta(x)-\log Z_\theta
$$

Negative log likelihood:

$$
\mathcal L =
E_\theta(x)+\log Z_\theta
$$

Interpretation:

- first term lowers energy of real data
- second term prevents assigning low energy everywhere

---

# 5. Gradient of the Likelihood

Differentiating gives:

$$
\nabla_\theta \log P_\theta(x) =
-\nabla_\theta E_\theta(x)
+
\mathbb E_{x\sim P_\theta}
\left[
\nabla_\theta E_\theta(x)
\right]
$$

This is the key result.

Interpretation:

First term:

$$
-\nabla_\theta E_\theta(x)
$$

Lower energy for real data.

Second term:

$$
\mathbb E_{x\sim P_\theta}
\left[
\nabla_\theta E_\theta(x)
\right]
$$

Raise energy where the model currently places probability mass.

---

# 6. The Correct Intuition (Important)

A common misunderstanding:

"Why raise energy for model samples if they become good?"

The correct interpretation is:

EBMs are **distribution matching**, not "punishing fake samples."

Training matches:

$$
\mathbb E_{x\sim P_{data}}[\nabla E(x)]
$$

and

$$
\mathbb E_{x\sim P_\theta}[\nabla E(x)]
$$

If:

$$
P_\theta=P_{data}
$$

then:

$$
\mathbb E_{data}[\nabla E] =
\mathbb E_{model}[\nabla E]
$$

and the gradient becomes zero.

So EBM training is fundamentally:

**moment / expectation matching between data and model distributions.**

---

# 7. Why Sampling is Needed

The second expectation:

$$
\mathbb E_{x\sim P_\theta}[\cdot]
$$

requires sampling from:

$$
P_\theta(x)
$$

But:

$$
P_\theta(x)=\frac{e^{-E(x)}}{Z}
$$

has no direct sampler.

So we approximate the expectation using sampling.

This is where MCMC comes in.

---

# 8. What is MCMC?

MCMC = Markov Chain Monte Carlo

Idea:

Instead of sampling directly from:

$$
P_\theta(x)
$$

create a stochastic process that eventually visits states according to that distribution.

Intuition:

Think of the energy function as a landscape:

- valleys = low energy = high probability
- mountains = high energy = low probability

A random walker moves through the landscape, preferring low energy regions.

After enough steps, visited samples approximate:

$$
P_\theta(x)
$$

---

# 9. Langevin Dynamics

A common MCMC method for EBMs:

$$
x_{t+1} = x_t -
\eta \nabla_x E_\theta(x_t)
+
\sqrt{2\eta}\epsilon
$$

where:

$$
\epsilon \sim \mathcal N(0,I)
$$

Interpretation:

Gradient term:

$$
-\nabla_x E(x)
$$

moves toward lower energy.

Noise term:

$$
\sqrt{2\eta}\epsilon
$$

adds exploration.

Without noise:

pure gradient descent collapses to one mode.

With noise:

sampling explores the full distribution.

---

# 10. Training Algorithm

Training loop:

## Step 1: Real data

Take:

$$
x_{real}
$$

Compute:

$$
E_\theta(x_{real})
$$

Goal:

lower energy.

---

## Step 2: Initialize random noise

Start:

$$
x_0 \sim \mathcal N(0,I)
$$

---

## Step 3: Run Langevin sampling

Iterate:

$$
x_{t+1} = x_t -
\eta \nabla_x E(x_t)
+
noise
$$

Obtain approximate model samples:

$$
x_{model}
\sim P_\theta(x)
$$

---

## Step 4: Compute energies

Evaluate:

$$
E_\theta(x_{model})
$$

---

## Step 5: Update parameters

Lower:

$$
E(x_{real})
$$

Raise:

$$
E(x_{model})
$$

Equivalent to matching:

- data distribution expectations
- model distribution expectations

---

# 11. Sampling After Training

Once trained:

start from noise:

$$
x_0 \sim \mathcal N(0,I)
$$

Run Langevin dynamics:

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots
$$

The trajectory moves toward low-energy regions.

Generated samples are the final states.

There is no explicit generator network.

The **sampler itself acts as the generator**.

---

# 12. Comparison to GANs

GAN:

generator learns:

$$
z \rightarrow x
$$

discriminator pushes generator toward realism.

---

EBM:

network learns:

$$
x \rightarrow E(x)
$$

sampling happens through MCMC over the learned energy landscape.

No explicit generator.

---

# 13. Comparison to VAEs

## EBM problem

Intractable partition function:

$$
Z=\int e^{-E(x)}dx
$$

Need:

$$
\mathbb E_{x\sim P_\theta}[\cdot]
$$

Solution:

sampling approximation (MCMC).

---

## VAE problem

Latent variable model:

$$
P_\theta(x,z)=P(z)P_\theta(x|z)
$$

Need:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

Posterior:

$$
P_\theta(z|x)
$$

is intractable.

Solution:

Variational inference.

Approximate posterior:

$$
q_\phi(z|x)
$$

Optimize ELBO instead of exact likelihood.

---

Core difference:

EBM:

**approximate expectations using sampling**

VAE:

**approximate posterior using variational inference**

---

# 14. Pros and Cons

Advantages:

- flexible probability modeling
- no need for explicit generator architecture
- theoretically principled MLE objective

Disadvantages:

- expensive MCMC sampling
- slow training
- slow generation
- difficult optimization

---

# Final Mental Model

Energy-Based Models learn:

$$
x \rightarrow E_\theta(x)
$$

where:

- real data should lie in low-energy regions
- model distribution should match the data distribution

Training uses:

- real samples
- MCMC-generated model samples

to match expectations.

Generation happens by:

starting from noise and descending the learned energy landscape with randomness.



---

[← Go to Previous Repository - Normalizing Flows](../GANs/README.md)

[→ Go to Next Repository - TBD]

[← Back to Main Repository](../README.md)
