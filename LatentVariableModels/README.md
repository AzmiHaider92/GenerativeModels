# Latent Variable Models

## 1. What is a Latent Variable?

A latent variable is a hidden variable that is not directly observed in the data but explains underlying structure.

Instead of modeling only:

$$
P(x)
$$

we introduce a hidden variable:

$$
z
$$

and model:

$$
P(x,z)
$$

where:

- $x$ = observed data
- $z$ = latent (hidden) variable

---

# 2. Generative Process

Latent variable models assume the following process:

1. Sample latent variable:

$$
z \sim P(z)
$$

2. Generate data conditioned on the latent:

$$
x \sim P_\theta(x|z)
$$

This defines the joint distribution:

$$
P_\theta(x,z)=P(z)P_\theta(x|z)
$$

To get the probability of the data:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

For discrete latent variables:

$$
P_\theta(x)=\sum_z P_\theta(x,z)
$$

This operation is called **marginalization**.

---

# 3. Gaussian Mixture Model (Shallow Latent Model)

A Gaussian Mixture Model (GMM) is a classic latent variable model.

Latent variable:

$$
z \in \{1,\dots,K\}
$$

represents which Gaussian generated the sample.

Generative process:

1. Choose Gaussian component:

$$
z \sim \text{Categorical}(\pi)
$$

2. Generate sample:

$$
x \sim \mathcal{N}(\mu_z,\Sigma_z)
$$

Overall distribution:

$$
P(x)=\sum_{k=1}^K \pi_k \mathcal{N}(x;\mu_k,\Sigma_k)
$$

---

# 4. Deep Latent Gaussian Models

Instead of simple Gaussian mappings, we use neural networks.

Usually:

$$
z \sim \mathcal{N}(0,I)
$$

and:

$$
x \sim P_\theta(x|z)
$$

where:

$$
P_\theta(x|z)
$$

is parameterized by a neural network decoder.

This is the foundation of Variational Autoencoders (VAEs).

---

# 5. Maximum Marginal Likelihood Learning

Given dataset:

$$
\mathcal{D}=\{x^{(i)}\}_{i=1}^N
$$

we want to maximize likelihood:

$$
\max_\theta \sum_i \log P_\theta(x^{(i)})
$$

But:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

Therefore:

$$
\log P_\theta(x) =
\log \int P_\theta(x,z)\,dz
$$

This integral is usually intractable.

---

# 6. Variational Inference

We introduce an approximate posterior distribution:

$$
q_\phi(z|x)
$$

Goal:

$$
q_\phi(z|x)\approx P_\theta(z|x)
$$

where the true posterior is:

$$
P_\theta(z|x) =
\frac{P_\theta(x,z)}{P_\theta(x)}
$$

Exact computation is usually impossible, so we approximate it.

---

# 7. ELBO Derivation

Start from:

$$
\log P_\theta(x)
$$

Insert $q_\phi(z|x)$:

$$
\log P_\theta(x) =
\log \int q_\phi(z|x)
\frac{P_\theta(x,z)}{q_\phi(z|x)}dz
$$

Rewrite as expectation:

$$
= \log
\mathbb{E}_{q_\phi(z|x)}
\left[
\frac{P_\theta(x,z)}{q_\phi(z|x)}
\right]
$$

Using Jensen's inequality:

$$
\log \mathbb{E}[X]
\geq
\mathbb{E}[\log X]
$$

we obtain:

$$
\log P_\theta(x)
\geq
\mathbb{E}_{q_\phi(z|x)}
\left[
\log
\frac{P_\theta(x,z)}
{q_\phi(z|x)}
\right]
$$

This lower bound is called the ELBO.

---

# 8. ELBO Formula

The ELBO can be written as:

$$
\mathcal{L}(\theta,\phi;x)
= \mathbb{E}_{q_\phi(z|x)} [\log P_\theta(x,z)] -
\mathbb{E}_{q_\phi(z|x)}
[\log q_\phi(z|x)]
$$

Using:

$$
P_\theta(x,z)=P(z)P_\theta(x|z)
$$

we get:

$$
\boxed{
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)} [\log P_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|P(z))
}
$$

---

# 9. ELBO Interpretation

## Reconstruction Term

$$
\mathbb{E}_{q_\phi(z|x)}
[\log P_\theta(x|z)]
$$

Encourages accurate reconstruction of the data.

---

## KL Regularization

$$
D_{KL}(q_\phi(z|x)\|P(z))
$$

Pushes latent representations toward the prior distribution.

Usually:

$$
P(z)=\mathcal{N}(0,I)
$$

---

# 10. Relation Between ELBO and KL-Divergence

Important identity:

$$
\log P_\theta(x) =
\mathcal{L}(\theta,\phi;x)
+
D_{KL}(q_\phi(z|x)\|P_\theta(z|x))
$$

Since KL-divergence is always nonnegative:

$$
D_{KL}(\cdot)\geq0
$$

then:

$$
\mathcal{L}\leq \log P_\theta(x)
$$

which explains why ELBO is a lower bound.

---

# 11. Why Choosing a Good $q$ Matters

If:

$$
q_\phi(z|x)=P_\theta(z|x)
$$

then:

$$
D_{KL}(q_\phi(z|x)\|P_\theta(z|x))=0
$$

Therefore:

$$
\mathcal{L}=\log P_\theta(x)
$$

Meaning the ELBO becomes exact.

Training therefore tries to:

1. maximize ELBO
2. make $q_\phi(z|x)$ close to the true posterior

---

# 12. Variational Inference Intuition

Variational Inference converts difficult probabilistic inference into optimization.

Instead of computing:

$$
P_\theta(z|x)
$$

exactly, we:

1. choose a tractable family $q_\phi(z|x)$
2. optimize its parameters
3. minimize:

$$
D_{KL}(q_\phi(z|x)\|P_\theta(z|x))
$$

Equivalent to maximizing the ELBO.

---

# 13. Missing Top Half of an Image Example

Suppose:

- bottom half of image observed
- top half missing

We want:

$$
P(x_{\text{top}}|x_{\text{bottom}})
$$

---

## Training Phase

The VAE is trained on full images.

Encoder:

$$
q_\phi(z|x)
$$

Decoder:

$$
P_\theta(x|z)
$$

---

## Inference Phase

Create masked image:

$$
\tilde{x} =
\begin{bmatrix}
0 \\
0 \\
x_{\text{bottom}}
\end{bmatrix}
$$

Feed into encoder:

$$
q_\phi(z|\tilde{x})
$$

Sample latent variable:

$$
z \sim q_\phi(z|\tilde{x})
$$

Generate full image:

$$
x \sim P_\theta(x|z)
$$

The decoder predicts a plausible missing top half.

Different latent samples:

$$
z_1,z_2,z_3
$$

produce different valid completions.

---

# 14. Key Takeaways

- Latent variables explain hidden structure in data.
- Generative process:

$$
z \sim P(z)
$$

$$
x \sim P_\theta(x|z)
$$

- Exact likelihood is difficult because:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

- Variational inference introduces:

$$
q_\phi(z|x)
$$

to approximate the true posterior.

- ELBO is derived using Jensen's inequality.
- Maximizing ELBO approximates maximum likelihood learning.
- VAEs are deep latent variable models trained with the ELBO objective.

---


# 15. Gradient-Based Optimization of the ELBO

We want to optimize:

$$
\max_{\theta,\phi} \mathcal{L}(\theta,\phi;x)
$$

where:

$$
\mathcal{L}(\theta,\phi;x) =
\mathbb{E}_{q_\phi(z|x)}
\left[
\log P_\theta(x,z) -
\log q_\phi(z|x)
\right]
$$

The difficulty is that:

$$
z \sim q_\phi(z|x)
$$

depends on the encoder parameters:

$$
\phi
$$

This creates a problem for backpropagation through the sampling operation.

---

# 16. Why Sampling Causes Problems

Normally in neural networks:

$$
z=f_\phi(x)
$$

and gradients flow using the chain rule:

$$
\frac{\partial \text{loss}}{\partial \phi} =
\frac{\partial \text{loss}}{\partial z}
\frac{\partial z}{\partial \phi}
$$

However, in VAEs we sample:

$$
z \sim q_\phi(z|x)
$$

Sampling is stochastic and is not a standard differentiable operation.

Autograd does not directly know how to compute:

$$
\frac{\partial z}{\partial \phi}
$$

through a random sampling step.

This motivates special gradient estimators.

---

# 17. REINFORCE / Score Function Estimator

REINFORCE uses the identity:

$$
\nabla_\phi
\mathbb{E}_{q_\phi(z)}
[f(z)] =
\mathbb{E}_{q_\phi(z)}
\left[
f(z)\nabla_\phi \log q_\phi(z)
\right]
$$

Applying this to the ELBO:

$$
\nabla_\phi \mathcal{L} =
\mathbb{E}_{q_\phi(z|x)}
\left[
(\log P_\theta(x,z)-\log q_\phi(z|x))
\nabla_\phi \log q_\phi(z|x)
\right]
$$

This estimator works even for discrete latent variables.

However:

- gradients have high variance
- optimization is noisy
- training can become unstable

---

# 18. Reparameterization Trick

Instead of sampling directly from:

$$
z \sim q_\phi(z|x)
$$

we rewrite the random variable as a deterministic transformation.

For Gaussian latent variables:

$$
q_\phi(z|x) =
\mathcal{N}(\mu_\phi(x),\sigma_\phi(x)^2 I)
$$

Sample noise:

$$
\epsilon \sim \mathcal{N}(0,I)
$$

Then define:

$$
\boxed{
z=\mu_\phi(x)+\sigma_\phi(x)\epsilon
}
$$

Now:

- randomness comes only from $\epsilon$
- $\epsilon$ does not depend on $\phi$
- $z$ becomes differentiable with respect to encoder parameters

This allows standard backpropagation.

---

# 19. Intuition of the Reparameterization Trick

Instead of:

$$
z \sim q_\phi(z|x)
$$

we write:

$$
z=g_\phi(x,\epsilon)
$$

where:

$$
\epsilon \sim \mathcal{N}(0,I)
$$

The stochasticity is separated from the learnable parameters.

The sampling process becomes:

- fixed random noise
- followed by a differentiable transformation

This dramatically reduces gradient variance compared to REINFORCE.

---

# 20. Amortized Inference

Classical variational inference optimized separate variational parameters for every datapoint.

For each sample $x_i$:

$$
q_i(z)=\mathcal{N}(\mu_i,\sigma_i^2)
$$

and inference required solving a new optimization problem.

This is computationally expensive.

---

## Amortized Inference Idea

Instead of optimizing separate parameters for every datapoint, we train a neural network:

$$
q_\phi(z|x)
$$

called the encoder.

The encoder predicts:

$$
\mu_\phi(x),\sigma_\phi(x)
$$

directly from the input.

Inference becomes:

$$
x \rightarrow (\mu,\sigma)
$$

with a single forward pass.

The optimization cost is shared ("amortized") across the whole dataset.

---

# 21. Variational Autoencoders (VAEs)

A Variational Autoencoder combines:

1. latent variable models
2. variational inference
3. amortized inference
4. neural networks
5. the reparameterization trick

---

# 22. VAE Architecture

A VAE consists of two networks.

---

## Encoder

Approximate posterior:

$$
q_\phi(z|x)
$$

Outputs:

$$
\mu_\phi(x),\sigma_\phi(x)
$$

Defines:

$$
q_\phi(z|x) =
\mathcal{N}(\mu_\phi(x),\sigma_\phi(x)^2 I)
$$

---

## Sampling Step

Sample noise:

$$
\epsilon \sim \mathcal{N}(0,I)
$$

Reparameterize:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\epsilon
$$

---

## Decoder

Generative model:

$$
P_\theta(x|z)
$$

The decoder reconstructs the input from the latent representation.

---

# 23. Full VAE Pipeline

Input image:

$$
x
$$

Encoder outputs:

$$
\mu_\phi(x),\sigma_\phi(x)
$$

Sample latent variable:

$$
z=\mu_\phi(x)+\sigma_\phi(x)\epsilon
$$

Decoder reconstructs:

$$
\hat{x}\sim P_\theta(x|z)
$$

Training optimizes the ELBO.

---

# 24. VAE Objective

The VAE objective is:

$$
\boxed{
\mathcal{L} =
\mathbb{E}_{q_\phi(z|x)}
[\log P_\theta(x|z)] -
D_{KL}(q_\phi(z|x)\|P(z))
}
$$

Usually:

$$
P(z)=\mathcal{N}(0,I)
$$

---

# 25. Reconstruction Term

The reconstruction term:

$$
\mathbb{E}_{q_\phi(z|x)}
[\log P_\theta(x|z)]
$$

encourages accurate reconstructions.

Depending on the decoder distribution:

- Bernoulli likelihood $\rightarrow$ Binary Cross Entropy
- Gaussian likelihood $\rightarrow$ Mean Squared Error

---

# 26. KL Regularization Term

The KL term:

$$
D_{KL}(q_\phi(z|x)\|P(z))
$$

pushes latent representations toward the prior distribution.

This creates:

- smooth latent spaces
- interpolation ability
- meaningful sampling
- generative capabilities

---

# 27. Closed-Form KL for Gaussian VAEs

In standard VAEs:

$$
q_\phi(z|x) =
\mathcal{N}(\mu,\sigma^2 I)
$$

and:

$$
P(z)=\mathcal{N}(0,I)
$$

The KL-divergence has a closed-form solution:

$$
\boxed{
D_{KL}(q_\phi(z|x)\|P(z)) =
\frac{1}{2}
\sum_{j=1}^{d}
\left(
\mu_j^2+\sigma_j^2-\log \sigma_j^2-1
\right)
}
$$

This makes VAE optimization practical.

---

# 28. Approximate Maximum Likelihood Learning

Even in VAEs, the true likelihood:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

is still usually intractable.

VAEs therefore optimize the ELBO instead:

$$
\mathcal{L}\leq \log P_\theta(x)
$$

This is approximate maximum likelihood learning.

Using:

$$
\log P_\theta(x) =
\mathcal{L}
+
D_{KL}(q_\phi(z|x)\|P_\theta(z|x))
$$

if:

$$
q_\phi(z|x)\approx P_\theta(z|x)
$$

then:

$$
\mathcal{L}\approx \log P_\theta(x)
$$

meaning ELBO optimization approximates likelihood maximization.

---

# 29. Key Takeaways

- Latent-variable models introduce hidden variables:

$$
z
$$

to explain structure in the data.

- Exact marginal likelihood is difficult because:

$$
P_\theta(x)=\int P_\theta(x,z)\,dz
$$

- Variational inference approximates the true posterior using:

$$
q_\phi(z|x)
$$

- ELBO provides a tractable lower bound on the likelihood.

- Reparameterization enables low-variance gradient optimization.

- Amortized inference replaces per-datapoint optimization with a neural encoder.

- VAEs combine:
  - latent variable modeling
  - variational inference
  - neural networks
  - reparameterization
  - amortized inference

into a practical generative model.

---

[← Go to Previous Repository - Latent Variable Models](../LatentVariableModels/README.md)

[→ Go to Next Repository - TBD]

[← Back to Main Repository](../README.md)

