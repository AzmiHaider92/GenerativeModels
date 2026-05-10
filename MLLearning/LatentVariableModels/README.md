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
P_\theta(z|x)
=
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
=
\log
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
\tilde{x}
=
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
