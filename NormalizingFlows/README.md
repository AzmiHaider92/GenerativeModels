# Normalizing Flows

Normalizing Flows are generative models that combine the best properties of two earlier model families:

- **Autoregressive models:** exact likelihood, but slow sampling
- **Latent variable models (VAEs):** latent representations, but approximate inference and intractable likelihood

Normalizing flows aim to get:

- Exact likelihood computation
- Efficient sampling
- Meaningful latent representations
- Exact inference through invertibility

The key mathematical tool behind flows is the **change of variables formula**.

---

# 1. Core Idea

Start with a simple random variable:

$$
z \sim p_Z(z)
$$

and transform it using an invertible function:

$$
x = f_\theta(z)
$$

Because the function is invertible:

$$
z = f_\theta^{-1}(x)
$$

This gives a mapping between latent space and data space:

$$
z \leftrightarrow x
$$

The key question:

> If we know the density of $z$, what is the density of $x$?

---

# 2. Change of Variables (1D Intuition)

Suppose:

$$
x = f(z)
$$

with $f$ invertible and differentiable.

A small interval around $z$:

$$
[z, z+dz]
$$

contains probability:

$$
p_Z(z)\,dz
$$

After transformation, this becomes:

$$
[x, x+dx]
$$

with probability:

$$
p_X(x)\,dx
$$

Probability must be conserved:

$$
p_Z(z)\,dz = p_X(x)\,dx
$$

Rearranging:

$$
p_X(x)=p_Z(z)\frac{dz}{dx}
$$

Since density must remain positive:

$$
p_X(x)=p_Z(z)\left|\frac{dz}{dx}\right|
$$

Substituting:

$$
z=f^{-1}(x)
$$

gives:

$$
p_X(x)=p_Z(f^{-1}(x))
\left|\frac{d f^{-1}(x)}{dx}\right|
$$

This is the 1D change of variables formula.

---

# 3. Intuition: Stretching and Compression

Suppose:

$$
x=2z
$$

Then:

$$
dx=2dz
$$

so:

$$
dz=\frac12 dx
$$

Therefore:

$$
p_X(x)=\frac12 p_Z(z)
$$

Interpretation:

The same probability mass is spread over twice the space, so density decreases.

---

If instead:

$$
x=\frac12 z
$$

then:

$$
dx=\frac12 dz
$$

so density increases.

Interpretation:

Compressing space increases density.

The derivative tells us how much local stretching/compression occurs.

---

# 4. Multi-Dimensional Case

Now let:

$$
z \in \mathbb{R}^D
$$

and:

$$
x=f(z)
$$

Now we are transforming volumes instead of intervals.

A small volume in latent space:

$$
dz_1dz_2\dots dz_D
$$

maps to a volume in data space.

---

# 5. Jacobian Matrix

Locally, any smooth nonlinear function looks linear:

$$
f(z+\Delta z)\approx f(z)+J_f(z)\Delta z
$$

where the Jacobian is:

$$
J_f(z)=
\frac{\partial x}{\partial z} =
\begin{bmatrix}
\frac{\partial x_1}{\partial z_1} & \cdots & \frac{\partial x_1}{\partial z_D} \\
\vdots & \ddots & \vdots \\
\frac{\partial x_D}{\partial z_1} & \cdots & \frac{\partial x_D}{\partial z_D}
\end{bmatrix}
$$

The Jacobian tells us how each output dimension changes with respect to each input dimension.

---

# 6. Why the Determinant?

A linear transformation scales volume by the determinant.

Example:

$$
A=
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix}
$$

This stretches:

- x-axis by 2
- y-axis by 3

Total area scaling:

$$
2 \times 3 = 6
$$

which equals:

$$
\det(A)=6
$$

So determinant measures local volume expansion/compression.

---

For a nonlinear transformation:

$$
x=f(z)
$$

the local volume scaling is:

$$
|\det J_f(z)|
$$

Thus:

$$
dx = |\det J_f(z)|\,dz
$$

or:

$$
dz=\frac{dx}{|\det J_f(z)|}
$$

---

# 7. Full Change of Variables Formula

Probability conservation:

$$
p_Z(z)\,dz=p_X(x)\,dx
$$

Substitute:

$$
dz=\frac{dx}{|\det J_f(z)|}
$$

giving:

$$
p_X(x)=p_Z(z)\frac{1}{|\det J_f(z)|}
$$

Since:

$$
z=f^{-1}(x)
$$

we get:

$$
p_X(x)=p_Z(f^{-1}(x))
\left|
\det J_{f^{-1}}(x)
\right|
$$

Equivalent form:

$$
p_X(x)=
p_Z(z)
\left|
\det J_f(z)
\right|^{-1}
$$

This is the central equation behind normalizing flows.

---

# 8. Log Likelihood Form

Taking logarithms:

$$
\log p_X(x) =
\log p_Z(z) -
\log |\det J_f(z)|
$$

This has two parts:

### Base Density

How likely the latent code is:

$$
\log p_Z(z)
$$

Usually:

$$
z \sim \mathcal N(0,I)
$$

---

### Volume Correction

How much the transformation expands or compresses space:

$$
-\log |\det J_f(z)|
$$

- Expansion → density decreases
- Compression → density increases

---

# 9. Normalizing Flows

Now the idea becomes natural.

Choose a simple base distribution:

$$
z \sim \mathcal N(0,I)
$$

Learn an invertible neural network:

$$
x=f_\theta(z)
$$

Then:

### Sampling

Easy:

$$
z \sim \mathcal N(0,I)
$$

then:

$$
x=f_\theta(z)
$$

---

### Exact Likelihood

Given data:

$$
x
$$

compute inverse:

$$
z=f_\theta^{-1}(x)
$$

then:

$$
\log p(x)=
\log p(z)
-\log |\det J_{f_\theta}(z)|
$$

Exact maximum likelihood training becomes possible.

---

# 10. Composition of Transformations

One transformation may not be expressive enough.

Compose many simple invertible transformations:

$$
z_1=f_1(z_0)
$$

$$
z_2=f_2(z_1)
$$

$$
\vdots
$$

$$
z_K=f_K(z_{K-1})
$$

Overall:

$$
x=f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)
$$

---

## Sampling

Sample:

$$
z_0 \sim p(z_0)
$$

Apply transformations:

$$
z_0 \rightarrow z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow x
$$

---

## Likelihood

Invert:

$$
x \rightarrow z_{K-1} \rightarrow \cdots \rightarrow z_0
$$

Likelihood becomes:

$$
\log p(x) =
\log p(z_0) -
\sum_{k=1}^{K}
\log
\left|
\det J_{f_k}(z_{k-1})
\right|
$$

because:

$$
\det(AB)=\det(A)\det(B)
$$

and:

$$
\log(ab)=\log a+\log b
$$

---

# 11. Why "Flow"?

Probability density is continuously reshaped through transformations.

A simple Gaussian density gradually flows into the complex data distribution.

Hence:

**Normalizing Flow**

---

# 12. Advantages

Compared to VAEs:

- Exact likelihood instead of ELBO
- Exact inference via inversion
- Direct latent representation
- Efficient sampling

Compared to autoregressive models:

- Parallel sampling
- Latent structure
- Exact likelihood

---

# 13. The Catch

This sounds perfect, but there are constraints.

Each transformation must be:

- Invertible
- Differentiable
- Efficient to invert
- Efficient to compute determinant

The determinant requirement is the major challenge.

A generic neural network has an expensive Jacobian determinant.

So practical flow models use carefully designed layers.

Examples:

- NICE
- RealNVP
- Glow
- Coupling layers
- Continuous Normalizing Flows

---

# Final Intuition

Normalizing flows do not generate data probabilistically step-by-step like VAEs.

Instead:

They start from a simple known distribution and **warp probability density into the data distribution using invertible transformations**, while keeping exact probability computation possible.




---

[← Go to Previous Repository - Latent Variable Models / VAE](../LatentVariableModels/README.md)

[→ Go to Next Repository - GANs](../GANs/README.md)

[← Back to Main Repository](../README.md)
