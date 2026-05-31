# Diffusion Models

## 1. Motivation

Diffusion models can be viewed from three equivalent perspectives:

1. Latent variable models trained using variational inference.
2. Score-based models trained using denoising score matching.
3. A learned reverse diffusion process that gradually converts noise into data.

---

## 2. Connection to Previous Lessons

### Energy-Based Models

EBMs learn an energy function:

$$
E_\theta(x)
$$

and define:

$$
p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta}
$$

Sampling requires expensive MCMC methods.

### Score-Based Models

Score models learn:

$$
s_\theta(x)\approx\nabla_x\log p(x)
$$

### Denoising Score Matching

$$
\tilde x=x+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

The model learns:

$$
\nabla_{\tilde x}\log p_\sigma(\tilde x).
$$

### Annealed Langevin Dynamics

$$
\sigma_1>\sigma_2>\cdots>\sigma_L
$$

Sampling starts from Gaussian noise and gradually denoises.

---

# 3. Forward Diffusion Process

The forward process is:

$$
q(x_t|x_{t-1}) =
\mathcal N
\left(
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right)
$$

Equivalent sampling form:

$$
x_t =
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{1-\alpha_t}\epsilon
$$

where

$$
\epsilon\sim\mathcal N(0,I).
$$

---

## Markov Chain

$$
q(x_{1:T}|x_0) =
\prod_{t=1}^{T}
q(x_t|x_{t-1})
$$

$$
x_0
\rightarrow
x_1
\rightarrow
\cdots
\rightarrow
x_T
$$

After many steps:

$$
x_T\approx\mathcal N(0,I).
$$

---

# 4. Closed-Form Noising Equation

$$
q(x_t|x_0) =
\mathcal N
\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
$$

where

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

Equivalent form:

$$
x_t =
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

---

# 5. Reverse Diffusion Process

The ideal reverse process is:

$$
q(x_{t-1}|x_t).
$$

This distribution is unknown.

Instead we learn:

$$
p_\theta(x_{t-1}|x_t).
$$

Sampling:

$$
x_T\sim\mathcal N(0,I)
$$

followed by

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0.
$$

---

# 6. Why Introduce $$ q(x_{t-1}|x_t,x_0) $$ ?

During training we know the clean image:

$$
x_0.
$$

The forward chain is:

$$
x_0
\rightarrow
x_{t-1}
\rightarrow
x_t.
$$

We can therefore ask:

$$
q(x_{t-1}|x_t,x_0).
$$

Using Bayes:

$$
q(x_{t-1}|x_t,x_0)
\propto
q(x_t|x_{t-1})
q(x_{t-1}|x_0).
$$

Since both terms are Gaussian:

$$
q(x_{t-1}|x_t,x_0) =
\mathcal N
(
\tilde\mu_t,
\tilde\beta_t I
).
$$

---

# 7. Reverse Mean

The posterior mean has the form:

$$
\tilde\mu_t =
A_t x_0
+
B_t x_t.
$$

It combines information from the clean image and the noisy image.

---

# 8. Variational Inference View

Latent variables:

$$
x_1,x_2,\ldots,x_T.
$$

Encoder:

$$
q(x_{1:T}|x_0).
$$

Decoder:

$$
p_\theta(x_{0:T}) =
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}|x_t).
$$

ELBO:

$$
\log p_\theta(x_0)
\ge
\mathbb E_q
\left[
\log p_\theta(x_{0:T}) -
\log q(x_{1:T}|x_0)
\right].
$$

This becomes a sum of KL terms:

$$
D_{KL}
\left(
q(x_{t-1}|x_t,x_0)
\|
p_\theta(x_{t-1}|x_t)
\right).
$$

---

# 9. Noise Prediction

Forward equation:

$$
x_t =
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

The network predicts:

$$
\epsilon_\theta(x_t,t).
$$

Training loss:

$$
L =
\mathbb E
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
$$

Recover the clean image:

$$
\hat x_0 =
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}
{\sqrt{\bar\alpha_t}}.
$$

Reverse mean:

$$
\mu_\theta(x_t,t) =
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t -
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

Sampling step:

$$
x_{t-1} =
\mu_\theta(x_t,t)
+
\sigma_t z,
\qquad
z\sim\mathcal N(0,I).
$$

---

# 10. Connection to Score Matching

For Gaussian corruption:

$$
\nabla_{x_t}\log p_t(x_t) =
-\frac{\epsilon}{\sigma_t}.
$$

Therefore:

$$
s_\theta(x_t,t) =
-\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar\alpha_t}}.
$$

Predicting noise is equivalent to predicting the score.

---

# Key Takeaways

- Forward process: fixed Gaussian noising process.
- Reverse process: learned denoising process.
- Training uses the tractable posterior:

$$
q(x_{t-1}|x_t,x_0).
$$

- Diffusion can be derived from variational inference.
- Diffusion can be derived from denoising score matching.
- Noise prediction and score prediction are equivalent.
