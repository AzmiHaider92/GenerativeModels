# Generative Adversarial Networks (GANs)

## 1. Motivation: Why Move Beyond Likelihood?

So far, many generative models are **likelihood-based**:

- Autoregressive models
- Variational Autoencoders (VAEs)
- Normalizing Flows

Their philosophy is:

> If the model assigns high probability to real data, then generated samples should be good.

This sounds reasonable, but it can fail.

---

## Failure Case: Likelihood Encourages Blurry Averages

Suppose the dataset contains only two images:

- a sharp cat facing left
- a sharp cat facing right

So the true distribution is:

$$
P_{data}(x)=0.5\,Cat_{left}+0.5\,Cat_{right}
$$

Now suppose our model uses a Gaussian likelihood:

$$
P_\theta(x|z)=\mathcal N(\mu_\theta(z), \sigma^2 I)
$$

Maximizing likelihood becomes minimizing squared reconstruction error:

$$
\min ||x-\mu_\theta(z)||^2
$$

What minimizes this loss?

The average:

$$
\mu_\theta=
\frac{Cat_{left}+Cat_{right}}{2}
$$

This produces:

a blurry ghost-cat.

Humans say:

> "That looks terrible."

But the objective says:

> "Excellent prediction."

---

## Core Insight

Likelihood rewards:

**matching average pixel values**

Humans care about:

**sharp realistic samples**

These are not the same objective.

This is why early VAEs often produced blurry images.

---

## GAN Motivation

Researchers asked:

> Why optimize likelihood if what we actually care about is realistic samples?

GANs change the problem completely.

Instead of modeling:

$$
P(x)
$$

explicitly, GANs ask:

> Can we generate samples that are indistinguishable from real ones?

---

# 2. Two-Sample Testing

Suppose we have:

Real samples:

$$
x \sim P_{data}
$$

Generated samples:

$$
x \sim P_G
$$

Question:

Do these two sample sets come from the same distribution?

This is a classical **two-sample test**.

Hypotheses:

Null hypothesis:

$$
H_0: P_{data}=P_G
$$

Alternative:

$$
H_1: P_{data}\neq P_G
$$

If we cannot distinguish the two:

the generator succeeded.

---

## Why Not Classical Statistical Tests?

Traditional tests include:

- Kolmogorov–Smirnov
- MMD
- Energy distance

These work in low dimensions.

But images are extremely high-dimensional.

Hand-designed tests do not know what makes an image "look real."

So instead:

**learn the test.**

---

# 3. Discriminator as a Learned Two-Sample Test

Introduce a neural network:

$$
D(x)
$$

that outputs:

probability that a sample is real:

$$
D(x)=P(real|x)
$$

Interpretation:

- $D(x)=1$ → definitely real
- $D(x)=0$ → definitely fake
- $D(x)=0.5$ → cannot tell

The discriminator is a learned statistical hypothesis test.

---

# 4. Generator

The generator maps noise into samples.

Sample latent noise:

$$
z \sim P(z)
$$

Usually:

$$
z\sim \mathcal N(0,I)
$$

Generate:

$$
x=G(z)
$$

This defines the generator distribution:

$$
P_G
$$

Goal:

make:

$$
P_G \approx P_{data}
$$

without explicitly modeling likelihood.

---

# 5. Discriminator Objective

The discriminator solves a binary classification problem.

For real samples:

want:

$$
D(x)\to 1
$$

For fake samples:

want:

$$
D(G(z))\to 0
$$

Binary cross entropy objective:

$$
\max_D V(D,G) =
\mathbb E_{x\sim P_{data}}[\log D(x)]
+
\mathbb E_{z\sim P(z)}[\log(1-D(G(z)))]
$$

Interpretation:

- reward high confidence on real samples
- reward rejecting fake samples

The discriminator learns the best test to separate real and fake distributions.

---

# 6. Generator Objective

The generator wants to fool the discriminator.

So it wants:

$$
D(G(z))\to 1
$$

Original GAN objective:

$$
\min_G \mathbb E_{z\sim P(z)}[\log(1-D(G(z)))]
$$

Combined with discriminator:

$$
\min_G \max_D V(D,G)
$$

This is the GAN minimax game.

Interpretation:

Discriminator says:

> "I can tell fake from real."

Generator says:

> "Try me."

---

# 7. Optimal Discriminator

Fix the generator.

The discriminator maximizes:

$$
P_{data}(x)\log D(x)+P_G(x)\log(1-D(x))
$$

Differentiate with respect to $D(x)$:

$$
\frac{P_{data}(x)}{D(x)} -
\frac{P_G(x)}{1-D(x)}
=0
$$

Solve:

$$
D^*(x)=
\frac{P_{data}(x)}
{P_{data}(x)+P_G(x)}
$$

Interpretation:

The optimal discriminator estimates:

the probability that a sample came from the real distribution.

Special cases:

If:

$$
P_{data}(x)\gg P_G(x)
$$

then:

$$
D^*(x)\approx 1
$$

If:

$$
P_G(x)\gg P_{data}(x)
$$

then:

$$
D^*(x)\approx 0
$$

If equal:

$$
D^*(x)=0.5
$$

meaning perfect confusion.

---

# 8. Connection to Jensen-Shannon Divergence

Substitute the optimal discriminator back into the GAN objective.

Result:

$$
V(G,D^*) =
-\log 4
+
2\,JS(P_{data}||P_G)
$$

So GAN training minimizes:

$$
JS(P_{data}||P_G)
$$

not likelihood.

---

# 9. Jensen-Shannon Divergence

Define the midpoint distribution:

$$
M=\frac{1}{2}(P+Q)
$$

Then:

$$
JS(P||Q) =
\frac{1}{2}KL(P||M)
+
\frac{1}{2}KL(Q||M)
$$

Properties:

- symmetric
- bounded
- stable

Unlike KL:

$$
JS(P||Q)=JS(Q||P)
$$

and:

$$
0\le JS \le \log 2
$$

Interpretation:

Both distributions compare themselves to a shared midpoint.

---

# 10. KL vs JS Behavior

Likelihood-based models minimize:

$$
KL(P_{data}||P_G)
$$

GANs minimize:

$$
JS(P_{data}||P_G)
$$

This leads to very different behavior.

---

## Forward KL (Likelihood / MLE)

Definition:

$$
KL(P_{data}||P_G) =
\int P_{data}(x)
\log \frac{P_{data}(x)}{P_G(x)}dx
$$

Huge penalty if:

$$
P_G(x)\to 0
$$

where real data exists.

Meaning:

missing modes is extremely expensive.

So the model spreads probability broadly.

Behavior:

- mode covering
- averaging
- blurry outputs

Example:

If real data has:

- left cat
- right cat

and generator only produces left cat:

then right cat gets zero probability.

KL heavily punishes this.

So safer solution:

cover both, even imperfectly.

---

## JS (GAN Objective)

JS compares both distributions through the midpoint.

Missing some modes is less aggressively punished.

Generator only needs to produce samples that fool the discriminator.

Behavior:

- sharp realistic samples
- mode seeking
- possible mode collapse

Intuition:

KL says:

> Cover everything.

JS says:

> Look convincing.

---

# 11. Practical Generator Loss

Original generator objective:

$$
\min_G \log(1-D(G(z)))
$$

Problem:

If discriminator becomes very strong:

$$
D(G(z))\approx 0
$$

Then gradients vanish.

Because:

$$
\log(1-0)=0
$$

Flat gradients make learning difficult.

Practical GANs use the non-saturating generator loss:

$$
\max_G \log D(G(z))
$$

equivalently:

$$
\min_G -\log D(G(z))
$$

This gives stronger gradients.

Same equilibrium.

Much better optimization.

---

# 12. GAN Training Algorithm

Training alternates between discriminator and generator updates.

---

## Step 1: Train Discriminator

Sample real batch:

$$
x^{(1)},...,x^{(m)} \sim P_{data}
$$

Sample latent noise:

$$
z^{(1)},...,z^{(m)} \sim P(z)
$$

Generate fake samples:

$$
\tilde x^{(i)}=G(z^{(i)})
$$

Update discriminator:

$$
\max_D
\frac{1}{m}
\sum_i
\log D(x^{(i)})
+
\log(1-D(\tilde x^{(i)}))
$$

---

## Step 2: Train Generator

Sample fresh noise:

$$
z^{(1)},...,z^{(m)} \sim P(z)
$$

Generate fake samples:

$$
\tilde x^{(i)}=G(z^{(i)})
$$

Update generator:

$$
\min_G
-\frac{1}{m}
\sum_i \log D(\tilde x^{(i)})
$$

---

Repeat until convergence.

At equilibrium:

$$
P_G=P_{data}
$$

and discriminator cannot distinguish:

$$
D(x)=0.5
$$

for both real and fake samples.

---

# Final Intuition

GANs do not explicitly model density.

They solve a game:

- discriminator learns to distinguish real vs fake
- generator learns to fool discriminator

Instead of asking:

> "Did I assign high probability to the data?"

GANs ask:

> "Do my samples look real?"




---

[← Go to Previous Repository - Normalizing Flows](../NormalizingFlows/README.md)

[→ Go to Next Repository - Energy Based Models](../EBMs/README.md)

[← Back to Main Repository](../README.md)
