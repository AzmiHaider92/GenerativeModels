## Generative vs Discriminative Models

### Discriminative Models

A discriminative model learns the conditional distribution (Posterior):

$$
P(y \mid x)
$$

Given an input $x$, it directly predicts the label $y$.

**Goal:** learn decision boundaries between classes.

---

### Generative Models

A generative model learns the data distribution:

$$
P(x, y) = P(x \mid y) P(y)
$$

That means:
- $P(x \mid y)$: what data looks like for each class (Likelihood)
- $P(y)$: class prior  

**Goal:** model how the data is generated.

---

## Using a Generative Model for Classification

We can recover the discriminative quantity using Bayes’ rule:

$$
P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}
$$

For classification:

$$
\hat{y} = \arg\max_y P(x \mid y) P(y)
$$

If classes are uniform:

$$
\hat{y} = \arg\max_y P(x \mid y)
$$

### Example: Gaussian Naive Bayes

Assume:

$$
P(x \mid y) = \mathcal{N}(x \mid \mu_y, \Sigma_y)
$$

### Training
For each class $y$:
- Estimate $\mu_y$
- Estimate $\Sigma_y$

### Inference

Given a new sample $x$:

$$
\hat{y} = \arg\max_y \log P(x \mid y)
$$

---

## Key Insight

- Discriminative → learns $P(y \mid x)$ directly  
- Generative → learns $P(x \mid y)$ and derives $P(y \mid x)$  

$$
\text{Generative} \Rightarrow \text{Discriminative (via Bayes)}
$$

---

## One-Line Summary

> A generative model classifies by asking:  
> **"Which class is most likely to have generated this data?"**
