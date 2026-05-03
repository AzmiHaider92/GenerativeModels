## Generative vs Discriminative Models

### Discriminative Models

A discriminative model learns:
```
P(y | x)
```

\begin{aligned}
  P(y | x)
\end{aligned}


Given an input x, it directly predicts the label y.

Goal: learn decision boundaries between classes.

---

### Generative Models

A generative model learns:

P(x, y) = P(x | y) * P(y)

That means:
- P(x | y): what data looks like for each class  
- P(y): how common each class is  

Goal: model how the data is generated.

---

## Using a Generative Model for Classification

We use Bayes rule:

P(y | x) ∝ P(x | y) * P(y)

For prediction:

y_hat = argmax_y P(x | y) * P(y)

If classes are uniform:

y_hat = argmax_y P(x | y)

---

## Simple Example: Gaussian Naive Bayes

Assume each class follows a Gaussian:

P(x | y) = N(x | μ_y, Σ_y)

### Training
For each class y:
- Compute mean μ_y
- Compute covariance Σ_y

### Inference
Given a new sample x:

1. Compute log-likelihood for each class:
   log P(x | y)

2. Predict:
   y_hat = argmax_y log P(x | y)

---

## Key Insight

- Discriminative → learns P(y | x) directly  
- Generative → learns P(x | y) and derives P(y | x)

Generative ⇒ Discriminative (via Bayes), but not the opposite

---

## One-Line Summary

A generative model classifies by asking:
"Which class is most likely to have generated this data?"
