# Probabilistic Modeling – Notes

## 1. What is Probabilistic Modeling?

Probabilistic modeling is about defining a **distribution over data**.

Given a random variable \( X \), we model:
\[
p(X)
\]

For multiple variables:
\[
p(x_1, x_2, \dots, x_n)
\]

This is called the **joint distribution**.

---

## 2. Example: Distribution over a Single Pixel

A pixel in an RGB image has values:
\[
(R, G, B), \quad R,G,B \in \{0, \dots, 255\}
\]

Total number of possible colors:
\[
256^3
\]

To define the distribution:
- Assign a probability to each \((r,g,b)\)
- Probabilities must sum to 1

Number of parameters:
\[
256^3 - 1
\]

👉 This is a **categorical distribution over colors**

---

## 3. Example: Binary Image

Let an image be:
\[
I = (x_1, x_2, \dots, x_n), \quad x_i \in \{0,1\}
\]

Number of possible images:
\[
2^n
\]

Number of parameters for full joint:
\[
2^n - 1
\]

---

## 4. Curse of Dimensionality

As dimension increases:

- Number of configurations grows **exponentially**
- Number of parameters becomes intractable
- Data becomes sparse

Example:
- \(28 \times 28\) image → \(n = 784\)
- Number of images:
\[
2^{784}
\]

👉 Impossible to model directly

---

## 5. Chain Rule of Probability

We can always factorize:
\[
p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \dots, x_{i-1})
\]

But this does **not reduce complexity**:
\[
2^n - 1 \text{ parameters}
\]

---

## 6. Conditional Independence

We simplify by assuming:

\[
x_i \perp (x_1,\dots,x_{i-2}) \mid x_{i-1}
\]

So:
\[
p(x_i \mid x_1,\dots,x_{i-1}) = p(x_i \mid x_{i-1})
\]

---

## 7. Parameter Reduction

### Full joint:
\[
2^n - 1
\]

### With conditional independence (Markov assumption):

- \(p(x_1)\): 1 parameter  
- Each \(p(x_i \mid x_{i-1})\): 2 parameters  

Total:
\[
2n - 1
\]

---

## 8. Why This Helps

- **Exponential → Linear complexity**
- Fewer parameters
- Easier learning
- Better generalization

---

## 9. Bayesian Networks

A **Bayesian Network** is:

- A **Directed Acyclic Graph (DAG)**
- Nodes = variables
- Edges = dependencies

Factorization:
\[
p(x_1,\dots,x_n) = \prod_i p(x_i \mid \text{parents}(x_i))
\]

---

## 10. Conditional Independence in Graphs

Each node satisfies:

> A variable is independent of all others given its parents.

Example (chain):
\[
x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \dots \rightarrow x_n
\]

---

## 11. Neural Models (Reminder)

Discriminative models:
\[
P_\theta(Y \mid x)
\]

- \( \theta \): model parameters  
- Output via:
  - sigmoid (binary)
  - softmax (multi-class)

They learn a function:
\[
x \rightarrow P_\theta(Y \mid x)
\]

---

## 12. Neural Parameterization of Probabilities

Instead of storing probability tables:

\[
p(x_i \mid \text{parents}) \approx P_\theta(x_i \mid \text{parents})
\]

---

## 13. Final Unified View

\[
p(x_1,\dots,x_n) = \prod_i P_\theta(x_i \mid \text{parents}(x_i))
\]

- Bayesian Network → structure
- Neural model → function

---

## 14. Key Takeaways

- Full joint distributions scale exponentially
- Conditional independence reduces complexity
- Bayesian networks encode structure
- Neural networks replace tables with functions
- This enables scalable probabilistic modeling

---

## 15. What’s Next

We will build on this to derive:

👉 Autoregressive models  
👉 Neural generative models for high-dimensional data

---
