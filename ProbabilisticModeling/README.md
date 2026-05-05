# Probabilistic Modeling – Notes

## 1. What is Probabilistic Modeling?

Probabilistic modeling defines a **distribution over data**.

For a random variable:

$$
p(X)
$$

For multiple variables:

$$
p(x_1, x_2, \dots, x_n)
$$

This is called the **joint distribution**.

---

## 2. Example: Distribution over a Single Pixel

A pixel has RGB values:

$$
(R, G, B), \quad R,G,B \in \{0, \dots, 255\}
$$

Number of possible colors:

$$
256^3
$$

To define the distribution:
- Assign a probability to each $(r,g,b)$
- Probabilities must sum to 1

Number of parameters:

$$
256^3 - 1
$$

👉 This is a **categorical distribution over colors**

---

## 3. Example: Binary Image

Let an image be:

$$
I = (x_1, x_2, \dots, x_n), \quad x_i \in \{0,1\}
$$

Number of possible images:

$$
2^n
$$

Number of parameters for full joint:

$$
2^n - 1
$$

---

## 4. Curse of Dimensionality

As dimension increases:
- Number of configurations grows **exponentially**
- Number of parameters becomes intractable
- Data becomes sparse

Example:

$$
28 \times 28 \Rightarrow n = 784
$$

$$
\text{Number of images} = 2^{784}
$$

👉 Impossible to model directly

---

## 5. Chain Rule of Probability

We can always factorize:

$$
p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \dots, x_{i-1})
$$

But this does **not reduce complexity**:

$$
2^n - 1 \text{ parameters}
$$

---

## 6. Conditional Independence

Assume:

$$
x_i \perp (x_1,\dots,x_{i-2}) \mid x_{i-1}
$$

So:

$$
p(x_i \mid x_1,\dots,x_{i-1}) = p(x_i \mid x_{i-1})
$$

---

## 7. Parameter Reduction

### Full joint:

$$
2^n - 1
$$

### Markov assumption:

- $p(x_1)$ → 1 parameter  
- Each $p(x_i \mid x_{i-1})$ → 2 parameters  

Total:

$$
2n - 1
$$

👉 Exponential → Linear reduction

---

## 8. Why This Helps

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

$$
p(x_1,\dots,x_n) = \prod_i p(x_i \mid \text{parents}(x_i))
$$

---

## 10. Neural Models (Reminder)

Discriminative models:

$$
P_\theta(Y \mid x)
$$

They learn a function:

$$
x \rightarrow P_\theta(Y \mid x)
$$

Using:
- Sigmoid (binary)
- Softmax (multi-class)

---

## 11. Neural Probabilistic Modeling

Instead of tables:

$$
p(x_i \mid \text{parents}) \approx P_\theta(x_i \mid \text{parents})
$$

---

## 12. Final Unified View

$$
p(x_1,\dots,x_n) = \prod_i P_\theta(x_i \mid \text{parents}(x_i))
$$

- Bayesian Network → structure  
- Neural model → function  

---

## 13. Key Takeaways

- Full joint = exponential complexity  
- Conditional independence = tractable modeling  
- Bayesian networks encode structure  
- Neural networks replace tables with functions  

---


[← Go to Previous Repository - Probabilistic Modeling](../ProbabilisticModeling/README.md)

[→ Go to Next Repository - Auto Regressive Models](../AutoRegressiveModels/README.md)

[← Back to Main Repository](../README.md)
