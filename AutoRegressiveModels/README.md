# Autoregressive Models

# 1. Logistic Regression

Suppose we want to predict a binary variable:

$$
y \in \{0,1\}
$$

Logistic regression models:

$$
P_\theta(y=1 \mid x)
$$

using:

$$
P_\theta(y=1 \mid x)=\sigma(wx+b)
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The model outputs a probability between 0 and 1.

---

# 2. Turning Logistic Regression into a Generative Model

Instead of predicting a label from an input, we can predict one variable from previous variables.

Suppose an MNIST image is flattened into binary pixels:

$$
x=(x_1,x_2,\dots,x_n)
$$

where:

$$
x_i \in \{0,1\}
$$

Now we model conditional probabilities like:

$$
P(x_2 \mid x_1)
$$

using logistic regression:

$$
P_\theta(x_2=1 \mid x_1)
=
\sigma(wx_1+b)
$$

---

# 3. Number of Parameters

## Example 1

For:

$$
P(x_2 \mid x_1)
$$

we use:

$$
P_\theta(x_2=1 \mid x_1)
=
\sigma(wx_1+b)
$$

Parameters:

- one weight
- one bias

Total:

$$
2
$$

parameters.

---

## Example 2

For:

$$
P(x_3 \mid x_1,x_2)
$$

we use:

$$
P_\theta(x_3=1 \mid x_1,x_2)
=
\sigma(w_1x_1+w_2x_2+b)
$$

Parameters:

- two weights
- one bias

Total:

$$
3
$$

parameters.

---

## General Case

For:

$$
P(x_n \mid x_{<n})
$$

where:

$$
x_{<n}=(x_1,\dots,x_{n-1})
$$

we use:

$$
P_\theta(x_n=1 \mid x_{<n})
=
\sigma\left(\sum_{i=1}^{n-1} w_i x_i + b\right)
$$

Number of parameters:

- \(n-1\) weights
- 1 bias

Total:

$$
n
$$

parameters.

---

# 4. Full Autoregressive Distribution

Using the chain rule:

$$
P(x_1,\dots,x_n)
=
\prod_{i=1}^n P(x_i \mid x_{<i})
$$

The model predicts pixels sequentially.

Examples:

$$
P(x_1)
$$

$$
P(x_2 \mid x_1)
$$

$$
P(x_3 \mid x_1,x_2)
$$

etc.

---

# 5. Parameter Count

Total parameters:

$$
1+2+3+\dots+n
$$

which equals:

$$
\frac{n(n+1)}{2}
$$

Complexity:

$$
O(n^2)
$$

This is dramatically smaller than directly modeling the full joint distribution:

$$
P(x_1,\dots,x_n)
$$

which would require:

$$
2^n
$$

possible binary configurations.

---

# 6. Neural Autoregressive Models

Instead of separate logistic regressions, we use a neural network with shared parameters.

A hidden layer:

$$
h = \phi(Wx+b)
$$

where:

- \(W\) is shared
- different outputs use different subsets of information

The model becomes:

$$
P_\theta(x_i \mid x_{<i})
$$

with shared neural features.

This allows the network to learn reusable visual patterns.

---

# 7. Masking

Autoregressive models must obey:

$$
x_i \text{ can only depend on } x_{<i}
$$

Future pixels are forbidden.

To enforce this, we use masks.

Illegal connections are zeroed out.

Examples:

- PixelCNN → masked convolutions
- MADE → masked linear layers
- GPT → masked self-attention

![autoregressive_image_modeling](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/5/5b/xi_img.png/500px-xi_img.png)  


---

# 8. Multiple Hidden Layers

With multiple hidden layers, information could leak indirectly from future pixels.

To prevent this, hidden neurons are assigned degrees.

A neuron with degree \(k\) may only depend on:

$$
x_1,\dots,x_k
$$

Connections are only allowed if ordering constraints are respected.

This preserves:

$$
P(x_i \mid x_{<i})
$$

throughout the entire network.

---

# 9. RNNs as Autoregressive Models

RNNs are naturally autoregressive.

They process sequences one step at a time.

Hidden state update:

$$
h_t = f(h_{t-1}, x_t)
$$

Prediction:

$$
P_\theta(x_{t+1} \mid h_t)
$$

Generation process:

1. read previous token/pixel
2. update hidden state
3. predict next variable
4. sample
5. repeat

---

# 10. RNN Pros and Cons

## Pros

- shared parameters through time
- naturally sequential
- handles variable-length sequences

## Cons

- vanishing/exploding gradients
- difficult long-range dependencies
- sequential computation is slow
- hidden-state bottleneck

LSTMs and GRUs improved stability using gates.

Transformers later replaced recurrence with masked attention.

---

# 11. PixelCNN Training

PixelCNN models:

$$
P(x)=\prod_i P(x_i \mid x_{<i})
$$

using masked convolutions.

During training:

- the full image is given
- all conditionals are predicted simultaneously

---

# 12. PixelCNN Sampling

Sampling is sequential.

Initially:

$$
x=[?, ?, ?, \dots]
$$

## First Pixel

Model predicts:

$$
P(x_1)
$$

Example:

$$
P(x_1=1)=0.7
$$

Then sample:

$$
x_1 \sim \text{Bernoulli}(0.7)
$$

Suppose:

$$
x_1=1
$$

Now:

$$
x=[1, ?, ?, \dots]
$$

---

## Next Pixel

Model predicts:

$$
P(x_2 \mid x_1)
$$

Sample:

$$
x_2 \sim P(x_2 \mid x_1)
$$

Repeat sequentially until the full image is generated.

---

# 13. Important Insight

The network does NOT output the pixel itself.

It outputs:

> parameters of a probability distribution.

Examples:

- Bernoulli probability for binary pixels
- softmax logits for grayscale values
- distributions over RGB channels

Sampling happens separately.

---

# 14. Training Objective

The model learns by maximizing probability assigned to the true pixel value.

Equivalent to minimizing negative log likelihood.

---

## Binary Pixel Loss

Suppose:

$$
p_i=P_\theta(x_i=1 \mid x_{<i})
$$

If ground truth is:

$$
x_i=1
$$

loss is:

$$
-\log p_i
$$

If:

$$
x_i=0
$$

loss is:

$$
-\log(1-p_i)
$$

Combined:

$$
\mathcal{L}
=
-
\left[
x_i \log p_i
+
(1-x_i)\log(1-p_i)
\right]
$$

This is Binary Cross Entropy (BCE).

---

# 15. Full Image Loss

Since:

$$
P(x)=\prod_i P(x_i \mid x_{<i})
$$

taking log gives:

$$
\log P(x)
=
\sum_i \log P(x_i \mid x_{<i})
$$

Final training loss:

$$
\mathcal{L}
=
-\sum_i \log P_\theta(x_i \mid x_{<i})
$$

---

# 16. Core Concept

Autoregressive models learn:

> probability distributions over the next variable.

They do NOT learn deterministic outputs.

Generation works by:

1. predicting a distribution
2. sampling from it
3. feeding sampled values back into the model
4. repeating sequentially

This is the foundation of autoregressive generative modeling.
