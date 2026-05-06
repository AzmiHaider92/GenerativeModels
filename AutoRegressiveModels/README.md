# Autoregressive Models from Logistic Regression

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
P_\theta(x_2=1 \mid x_1) =
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
P_\theta(x_2=1 \mid x_1) =
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
P_\theta(x_3=1 \mid x_1,x_2) =
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
P(x_n \mid x_{\lt n})
$$

where:

$$
x_{\lt n}=(x_1,\dots,x_{n-1})
$$

we use:

$$
P_\theta(x_n=1 \mid x_{\lt n}) =
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
P(x_1,\dots,x_n) =
\prod_{i=1}^n P(x_i \mid x_{\lt i})
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
P_\theta(x_i \mid x_{\lt i})
$$

with shared neural features.

This allows the network to learn reusable visual patterns.

---

# 7. Masking

Autoregressive models must obey:

$$
x_i \text{ can only depend on } x_{\lt i}
$$

Future pixels are forbidden.

To enforce this, we use masks.

Illegal connections are zeroed out.

Examples:

- PixelCNN → masked convolutions
- MADE → masked linear layers
- GPT → masked self-attention

---

# 8. Multiple Hidden Layers

With a single masked layer, preventing future information leakage is simple:

- remove illegal connections
- future pixels cannot influence current predictions

But with multiple hidden layers:

$$
x \rightarrow h^{(1)} \rightarrow h^{(2)} \rightarrow output
$$

future information could leak indirectly through hidden neurons.

Example:

- hidden neuron \(h_1^{(1)}\) sees future pixel \(x_5\)
- another hidden neuron in layer 2 reads from \(h_1^{(1)}\)
- output for \(x_3\) reads from layer 2

Now:

$$
x_3
$$

indirectly depends on future pixel:

$$
x_5
$$

which violates autoregressive ordering.

---

## The Solution: Degrees

To prevent leakage, each hidden neuron is assigned a degree.

The degree determines:

> the largest pixel index this neuron is allowed to depend on.

Suppose pixels are ordered:

$$
x_1,x_2,x_3,x_4
$$

Now assign degrees to hidden neurons.

Example:

$$
m(h_1)=1
$$

$$
m(h_2)=2
$$

$$
m(h_3)=3
$$

Interpretation:

- neuron with degree 1 may only depend on \(x_1\)
- degree 2 may depend on \(x_1,x_2\)
- degree 3 may depend on \(x_1,x_2,x_3\)

The degree is NOT assigned to pixels.

Pixels already have a natural ordering:

$$
x_1 < x_2 < x_3 < \dots
$$

Degrees are assigned to hidden neurons to control which pixels they may access.

---

## Masking Between Layers

Connections are only allowed if they preserve ordering.

A connection from neuron \(a\) to neuron \(b\) is allowed only if:

$$
m(a) \le m(b)
$$

This ensures information can only flow forward in the allowed autoregressive direction.

---

## Output Layer

To predict:

$$
x_i
$$

the output neuron may only connect to hidden neurons whose degree is strictly smaller than:

$$
i
$$

because:

$$
x_i
$$

may only depend on:

$$
x_1,\dots,x_{i-1}
$$

---

## Final Result

Even with many hidden layers:

$$
x \rightarrow h^{(1)} \rightarrow h^{(2)} \rightarrow \dots \rightarrow output
$$

future pixels can never leak backward.

The network still correctly models:

$$
P(x_i \mid x_{\lt i})
$$

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

# 11. PixelCNN

# Important Insight

PixelCNN does NOT directly generate pixel values.

Instead, the network outputs:

> parameters of probability distributions.

For binary pixels:

the network outputs:

$$
p_i = P_\theta(x_i=1 \mid x_{\lt i})
$$

which is the Bernoulli probability for the next pixel.

For grayscale images:

the network outputs probabilities over:

$$
0,\dots,255
$$

using softmax logits.

---

# Training Objective

During training:

- the full image is already known
- the model predicts probability distributions for every pixel simultaneously
- we compare the predicted probabilities against the true pixel values

The goal is:

> assign high probability to the correct observed pixel.

This is Maximum Likelihood Estimation (MLE).

Equivalently:

> minimize Negative Log Likelihood (NLL).

---

# Binary Pixel Example

Suppose:

$$
x_i \in \{0,1\}
$$

The network predicts:

$$
p_i=P_\theta(x_i=1 \mid x_{\lt i})
$$

Suppose the true pixel value is:

$$
x_i=1
$$

Then the model should assign high probability to:

$$
x_i=1
$$

Loss:

$$
-\log p_i
$$

If:

$$
p_i=0.99
$$

loss is very small.

If:

$$
p_i=0.01
$$

loss becomes very large.

---

## If Ground Truth Is Zero

Suppose:

$$
x_i=0
$$

Then the correct probability is:

$$
P_\theta(x_i=0 \mid x_{\lt i}) = 1-p_i
$$

Loss becomes:

$$
-\log(1-p_i)
$$

---

# Binary Cross Entropy (BCE)

Combining both cases:

$$
\mathcal{L} = - \left[ x_i \log p_i + (1-x_i)\log(1-p_i) \right]
$$

This is Binary Cross Entropy.

Importantly:

> BCE is simply the negative log likelihood of a Bernoulli distribution.

---

# Full Image Loss

PixelCNN models:

$$
P(x)=\prod_i P(x_i \mid x_{\lt i})
$$

Taking log:

$$
\log P(x) =
\sum_i \log P(x_i \mid x_{\lt i})
$$

Total training loss:

$$
\mathcal{L} =
-\sum_i \log P_\theta(x_i \mid x_{\lt i})
$$

The model learns all conditional distributions simultaneously.

---

# Sampling

Sampling is sequential because future pixels depend on previous sampled pixels.

Initially:

$$
x=[?, ?, ?, \dots]
$$

unknown.

---

## First Pixel

The network predicts:

$$
P(x_1)
$$

Example:

$$
P(x_1=1)=0.7
$$

Now we sample:

$$
x_1 \sim \text{Bernoulli}(0.7)
$$

Suppose:

$$
x_1=1
$$

Now the partially generated image becomes:

$$
x=[1, ?, ?, \dots]
$$

---

## Next Pixel

Run the network again.

Now it predicts:

$$
P(x_2 \mid x_1)
$$

Sample:

$$
x_2 \sim P(x_2 \mid x_1)
$$

Suppose:

$$
x_2=0
$$

Now:

$$
x=[1,0,?, ?, \dots]
$$

Continue sequentially:

$$
x_3 \sim P(x_3 \mid x_1,x_2)
$$

$$
x_4 \sim P(x_4 \mid x_1,x_2,x_3)
$$

until the full image is generated.

---

# Important Sampling Insight

The network never outputs the final image directly.

At every step it outputs:

> a probability distribution for the next pixel.

Generation is produced by repeatedly:

1. predicting a distribution
2. sampling from it
3. inserting the sampled pixel back into the image
4. repeating sequentially


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
