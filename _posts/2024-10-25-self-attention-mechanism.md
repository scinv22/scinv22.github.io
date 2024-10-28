---
layout: post
title:  Understanding the Self-Attention Mechanism in Transformers
date: 2024-10-27 21:01:00
description: Demystifying the elegance of the self-attention mechanism in Transformers through mathematics and intuitive explanations.
tags: attention 
categories: artificial-intelligence
---

The **self-attention mechanism** lies at the heart of modern Transformer architectures. It enables models to capture **contextual relationships** between words (or tokens) within a sequence. In this post, we’ll break down the **math behind self-attention** step by step and show how it transforms input embeddings into **context-aware representations**.

---

## 1. Input Embeddings

Let’s assume we have a **5-word sentence**:  
> "The man ate the apple."

Each word is mapped to a **vector representation** (embedding) of fixed dimension $d = 3$. For simplicity, let’s define the embeddings for our 5 words as:

$$
X = 
\begin{bmatrix}
1 & 0 & 1 \\ 
2 & 1 & 0 \\ 
0 & 1 & 1 \\ 
1 & 0 & 1 \\ 
0 & 1 & 2
\end{bmatrix}
$$

---

## 2. Creating Queries, Keys, and Values (Q, K, V)

In the **self-attention mechanism**, we generate **three learned copies** of the input matrix:
- **Query matrix $Q$**  
- **Key matrix $K$**  
- **Value matrix $V$**  

These matrices are obtained by multiplying the input embeddings $X$ with **learnable weight matrices** $W^Q$, $W^K$, and $W^V$:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

For simplicity, let’s assume $Q = X$, $K = X$, and $V = X$, i.e., we directly use the input embeddings as queries, keys, and values.

---

## 3. Calculating the Dot Product Scores

We compute the **dot product of each query with all keys** to measure their similarity. This gives us a **score matrix** $S$, where each element $S_{i,j}$ is the dot product between the **query for token $i$** and the **key for token $j$**.

$$
S = QK^T
$$

For our example:

$$
S = 
\begin{bmatrix}
1 & 0 & 1 \\ 
2 & 1 & 0 \\ 
0 & 1 & 1 \\ 
1 & 0 & 1 \\ 
0 & 1 & 2
\end{bmatrix}
\times
\begin{bmatrix}
1 & 2 & 0 & 1 & 0 \\ 
0 & 1 & 1 & 0 & 1 \\ 
1 & 0 & 1 & 1 & 2
\end{bmatrix}
=
\begin{bmatrix}
2 & 2 & 1 & 2 & 2 \\
2 & 5 & 1 & 2 & 1 \\
1 & 1 & 2 & 1 & 3 \\
2 & 2 & 1 & 2 & 2 \\
2 & 1 & 3 & 2 & 5
\end{bmatrix}

$$

---

## 4. Scaling the Scores by $ \sqrt{d} $

To prevent the scores from becoming too large, we **scale** them by the square root of the embedding dimension $d$:

$$
\hat{S} = \frac{S}{\sqrt{d}}
=
\begin{bmatrix}
1.15 & 1.15 & 0.58 & 1.15 & 1.15 \\
1.15 & 2.89 & 0.58 & 1.15 & 0.58 \\
0.58 & 0.58 & 1.15 & 0.58 & 1.73 \\
1.15 & 1.15 & 0.58 & 1.15 & 1.15 \\
1.15 & 0.58 & 1.73 & 1.15 & 2.89
\end{bmatrix}

$$

Since $d = 3$, we divide each element by $ \sqrt{3} \approx 1.732 $.

---

## 5. Applying Softmax to Get Attention Weights

We apply the **softmax function row-wise** to convert the scores into **probabilities**. The softmax function is defined as:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

After applying softmax to the scaled score matrix, we obtain the **attention weight matrix**:

$$
A = 
\begin{bmatrix}
0.22 & 0.11 & 0.13 & 0.22 & 0.10 \\
0.22 & 0.65 & 0.13 & 0.22 & 0.06 \\
0.12 & 0.06 & 0.22 & 0.12 & 0.18 \\
0.22 & 0.11 & 0.13 & 0.22 & 0.10 \\
0.22 & 0.06 & 0.40 & 0.22 & 0.57
\end{bmatrix}
$$

---

## 6. Calculating the Context Vectors

The **context vector** for each token is computed as a **weighted sum of the value vectors** using the attention weights. Mathematically:

$$
\text{Context Vector for Token } i = A_{i}V
$$

The **full set of context vectors** is:

$$
C = 
\begin{bmatrix}
0.66 & 0.34 & 0.76 \\
1.73 & 0.83 & 0.68 \\
0.38 & 0.47 & 0.83 \\
0.66 & 0.34 & 0.76 \\
0.57 & 1.03 & 1.97
\end{bmatrix}
$$

---

## 7. Summary of the Self-Attention Mechanism

Here is a step-by-step summary of how self-attention works:

1. Generate $Q$, $K$, and $V$ matrices from the input embeddings.
2. Compute the **dot product** of $Q$ and $K^T$ to get the score matrix.
3. **Scale** the scores by $ \sqrt{d} $.
4. Apply **softmax** to convert the scores into attention weights.
5. Use the attention weights to compute **weighted sums** of the value vectors, resulting in the context vectors.

---

## 8. Conclusion

The **self-attention mechanism** allows a model to **dynamically focus** on the most relevant parts of a sequence. Each token builds a **context-aware representation** by attending to other tokens. This ability to model dependencies between words is what makes Transformers so powerful for tasks like **translation, summarization, and language modeling**.

---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention-white.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
