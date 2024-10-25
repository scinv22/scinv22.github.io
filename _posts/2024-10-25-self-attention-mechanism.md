---
layout: post
title:  Understanding the Self-Attention Mechanism in Transformers
date: 2024-10-25 21:01:00
description: this is what included images could look like
tags: formatting images
categories: sample-posts
---

The **self-attention mechanism** lies at the heart of modern Transformer architectures. It enables models to capture **contextual relationships** between words (or tokens) within a sequence. In this post, we’ll break down the **math behind self-attention** step by step and show how it transforms input embeddings into **context-aware representations**.

---

## 1. Input Embeddings

Let’s assume we have a **5-word sentence**:  
> "The cat ate the fish."

Each word is mapped to a **vector representation** (embedding) of fixed dimension \(d = 3\). For simplicity, let’s define the embeddings for our 5 words as:

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
- **Query matrix \(Q\)**  
- **Key matrix \(K\)**  
- **Value matrix \(V\)**  

These matrices are obtained by multiplying the input embeddings \(X\) with **learnable weight matrices** \(W^Q\), \(W^K\), and \(W^V\):

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

For simplicity, let’s assume \(Q = X\), \(K = X\), and \(V = X\), i.e., we directly use the input embeddings as queries, keys, and values.

---

## 3. Calculating the Dot Product Scores

We compute the **dot product of each query with all keys** to measure their similarity. This gives us a **score matrix** \(S\), where each element \(S_{i,j}\) is the dot product between the **query for token \(i\)** and the **key for token \(j\)**.

$$
S = Q \cdot K^T
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
\cdot
\begin{bmatrix}
1 & 2 & 0 & 1 & 0 \\ 
0 & 1 & 1 & 0 & 1 \\ 
1 & 0 & 1 & 1 & 2
\end{bmatrix}
=
\begin{bmatrix}
2 & 2 & 2 & 2 & 3 \\
2 & 5 & 1 & 2 & 2 \\
1 & 1 & 2 & 1 & 3 \\
2 & 2 & 2 & 2 & 3 \\
1 & 2 & 2 & 1 & 5
\end{bmatrix}
$$

---

## 4. Scaling the Scores by \( \sqrt{d} \)

To prevent the scores from becoming too large, we **scale** them by the square root of the embedding dimension \(d\):

$$
\hat{S} = \frac{S}{\sqrt{d}}
$$

Since \(d = 3\), we divide each element by \( \sqrt{3} \approx 1.732 \).

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
0.29 & 0.16 & 0.16 & 0.29 & 0.09 \\
0.16 & 0.29 & 0.16 & 0.16 & 0.16 \\
0.16 & 0.16 & 0.29 & 0.16 & 0.16 \\
0.29 & 0.16 & 0.16 & 0.29 & 0.09 \\
0.09 & 0.16 & 0.16 & 0.09 & 0.29
\end{bmatrix}
$$

---

## 6. Calculating the Context Vectors

The **context vector** for each token is computed as a **weighted sum of the value vectors** using the attention weights. Mathematically:

$$
\text{Context Vector for Token } i = \sum_j A_{i,j} \cdot V_j
$$

For example, the **context vector for the first token** ("The (1)") is:

$$
C_1 = 0.29 \cdot [1, 0, 1] + 0.16 \cdot [2, 1, 0] + 0.16 \cdot [0, 1, 1] + 0.29 \cdot [1, 0, 1] + 0.09 \cdot [0, 1, 2]
$$

Evaluating:

$$
C_1 = [0.90, 0.41, 0.92]
$$

The **full set of context vectors** is:

$$
C = 
\begin{bmatrix}
0.90 & 0.41 & 0.92 \\
0.90 & 0.61 & 0.80 \\
0.64 & 0.61 & 0.93 \\
0.90 & 0.41 & 0.92 \\
0.50 & 0.61 & 0.92
\end{bmatrix}
$$

---

## 7. Summary of the Self-Attention Mechanism

Here is a step-by-step summary of how self-attention works:

1. Generate \(Q\), \(K\), and \(V\) matrices from the input embeddings.
2. Compute the **dot product** of \(Q\) and \(K^T\) to get the score matrix.
3. **Scale** the scores by \( \sqrt{d} \).
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
