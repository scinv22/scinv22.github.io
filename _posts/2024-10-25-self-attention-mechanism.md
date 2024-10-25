---
layout: post
title:  Understanding the Self-Attention Mechanism in Transformers
date: 2024-10-25 21:01:00
description: this is what included images could look like
tags: formatting images
categories: sample-posts
---

The self-attention mechanism lies at the heart of modern Transformer architectures, enabling models to capture contextual relationships between words (or tokens) in a sequence. In this post, we’ll break down the math behind self-attention step by step and show how it transforms input embeddings into context-aware representations.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention-white.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

In scaled dot-product attention, the dot product is divided by the square root of the dimensionality of the vectors
#Images can be made zoomable.
#Simply add `data-zoomable` to `<img>` tags that you want to make zoomable.
