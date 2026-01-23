# Level 3: Functions - The ML Primitive

## Overview

Welcome to Level 3, where we explore **functions** - arguably the most fundamental concept in machine learning. If you understand functions deeply, you understand the essence of what ML models actually do.

## Why Functions Are the ML Primitive

At its core, every machine learning model is a function. When you strip away all the fancy terminology - neural networks, transformers, gradient boosting - what remains is simply a mathematical function that maps inputs to outputs.

$$\text{Model}: \text{Input} \rightarrow \text{Output}$$

Consider these examples:

| ML Task | Input | Output | Function Type |
|---------|-------|--------|---------------|
| Image Classification | Pixel values | Class probabilities | $f: \mathbb{R}^{n \times m \times 3} \rightarrow [0,1]^k$ |
| Regression | Features | Predicted value | $f: \mathbb{R}^n \rightarrow \mathbb{R}$ |
| Language Model | Token sequence | Next token probabilities | $f: \mathbb{Z}^n \rightarrow [0,1]^{|V|}$ |
| Recommendation | User + Item features | Rating prediction | $f: \mathbb{R}^{u} \times \mathbb{R}^{i} \rightarrow \mathbb{R}$ |

## The Function Perspective

Understanding ML through the lens of functions gives you:

1. **Clarity**: Instead of black boxes, you see mappings with precise domains and ranges
2. **Composability**: Complex models are compositions of simpler functions
3. **Debugging intuition**: When something fails, you can trace through function transformations
4. **Architecture design**: Choosing layers means choosing function types

## What You Will Learn

### Chapter 1: Functions Fundamentals
- Domain and range - understanding input/output spaces
- One-to-one vs many-to-one - why this matters for inverse problems
- Function composition - how deep learning stacks transformations
- Inverse functions - the key to understanding encoders/decoders

### Chapter 2: Common Function Types
- Linear functions - the backbone of ML
- Polynomial functions - adding expressiveness
- Exponential and logarithmic functions - handling scale
- Sigmoid and softmax - probabilities from raw scores
- Step functions - decisions and thresholds

### Chapter 3: Multivariable Functions
- Functions of many inputs - real-world feature spaces
- Parameterized functions - what "learning" actually means
- The $y = f(x, \theta)$ notation - separating data from parameters

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING MODEL                       │
│                                                                 │
│   Input x ──▶ [Function f(x, θ)] ──▶ Output ŷ                  │
│                      │                                          │
│                      │ Parameters θ                             │
│                      │ (learned from data)                      │
│                                                                 │
│   Training: Find θ that makes f(x, θ) ≈ y for training data    │
│   Inference: Apply f(x, θ*) to new inputs                       │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before diving into this level, ensure you're comfortable with:
- Basic algebra (Level 1)
- Set notation and intervals (Level 2)
- Python basics (NumPy arrays, basic plotting)

## How to Use This Level

1. **Read actively**: Work through the examples with pen and paper
2. **Run the code**: Every code block is designed to be executable
3. **Visualize**: The diagrams and plots build intuition
4. **Do the exercises**: They reinforce the ML connections

## Key Insight

> **"Learning" in machine learning means finding the right function parameters. "Inference" means applying that function to new inputs. Everything else is details.**

When you finish this level, you'll see every ML model as what it truly is: a parameterized function that we optimize to map inputs to desired outputs. This perspective will serve you throughout your ML journey.

Let's begin!

---

**Next**: [Chapter 1 - Functions Fundamentals](01-functions.md)
