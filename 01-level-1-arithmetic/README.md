# Level 1: Arithmetic & Numbers

## Overview

Welcome to the foundation of all mathematics used in machine learning. Before diving into linear algebra, calculus, or probability, you must have a solid understanding of how numbers work at a fundamental level. This section covers the building blocks that every ML algorithm relies upon.

## Why This Matters for Machine Learning

Every machine learning model, from simple linear regression to complex neural networks, ultimately performs arithmetic operations on numbers. Understanding number systems helps you:

- **Debug numerical issues**: Why does your loss suddenly become `NaN`? Why do gradients explode or vanish?
- **Choose appropriate data types**: When to use `float32` vs `float64`, and why it matters for training speed and accuracy
- **Understand model behavior**: How rounding errors accumulate and affect model predictions
- **Optimize performance**: Knowing the computational cost of different operations

## What You'll Learn

### Chapter 1: Number Systems
Explore the hierarchy of numbers from natural numbers to floating-point representations. Understand how computers store and manipulate numbers, and why this matters for numerical stability in ML.

### Chapter 2: Arithmetic Operations
Master the fundamental operations that form the basis of all computations. Learn about addition, subtraction, multiplication, division, modulo, and absolute value in the context of array operations and ML algorithms.

### Chapter 3: Ratios and Scales
Understand proportional relationships, percentages, and normalization techniques. These concepts are essential for feature scaling, learning rates, and interpreting model outputs.

## Prerequisites

- Basic programming knowledge (preferably Python)
- Familiarity with NumPy is helpful but not required

## Learning Objectives

By the end of this level, you will be able to:

1. Explain the differences between number systems and their use cases
2. Identify potential numerical issues in ML code
3. Apply appropriate data types for different scenarios
4. Understand floating-point precision and its implications
5. Normalize and scale data correctly
6. Calculate and interpret ratios, percentages, and growth rates

## Chapter Navigation

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| [01](./01-number-systems.md) | Number Systems | Natural, integers, rationals, reals, IEEE 754 |
| [02](./02-arithmetic-operations.md) | Arithmetic Operations | Add, subtract, multiply, divide, modulo, abs |
| [03](./03-ratios-and-scales.md) | Ratios and Scales | Ratios, proportions, percentages, normalization |

## Quick Reference

### Number Type Selection Guide

```
Need whole numbers only? --> Use integers (int32, int64)
Need decimals? --> Use floating-point (float32, float64)
Need maximum precision? --> Use float64 or decimal
Training neural networks? --> float32 is usually sufficient
Doing scientific computing? --> float64 is often preferred
```

### Common Numerical Issues in ML

| Issue | Symptom | Solution |
|-------|---------|----------|
| Overflow | Values become `inf` | Use log-space, scale inputs |
| Underflow | Values become 0 | Use log-space, numerical stability tricks |
| NaN | Invalid operations | Check for division by zero, log of negative |
| Precision loss | Accumulated errors | Use higher precision, Kahan summation |

## Estimated Time

- Reading: 2-3 hours
- Exercises: 1-2 hours
- Total: 3-5 hours

---

*Proceed to [Chapter 1: Number Systems](./01-number-systems.md) to begin your journey.*
