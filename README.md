# PINN
Training a Physics-informed NN on Maxwell-Lorentz Equations

This repository contains an implementation of a **Physics-Informed Neural Network** designed to solve **1D Maxwell equations** through Faraday's and Ampère-Maxwell's laws.

Unlike traditional neural networks that rely solely on data, this model integrates physical laws directly into its loss function, ensuring that the predicted electric ($E$) and magnetic ($B$) fields satisfy electromagnetic theory.

---

## Physical Context

The model solves the coupled first-order partial differential equations for a transverse electromagnetic wave propagating in a vacuum, normalized with the speed of light $c = 1$:

1. **Faraday's Law:** $$\frac{\partial E}{\partial x} + \frac{\partial B}{\partial t} = 0$$

2. **Ampère-Maxwell Law:** $$\frac{\partial B}{\partial x} + \frac{\partial E}{\partial t} = 0$$

### Exact Solution
To validate the model, we simulate a traveling wave where the exact fields are given by:
* $E(x, t) = \sin(x - t)$
* $B(x, t) = \sin(x - t)$

## Features

* **Multi-Output Architecture:** A custom Keras model simultaneously predicts both $E$ and $B$ fields.
* **Custom Physics Loss:** Uses `tf.GradientTape(persistent=True)` to compute first-order analytical partial derivatives ($E_x, E_t, B_x, B_t$) and penalize deviations from Maxwell's laws.
* **Hybrid Training:** Combines a data-driven loss with a PDE-based physics residual loss.

---

## Performance & Results

After training for 1000 epochs, the network demonstrates excellent convergence, accurately capturing wave dynamics with high precision:

* **Final Physics Loss:** `~0.0008` (The NN successfully learned the PDE)
* **Final Data Loss:** `~0.0071`
* **Test MSE on E field:** `~0.0003`
