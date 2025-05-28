+++
title = "Calculus Overview: From Limits to Hessians — a Machine‑Learning‑Centric Journey through Calculus"
date  = "2025‑05‑28"
author = "Artintellica"
+++

### Why this series?

Modern machine‑learning engineers don’t study calculus for the pleasure of
proofs; we study it because every gradient, every optimizer, and every diffusion
model is built on it.  This 16‑post series walks from single‑variable limits all
the way to Newton‑like second‑order methods, sprinkling **PyTorch**, **NumPy**,
**matplotlib**, and even **torchdiffeq** demos along the way. Each post is two
things at once:

- a _mini lesson_ in classical calculus,
- a _code lab_ that shows the idea living inside ML tooling.

Below is the full outline, grouped into four thematic parts.

---

## Part I – Single‑Variable Foundations

| #   | Topic                                                                     | Code‑Lab Highlights                    | Why ML cares                             |
| --- | ------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------- |
| 1   | **[Limits & Continuity](/blog/0003-calculus-1-limits-and-continuity.md)** | zoom‐in plots, ε–δ checker             | numerical stability, vanishing grads     |
| 2   | **[Derivatives](/blog/0004-calculus-2-derivatives.md)**                   | finite diff vs autograd on `torch.sin` | gradients drive learning                 |
| 3   | **[Fundamental Theorem](/blog/0005-calculus-3-fundamental-theorem.md)**   | trapezoid & Simpson vs `autograd.grad` | loss ↔ derivatives ↔ integrals         |
| 4   | **[1‑D Optimization](/blog/0006-calculus-4-gradient-descent.md)**         | hand‑rolled gradient descent           | baby training loop                       |
| 5   | **[Taylor/Maclaurin](/blog/0007-calculus-5-taylor-series.md)**            | animated $e^{x}$ truncations           | activation approx., positional encodings |

---

## Part II – Multivariable Core

| #   | Topic                                                                                | Code‑Lab Highlights               | Why ML cares                   |
| --- | ------------------------------------------------------------------------------------ | --------------------------------- | ------------------------------ |
| 6   | **[Vectors & ∇](/blog/0008-calculus-6-vectors-and-gradient.md)**                     | quiver of ∇$x^2+y^2$              | visual back‑prop intuition     |
| 7   | **[Jacobian & Hessian](/blog/0009-calculus-7-jacobian-hessian.md)**                  | tiny‑MLP Hessian spectrum         | curvature, 2‑nd‑order opt.     |
| 8   | **[Multiple Integrals](/blog/0010-calculus-8-multiple-integrals.md)**                | Monte‑Carlo 2‑D Gaussian          | expected loss, ELBO            |
| 9   | **[Change of Variables](/blog/0011-calculus-9-change-of-variables.md)**              | affine flow, log‑det via autograd | flow‑based generative models   |
| 10  | **[Line & Surface Integrals](/blog/0012-calculus-10-line-and-surface-integrals.md)** | streamplots, path work            | RL trajectories, gradient flow |

---

## Part III – Vector Calculus & Differential Eqs.

| #   | Topic                                                                                  | Code‑Lab Highlights         | Why ML cares                          |
| --- | -------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------- |
| 11  | **[Divergence, Curl, Laplacian](/blog/0013-calculus-11-divergence-curl-laplacian.md)** | heat‑equation on grid       | diffusion models, graph Laplacian     |
| 12  | **[ODEs](/blog/0014-calculus-12-odes.md)**                                             | train Neural‑ODE on spirals | continuous‑time nets                  |
| 13  | **[PDEs](/blog/0015-calculus-13-pdes.md)**                                             | finite‑diff wave equation   | physics‑informed nets, vision kernels |

---

## Part IV – Variations & Autodiff

| #   | Topic                                                                          | Code‑Lab Highlights           | Why ML cares                      |
| --- | ------------------------------------------------------------------------------ | ----------------------------- | --------------------------------- |
| 14  | **[Functional Derivatives](/blog/0016-calculus-14-calculus-of-variations.md)** | gradient of $\int (f')^2\!dx$ | weight decay as variational prob. |
| 15  | **[Back‑prop from Scratch](/blog/0017-calculus-15-back-prop-from-scratch.md)** | 50‑line reverse‑mode engine   | demystify autograd                |
| 16  | **[Hessian‑Vector / Newton](/blog/0018-calculus-16-second-order-methods.md)**  | SGD vs L‑BFGS, BFGS sketch    | faster second‑order ideas         |

---

### How to use this roadmap

- **Jump in anywhere.** Each post is self‑contained; hyperlinks point back when
  a prerequisite helps.
- **Code first, theory second.** Clone the repo and _run the notebooks_ — the
  math sticks once you see numbers move.
- **Connect the dots.** By Part IV the functional gradient of a smoothness prior
  becomes an explicit Laplacian, which you already met in Part III, and the
  Hessian‑vector trick you use in Newton steps is the same autograd magic you
  built in Part II.

### Where we go next

This wraps the calculus arc, but not the journey. Future posts will spin these
tools into:

- **Diffusion & score‑based generative models** (heat equation +
  change‑of‑variables)
- **Meta‑learning & higher‑order gradients** (double backward on your scratch
  autodiff)
- **Large‑scale second‑order optimizers** (Hessian‑vector tricks at GPT scale)

If you’ve followed along, you now wield derivatives, integrals, flows, and
variational principles — not as formalities, but as coding techniques that ship
in real ML systems. Onward!
