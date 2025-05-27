Below is a **“big‑picture syllabus”** you can turn into a sequence of
Artintellica blog posts. Each line tells you

- **Concept ➜ what to show in Python ➜ why ML/AI folks care.**
- Suggested libraries appear in `[brackets]`.
- Start anywhere—the order is logical but modular.

---

## Part I – Single‑Variable Foundations

| #   | Topic                               | Code‑Lab Ideas                                                                         | ML / AI Relevance                                                      |
| --- | ----------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1   | **Limits & Continuity**             | plot `sin(x)/x`, zoom on `x→0`; implement ε‑δ “tolerance checker” \[NumPy, Matplotlib] | Numerical stability, vanishing gradients                               |
| 2   | **Derivatives**                     | finite‑difference vs. automatic diff on `torch.sin`; visualize slope field \[PyTorch]  | Gradients drive learning                                               |
| 3   | **Fundamental Theorem of Calculus** | trapezoid & Simpson vs. `torch.autograd.grad`; area under normal PDF \[SciPy]          | Loss ↔ derivatives ↔ integrals of PDFs                                 |
| 4   | **Optimization in 1 D**             | write gradient‑descent for `f(x)=x⁴−3x²+2`; compare step sizes \[PyTorch]              | Toy intro to training loops                                            |
| 5   | **Taylor & Maclaurin Series**       | animate polynomial truncations of `eˣ`; show approximation error \[NumPy, Matplotlib]  | Activation‑function approximations, transformers′ positional encodings |

---

## Part II – Multivariable Core

| #   | Topic                                   | Code‑Lab Ideas                                                                | ML / AI Relevance                             |
| --- | --------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------- |
| 6   | **Vectors & ∇ Gradient**                | quiver plot of ∇ `f(x,y)=x²+y²`; compute with `torch.autograd.grad`           | Visual intuition for back‑prop                |
| 7   | **Jacobian & Hessian**                  | batch‑Jacobian of a tiny MLP; eigen‑plot of Hessian spectrum \[PyTorch]       | Second‑order optimizers, curvature            |
| 8   | **Multiple Integrals**                  | Monte‑Carlo integrate a 2‑D Gaussian; compare to closed form \[NumPy]         | Expected loss, evidence lower bound           |
| 9   | **Change of Variables / Jacobian Det.** | build a 2‑D affine normalizing flow, log‑det via autograd \[PyTorch]          | Flow‑based generative models                  |
| 10  | **Line & Surface Integrals**            | integrate vector field along paths; visualize with stream‑plots \[Matplotlib] | Work done by gradient flow, RL path integrals |

---

## Part III – Vector Calculus & Differential Equations

| #   | Topic                                      | Code‑Lab Ideas                                                 | ML / AI Relevance                     |
| --- | ------------------------------------------ | -------------------------------------------------------------- | ------------------------------------- |
| 11  | **Divergence, Curl, Laplacian**            | heat‑equation demo on a grid with convolution kernels \[NumPy] | Diffusion models, graph Laplacians    |
| 12  | **Ordinary Differential Equations (ODEs)** | use `torchdiffeq` to train a Neural‑ODE on spirals             | Continuous‑time nets, latent SDEs     |
| 13  | **Partial Differential Equations (PDEs)**  | finite‑difference wave equation; animate \[NumPy, Matplotlib]  | Physics‑informed nets, vision kernels |

---

## Part IV – Calculus of Variations & Automatic Differentiation

| #   | Topic                                           | Code‑Lab Ideas                                              | ML / AI Relevance                      |
| --- | ----------------------------------------------- | ----------------------------------------------------------- | -------------------------------------- |
| 14  | **Functional Derivatives**                      | derive & code gradient of ∫ (f′)² dx regularizer            | Weight‑decay as variational problem    |
| 15  | **Back‑prop from Scratch**                      | implement reverse‑mode AD in ≈50 lines; verify with PyTorch | Demystifies autograd                   |
| 16  | **Hessian‑vector Products & Newton‑like Steps** | compare SGD vs. L‑BFGS on convex function                   | Faster convergence, second‑order ideas |

---

## Part V – Modern/Advanced Topics

| #   | Topic                                       | Code‑Lab Ideas                                                  | ML / AI Relevance                       |
| --- | ------------------------------------------- | --------------------------------------------------------------- | --------------------------------------- |
| 17  | **Manifold Calculus**                       | optimize on sphere with Riemannian gradient descent \[GeoTorch] | Word‑embedding norm constraints         |
| 18  | **Measure Theory Lite**                     | discrete vs. continuous entropy; integrate via Monte Carlo      | KL divergence, VAEs                     |
| 19  | **Stochastic Calculus**                     | simulate Brownian motion; show Itô vs. Stratonovich \[NumPy]    | Stochastic gradient Langevin dynamics   |
| 20  | **Information Geometry / Natural Gradient** | code Fisher‑information metric for logistic reg.                | Stable training, e.g. K‑fac             |
| 21  | **Functional Analysis & RKHS**              | implement kernel regression; visualize feature space            | Gaussian processes, attention as kernel |

---

### How to turn each row into a post

1. **Concept intro (≈ 300 words).**
2. **Derivation / math sketch (LaTeX).**
3. **Python demo notebook** (MacBook‑friendly; GPU optional).
4. **“Try‑it‑yourself” tasks**—small code tweaks.
5. **ML connection sidebar**—why this matters in practice.
6. **GitHub repo folder** named `calc-XX-topic/` with `requirements.txt` and a
   short README.

### Tooling conventions

- **Environment**: Python ≥ 3.10, `conda` or `venv`; macOS Metal‐backed PyTorch
  where possible.
- **Packages**: `numpy`, `scipy`, `matplotlib`, `torch`, `torchvision`,
  `torchdiffeq`, `transformers` (for later Bayesian/probability posts), `jax`
  optional for comparison.
- **Notebook style**: Jupyter or Quarto; keep cells ≤ 40 lines; include `%time`
  benchmarks.
- **Repo layout**:

  ```
  posts/
    01-limits/
      post.md
      demo.ipynb
      utils.py
  ```

- **CI**: GitHub Actions matrix (CPU / M‑series GPU) to ensure notebooks run
  headless.

---

#### Next step

Pick Topic 1 or 2, draft a short post, and we’ll flesh out math, code, and
narrative together. When you’re ready, just tell me which topic to start coding,
and I’ll help you produce the notebook and blog text.
