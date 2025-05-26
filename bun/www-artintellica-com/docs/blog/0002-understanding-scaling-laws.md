+++
title = "Understanding \"Scaling Laws for Neural Language Models\" by Kaplan et al."
date = "2025-05-26 00:00:00"
author = "Artintellica"
number = 2
+++

## Part 1 – What the Paper Says ✍️

### 1. Why this paper matters

Training large language models used to feel like alchemy: _“Add more layers,
throw in more data, cross your fingers.”_  
In **“Scaling Laws for Neural Language Models”** (Kaplan, McCandlish, et al.,
OpenAI 2020) the authors show that progress is _predictable_. They fit simple
power‑law curves that relate model size, dataset size, and compute to the
cross‑entropy loss a model achieves on held‑out data.

> **Read the original PDF:** <https://arxiv.org/abs/2001.08361>

For beginners, the big idea is this:

| If you **double**…        | …your validation loss falls by ≈ a **fixed percentage** |
| ------------------------- | ------------------------------------------------------- |
| Model parameters *N*      | \(L \propto N^{-0.076}\)                                |
| Training tokens *D*       | \(L \propto D^{-0.095}\)                                |
| Total compute *C* (FLOPs) | \(L \propto C^{-0.057}\)                                |

That tiny exponent means you need _orders of magnitude_ more resources for each
constant jump in quality—but the payoff is steady and measurable.

### 2. How they discovered the laws

1. **Pick a simple architecture.**  
   All experiments use the same decoder‑only Transformer (no fancy tricks).
2. **Sweep one variable, hold the other two large.**  
   * Vary *N*: freeze *D* ≈ 300 B tokens, train to convergence.  
   * Vary *D*: fix a 1.5 B‑param model, stop once loss stops improving.  
   * Vary *C\*: early‑stop training runs at different compute budgets.
3. **Fit a power‑law curve** in log–log space.  
   A straight line appears over six to seven decades.

Because every curve is smooth, you can juggle the three dials (_N_, *D*, *C*) to
stay on the same “iso‑loss” contour.

### 3. The compute‑optimal recipe

Suppose you have a hard budget of **C FLOPs**. Kaplan et al. derive:

- **Optimal model size:** \(N \propto C^{0.73}\)
- **Optimal data seen:** \(D \propto C^{0.27}\)

Translated: spend most of your budget on a **larger network**, train it on a
**moderate amount of data**, and **stop early** once loss plateaus. (The later
“Chinchilla” paper revises the constants but not the logic.)

### 4. Practical take‑aways for newcomers

- **Rule of thumb:** If training loss is still falling sharply, you’re data‑ or
  compute‑limited. If it’s flat and you still have budget, scale the model.
- **Transfer works because scale works.** Bigger language models learn general
  representations that fine‑tune well on downstream tasks.
- **Budget planning:** Before renting GPUs, sketch where your planned run sits
  on a scaling curve; you can forecast returns in advance.

> **Further reading**
>
> - Hoffmann et al. “Training Compute‑Optimal Language Models” (“Chinchilla”).
> - Henighan et al. “Scaling Laws for Autoregressive Generative Modeling.”
> - Hestness et al. “Deep Learning Scaling Is Predictable, Empirically.”

### 5. What’s next in this post

In **Part 2** we’ll reproduce the _shape_ of these curves on a single MacBook
Pro:

1. **Error vs. data size** with a fixed‑size polynomial regressor.
2. **Error vs. model size** with a fixed‑size dataset.

You’ll see two log–log plots whose straight‑line slopes echo the OpenAI
results—no GPU cluster required. _(Code coming up in the next section.)_

<!--  =================================================================  -->
<!--                       Part 2: Python demos here                     -->
<!--  We'll fill this in next time.                                      -->
<!--  =================================================================  -->
