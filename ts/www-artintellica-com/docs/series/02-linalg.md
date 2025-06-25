+++
title = "Linear Algebra for Machine Learning"
icon = "linalg"
author = "Artintellica"
date = "2025-06-05"
+++

### Part I – Foundations

The first part laid the groundwork by introducing the basic building blocks of
linear algebra and their direct relevance to machine learning data structures
and operations.

- **[Part 1: Vectors, Scalars, and Spaces](/blog/0020-linalg-01-vectors-scalers-spaces.md)**  
  _ML/AI
  Relevance_: Features, weights, data representation  
  _Focus_: NumPy arrays, PyTorch tensors, 2D/3D plots  
  We started with the fundamentals, exploring how vectors and scalars represent
  data points and parameters in ML, and how vector spaces provide the framework
  for operations.

- **[Part 2: Matrices as Data & Transformations](/blog/0021-linalg-02-matrices-and-transformations.md)**  
  _ML/AI
  Relevance_: Images, datasets, linear layers  
  _Focus_: Image as matrix, reshaping  
  Matrices were introduced as representations of data (like images) and as
  transformations (like neural network layers), showing their dual role in ML.

- **[Part 3: Matrix Arithmetic: Add, Scale, Multiply](/blog/0022-linalg-03-matrix-arithmetic.md)**  
  _ML/AI
  Relevance_: Linear combinations, weighted sums  
  _Focus_: Broadcasting, matmul, matrix properties  
  We covered essential operations like addition, scaling, and multiplication,
  critical for combining features and computing outputs in models.

- **[Part 4: Dot Product and Cosine Similarity](/blog/0023-linalg-04-dot-product.md)**  
  _ML/AI
  Relevance_: Similarity, projections, word vectors  
  _Focus_: `np.dot`, `torch.cosine_similarity`  
  The dot product and cosine similarity were explored as measures of similarity,
  vital for tasks like recommendation systems and NLP embeddings.

- **[Part 5: Linear Independence & Span](/blog/0024-linalg-05-linear-independence.md)**  
  _ML/AI
  Relevance_: Feature redundancy, expressiveness  
  _Focus_: Gram matrix, visualization  
  We discussed how linear independence and span help identify redundant features
  and understand the expressive power of data representations.

- **[Part 6: Norms and Distances](/blog/0025-linalg-06-norms.md)**  
  _ML/AI Relevance_: Losses, regularization, gradient scaling  
  _Focus_: L1, L2 norms, distance measures  
  Norms and distances were introduced as tools for measuring magnitudes and
  differences, underpinning loss functions and regularization techniques.

### Part II – Core Theorems and Algorithms

The second part dove deeper into the theoretical underpinnings and algorithmic
machinery of linear algebra, connecting them to pivotal ML techniques.

- **[Part 7: Orthogonality and Projections](/blog/0026-linalg-07-orthogonality.md)**  
  _ML/AI
  Relevance_: Error decomposition, PCA, embeddings  
  _Focus_: Gram-Schmidt, projections, orthonormal basis  
  Orthogonality and projections were shown to be key for decomposing data and
  reducing dimensions, setting the stage for PCA.

- **[Part 8: Matrix Inverses and Systems of Equations](/blog/0027-linalg-08-matrix-inverses.md)**  
  _ML/AI
  Relevance_: Solving for parameters, backpropagation  
  _Focus_: `np.linalg.solve`, invertibility  
  We explored how matrix inverses solve systems of equations, a concept central
  to finding optimal parameters in models.

- **[Part 9: Rank, Nullspace, and the Fundamental Theorem](/blog/0028-linalg-09-rank.md)**  
  _ML/AI
  Relevance_: Data compression, under/over-determined systems  
  _Focus_: `np.linalg.matrix_rank`, SVD intuition  
  Rank and nullspace illuminated the structure of data and solutions, linking to
  compression and system solvability.

- **[Part 10: Eigenvalues and Eigenvectors](/blog/0029-linalg-10-eigen.md)**  
  _ML/AI Relevance_: Covariance, PCA, stability, spectral clustering  
  _Focus_: `np.linalg.eig`, geometric intuition  
  Eigenvalues and eigenvectors were introduced as tools for understanding data
  variance and stability, crucial for PCA and clustering.

- **[Part 11: Singular Value Decomposition (SVD)](/blog/0030-linalg-11-svd.md)**  
  _ML/AI
  Relevance_: Dimensionality reduction, noise filtering, LSA  
  _Focus_: `np.linalg.svd`, visual demo  
  SVD was presented as a powerful decomposition method for reducing dimensions
  and filtering noise in data.

- **[Part 12: Positive Definite Matrices](/blog/0031-linalg-12-pos-def.md)**  
  _ML/AI Relevance_: Covariance, kernels, optimization  
  _Focus_: Checking PD, Cholesky, quadratic forms  
  We examined positive definite matrices, essential for ensuring well-behaved
  optimization and valid covariance structures.

### Part III – Applications in ML & Advanced Topics

The final part focused on direct applications and advanced concepts, showcasing
how linear algebra drives cutting-edge ML techniques and large-scale systems.

- **[Part 13: Principal Component Analysis (PCA)](/blog/0032-linalg-13-pca.md)**  
  _ML/AI
  Relevance_: Dimensionality reduction, visualization  
  _Focus_: Step-by-step PCA in code  
  PCA was implemented as a practical method for reducing data dimensions while
  retaining key information, with hands-on coding.

- **[Part 14: Least Squares and Linear Regression](/blog/0033-linalg-14-least-squares.md)**  
  _ML/AI
  Relevance_: Linear models, fitting lines/planes  
  _Focus_: Normal equations, SGD, scikit-learn  
  We connected linear algebra to regression, using least squares to fit models
  via the normal equations and stochastic gradient descent.

- **[Part 15: Gradient Descent in Linear Models](/blog/0034-linalg-15-gradient-descent.md)**  
  _ML/AI
  Relevance_: Optimization, parameter updates  
  _Focus_: Matrix calculus, vectorized code  
  Gradient descent was explored through matrix operations, showing how linear
  algebra enables efficient optimization.

- **[Part 16: Neural Networks as Matrix Functions](/blog/0035-linalg-16-neural-networks.md)**  
  _ML/AI
  Relevance_: Layers, forward/backward pass, vectorization  
  _Focus_: PyTorch modules, parameter shapes  
  Neural networks were framed as sequences of matrix operations, highlighting
  vectorization in forward and backward passes.

- **[Part 17: Tensors and Higher-Order Generalizations](/blog/0036-linalg-17-tensors.md)**  
  _ML/AI
  Relevance_: Deep learning, NLP, computer vision  
  _Focus_: `torch.Tensor`, broadcasting, shape tricks  
  Tensors extended matrix concepts to higher dimensions, critical for deep
  learning tasks in NLP and vision.

- **[Part 18: Spectral Methods in ML (Graph Laplacians, etc.)](/blog/0037-linalg-18-spectral.md)**  
  _ML/AI
  Relevance_: Clustering, graph ML, signal processing  
  _Focus_: Laplacian matrices, spectral clustering  
  Spectral methods using graph Laplacians were introduced for clustering and
  graph-based learning.

- **[Part 19: Kernel Methods and Feature Spaces](/blog/0038-linalg-19-kernel.md)**  
  _ML/AI
  Relevance_: SVM, kernel trick, non-linear features  
  _Focus_: Gram matrix, RBF kernels, Mercer's theorem  
  Kernel methods enabled non-linear learning via the kernel trick, transforming
  data implicitly into higher-dimensional spaces.

- **[Part 20: Random Projections and Fast Transforms](/blog/0039-linalg-20-random.md)**  
  _ML/AI
  Relevance_: Large-scale ML, efficient computation  
  _Focus_: Johnson-Lindenstrauss, random matrix code  
  Finally, random projections and fast transforms addressed scalability,
  reducing dimensionality and speeding up computations for massive datasets.
