# Mathematics for Machine Learning – Detailed Answers

> Numeric matrices/vectors were not provided in the paper. Below are complete solution workflows and symbolic formulas; plug your given numbers directly to get the final numeric results.

## Q1. Solving linear systems and echelon form

**(a) Solve AX = b via echelon form**
1) Build [A | b]. 2) Pivot-eliminate downward to row echelon (all zeros below pivots). 3) If a row is [0 … 0 | c], c ≠ 0 → no solution. Otherwise back-substitute to get X. Unique solution if rank(A)=n; infinite solutions if rank(A)<n but consistent.

**(b) Echelon form, det(B), rank(B)**
1) Row-reduce B to echelon. Rank(B) = number of nonzero rows (pivots). 2) det(B) = (product of pivots)·(−1)^{#swaps}/(product of scaling factors). If you only use swaps/additions, det(B) equals the product of diagonal entries up to sign from swaps.

## Q2. Linear dependence and subspaces in \(\mathbb{R}^3\)

**(a) Linear combination & dependence**
- Solve \(\alpha v_1 + \beta v_2 = v_3\) (2 unknowns, 3 equations). If a solution exists, the three vectors are dependent; otherwise independent. Quick check: det([v1 v2 v3])=0 ⇒ dependent.

**(b) Subspace V**
- Subspace test: contains 0; closed under addition and scalar multiplication. Any solution set of homogeneous linear equations satisfies this. Basis: row-reduce stacked vectors, take pivot columns. Dimension = #pivots.

**(c) Membership**
- Plug the candidate into the defining linear equations of V or solve for coefficients in the basis; if it fits, it’s in V.

## Q3. SVD and diagonalizability

**(a) SVD of A (steps)**
1) Form \(A^T A\), find eigenvalues \(\lambda_i\). 2) Singular values: \(\sigma_i = \sqrt{\lambda_i}\) sorted. 3) Right singular vectors: eigenvectors of \(A^T A\) → V. 4) Left singulars: \(u_i = A v_i / \sigma_i\) → U. Assemble \(A = U \Sigma V^T\).

**(b) Diagonalizability**
- Diagonalizable if it has n independent eigenvectors (geom mult = alg mult). All real symmetric matrices are diagonalizable (orthogonally). Defective matrices are not, but still admit an SVD.

## Q4. Quadratic form analysis

Function: \(f(x,y) = x^2 + xy + 2y^2 - 4x + 3\).

**(a) Gradient**
- \(\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right) = (2x + y - 4,\; x + 4y).\)

**(b) Critical point**
- Solve \(2x + y - 4 = 0\) and \(x + 4y = 0\) ⇒ \(x = 16/7\), \(y = -4/7\).

**(c) Nature of critical point**
- Hessian \(H = \begin{pmatrix} 2 & 1 \\ 1 & 4 \end{pmatrix}\). Eigenvalues are positive (trace 6, determinant 7 > 0) ⇒ positive definite ⇒ critical point is a **strict local minimum**.

**(d) Taylor of \(g(x)=f(x,1)\) at \(x=1\)**
- \(g(x) = x^2 + x + 2 - 4x + 3 = x^2 - 3x + 5\).
- Derivatives: \(g'(x)=2x-3\), \(g''(x)=2\).
- At \(x=1\): \(g(1)=3\), \(g'(1)=-1\), \(g''(1)=2\).
- Second-order Taylor about 1: \(g(x) \approx 3 + (-1)(x-1) + \tfrac{1}{2}(2)(x-1)^2 = 3 - (x-1) + (x-1)^2\). Higher-order terms are zero because \(g\) is quadratic.

## Q5. Inner product via matrix A

Let \(\langle X, Y \rangle = X^T A Y\) on \(\mathbb{R}^2\).

**(a) Conditions on A**
- A must be **symmetric** (\(A = A^T\)) and **positive definite** (\(X^T A X > 0\) for all nonzero X). These guarantee symmetry, bilinearity, and positive definiteness of the induced form.

**(b) Distance between two vectors p, q**
- \(d(p,q) = \sqrt{(p-q)^T A (p-q)}\). Plug the given p, q and compute; positive definiteness ensures a real, positive distance.

**(c) Orthogonality test**
- Vectors u, v are A-orthogonal if \(u^T A v = 0\). Compute with given u, v; zero means orthogonal under this inner product.

## Quick practice (math for ML)
- Compute SVD for a concrete 2×2 (e.g., [[3,0],[4,5]]); verify diagonalizability by eigenvectors.
- Take three specific vectors in \(\mathbb{R}^3\), row-reduce to find dependence and a basis for their span.
- Pick a 2×2 positive definite matrix A, verify PD via eigenvalues, and compute A-induced distances for two points.