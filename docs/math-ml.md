# Mathematics for Machine Learning – Detailed Answers

> Note: The original paper omits explicit numeric entries for matrices/vectors. Below, I lay out the full solution methods with symbolic variables; you can drop in the actual numbers and evaluate directly.

## Q1. Solving linear systems and echelon form

**(a) Solve AX = b via echelon form**
1. Form the augmented matrix [A\,|\,b].
2. Apply Gaussian elimination: pivot on leading nonzero in each row, eliminate below; scale pivots to 1 if desired.
3. Back-substitute to get X. If any row becomes [0 … 0 | c] with c ≠ 0, the system is inconsistent; otherwise solution exists (unique if A is full-rank; infinitely many if rank-deficient but consistent).

**(b) Echelon form, det(B), rank(B)**
1. Row-reduce B to row-echelon (upper-triangular-like). The nonzero row count = rank(B).
2. Determinant: for a square matrix, det(B) = (product of pivots) × (−1)^{#row-swaps} ÷ (product of any row-scale factors you applied). If B is upper-triangular after elimination with only row-additions and swaps (no scaling), det(B) is just the product of diagonal entries up to sign from swaps.

## Q2. Linear dependence and subspaces in \(\mathbb{R}^3\)

**(a) Linear combination**
- To show \(v_3\) is a combination of \(v_1, v_2\), solve \(\alpha v_1 + \beta v_2 = v_3\). If a solution exists, the three vectors are linearly dependent (one is redundant).

**(b) Subspace V**
- To prove V is a subspace: show (i) contains 0; (ii) closed under addition; (iii) closed under scalar multiplication. For sets defined by linear equations (e.g., \(ax+by+cz=0\)), these hold automatically.
- Basis: row-reduce the matrix with vectors of V as rows/columns, keep pivot columns as basis vectors. Dimension = number of basis vectors (rank).

**(c) Membership test**
- Substitute the candidate vector into the defining linear conditions of V (or express as a combination of basis vectors). If it satisfies, it lies in V; otherwise not.

## Q3. SVD and diagonalizability

**(a) SVD of A**
1. Compute eigenvalues of \(A^T A\). Their square roots are singular values \(\sigma_i\), ordered \(\sigma_1 \ge \sigma_2 \ge \dots\).
2. Right singular vectors: normalized eigenvectors of \(A^T A\) (columns of V).
3. Left singular vectors: \(u_i = (1/\sigma_i) A v_i\) (columns of U). Assemble \(A = U \Sigma V^T\).

**(b) Diagonalizability**
- A is diagonalizable over \(\mathbb{R}\) if it has a full set of linearly independent eigenvectors (algebraic multiplicity = geometric multiplicity for each eigenvalue). For symmetric matrices, diagonalizable via orthogonal basis always holds. If A is defective (missing eigenvectors), it is not diagonalizable but still has an SVD.

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