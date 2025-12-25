# Mathematics for Machine Learning (Dec 2025) — Solved Answers

Some matrices/vectors are not visible in the extracted text available in this workspace, so full numeric row-reduction/SVD answers cannot be reproduced exactly here without the missing values. This page fully solves Q4 (which is fully present), and gives the correct procedures for the matrix-dependent questions.

---

## Q1. Echelon form, determinant, rank (procedure)

### (a) Solve $AX=b$ via echelon form
1. Form augmented matrix $[A\mid b]$.
2. Apply row operations to reach row-echelon form.
3. Back-substitute to get the solution (or identify no/infinitely-many solutions).

### (b) Echelon form of $B$, determinant, rank
- Row-reduce $B$ to echelon form.
- **Rank** = number of non-zero rows (pivots) in echelon form.
- **Determinant** can be tracked during elimination:
  - swapping two rows flips sign,
  - scaling a row by $c$ scales det by $c$,
  - adding a multiple of one row to another doesn’t change det.
  - If reduced to upper-triangular $U$, then $\det(U)$ is product of diagonal entries (after accounting for swaps/scales).

---

## Q2. Linear dependence + subspace

### (a) Linear combination and dependence
To show vectors $v_1,v_2,v_3\in\mathbb{R}^3$ are dependent, express one as a linear combination of the others:
$$v_3 = a v_1 + b v_2$$
Then:
$$a v_1 + b v_2 - v_3 = 0$$
with not-all-zero coefficients → linear dependence.

### (b) Prove $V$ is a subspace, basis, dimension
Subspace test:
1. $0\in V$
2. Closed under addition: $u,v\in V \Rightarrow u+v\in V$
3. Closed under scalar multiplication: $u\in V,\alpha\in\mathbb{R} \Rightarrow \alpha u\in V$

Then find a spanning set and reduce it to a basis (remove dependent vectors). Dimension = number of basis vectors.

### (c) Membership test
Given $x$, check if it satisfies the defining constraints of $V$ (equations/parametric form), or solve $x$ as combination of basis vectors.

---

## Q3. SVD + diagonalizability (procedure)

### (a) SVD of a matrix $A$
1. Compute $A^T A$.
2. Eigen-decompose $A^T A = V\Sigma^2 V^T$.
3. Singular values are $\sigma_i = \sqrt{\lambda_i}$.
4. Compute $U$ columns as $u_i = \frac{1}{\sigma_i} A v_i$.
5. Assemble $A = U\Sigma V^T$.

### (b) Diagonalizability test
A real square matrix is diagonalizable if it has a full set of linearly independent eigenvectors (geometric multiplicity sums to $n$). Equivalently, minimal polynomial has no repeated factors.

---

## Q4. Optimization + Taylor series (fully solved)

Given:
$$f(x,y)=x^2 + xy + 2y^2 - 4x + 3$$

### (a) Gradient
$$\nabla f(x,y)=\left[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\right]$$

- $\frac{\partial f}{\partial x}=2x + y - 4$
- $\frac{\partial f}{\partial y}=x + 4y$

So:
$$\nabla f(x,y) = (2x+y-4,\; x+4y)$$

### (b) Critical point(s)
Solve:
$$2x+y-4=0$$
$$x+4y=0$$

From $x=-4y$. Substitute into first:
$$2(-4y)+y-4=0 \Rightarrow -8y+y=4 \Rightarrow -7y=4 \Rightarrow y=-\frac{4}{7}$$
Then:
$$x=-4y = -4\left(-\frac{4}{7}\right)=\frac{16}{7}$$

Critical point: $\left(\frac{16}{7},-\frac{4}{7}\right)$.

### (c) Classify critical point
Hessian:
$$H=\begin{bmatrix} f_{xx} & f_{xy}\\ f_{yx} & f_{yy}\end{bmatrix} = \begin{bmatrix}2 & 1\\ 1 & 4\end{bmatrix}$$

Check leading principal minors:
- $2>0$
- $\det(H)=2\cdot 4 - 1\cdot 1 = 7>0$

So $H$ is positive definite → critical point is a **strict local minimum**.

### (d) Taylor series of $g(x)=f(x,1)$ at $x=1$
Compute $g(x)$:
$$g(x)=x^2 + x\cdot 1 + 2\cdot 1^2 - 4x + 3 = x^2 - 3x + 5$$

Taylor about $x=1$ (since polynomial, it’s exact):
Let $h=x-1$ so $x=1+h$.
$$g(1+h)=(1+h)^2 - 3(1+h) + 5 = (1+2h+h^2) - 3 - 3h + 5$$
$$= 3 - h + h^2$$

So:
$$g(x)=3 - (x-1) + (x-1)^2$$

---

## Q5. Inner product via $\langle x,y\rangle = x^T A y$ (procedure)
Matrix $A$ defines an inner product on $\mathbb{R}^2$ if:
- $A$ is **symmetric** ($A=A^T$)
- $A$ is **positive definite** ($x^T A x > 0$ for all $x\ne 0$)

Distance under this inner product:
$$\|x\|_A = \sqrt{x^T A x},\quad d_A(x,y)=\|x-y\|_A$$

Orthogonality test:
$$x \perp_A y \iff x^T A y = 0$$
