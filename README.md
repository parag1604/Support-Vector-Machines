# Support Vector Machines

Four Support Vector Machine (SVM) implementations in Python, each solved as a quadratic program with the `cvxopt` library:

- **version1.py**: Simple Primal SVM (hard margin)
- **version2.py**: Primal SVM with Slack Variables (soft margin)
- **version3.py**: Simple Dual SVM (linear kernel)
- **version4.py**: Dual SVM with Kernel (linear, polynomial and RBF kernels)

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Support-Vector-Machines)

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy - required for vectorized operations
- Matplotlib - required for plotting
- cvxopt - the quadratic program solver

## Data

No dataset is needed. `utils.py` generates small synthetic 2-D datasets (two Gaussian blobs, and points inside/outside the unit circle), splits them into train and test sets, and plots decision boundaries.

All versions use labels $y_i \in \{-1, +1\}$ and solve problems of the form `cvxopt.solvers.qp(P, q, G, h, A, b)`:

$$
\min_z \frac{1}{2} z^\top P z + q^\top z \quad \text{subject to } Gz \le h,\ Az = b
$$

- **P**: the quadratic part of the objective. For the dual forms it is the (linear or kernel) Gram matrix weighted by the labels.
- **q**: the linear part of the objective, e.g. $-1$ for each $\alpha$ in the dual.
- **G, h**: inequality constraints, e.g. $-\alpha \le 0$.
- **A, b**: the equality constraint of the dual forms, $\sum_i \alpha_i y_i = 0$.

## Running

Each script generates a dataset, trains the SVM, prints the train/test accuracy, and plots the decision boundary ($f(x) = 0$) together with the margin lines ($f(x) = \pm 1$) for the train and test sets.

### 1. Simple Primal SVM

```bash
python version1.py
```

- **Formulation**:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to } y_i (w^\top x_i + b) \geq 1
$$

- **cvxopt variables** ($z = [w, b]$):
  - `P`: identity matrix for $w$ and $b$ (last diagonal entry set to 0, so $b$ is not penalised).
  - `q`: zero vector.
  - `G`: $-y_i [x_i, 1]$ for each sample (encodes $y_i(w^\top x_i + b) \geq 1$).
  - `h`: $-1$ vector.

The blobs are well separated, so every training point lies on or outside the margin (train accuracy 1.000, test accuracy 0.975).

### 2. Primal SVM with Slack Variables

```bash
python version2.py --C 0.5
```

- **Formulation**:

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_i \xi_i \quad \text{subject to } y_i (w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

- **cvxopt variables** ($z = [w, b, \xi]$):
  - `P`: block matrix, identity for $w$ and zeros for $b$ and the slack variables $\xi$.
  - `q`: $C$ for the slack variables, 0 for $w$ and $b$.
  - `G`: stacks the margin constraints $-y_i [x_i, 1] z - \xi_i \le -1$ and the slack constraints $-\xi_i \leq 0$.
  - `h`: $-1$ for the margin constraints, $0$ for the slack constraints.

New: the blobs now overlap, so some points must violate the margin. Try `--C 0.01` (wide margin, many violations) and `--C 100` (close to the hard margin).

### 3. Simple Dual SVM

```bash
python version3.py
```

- **Formulation**:

$$
\max_{\alpha} \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \quad \text{subject to } \alpha_i \geq 0, \ \sum_i \alpha_i y_i = 0
$$

- **cvxopt variables**:
  - `P`: $y_i y_j x_i^\top x_j$ Gram matrix.
  - `q`: $-1$ vector.
  - `G`: $-\alpha \leq 0$.
  - `h`: $0$ vector.
  - `A`: $y^\top$ (equality constraint).
  - `b`: $0$.

New: the same data and classifier as version1, but solved for one multiplier per point. Only the support vectors (circled in the plot, 3 of them here) have $\alpha_i > 0$; $w = \sum_i \alpha_i y_i x_i$ matches version1.

### 4. Dual SVM with Kernel

```bash
python version4.py --kernel rbf --sigma 1.0
python version4.py --kernel poly --degree 2
python version4.py --kernel linear
```

- **Formulation**: same as version3, but $x_i^\top x_j$ is replaced by $K(x_i, x_j)$ for a chosen kernel.
- **cvxopt variables**:
  - `P`: $y_i y_j K(x_i, x_j)$ kernel Gram matrix.
  - Other variables as in version3.

New: the data is a circle (class 1 outside the unit circle), which no line separates. With `--kernel linear` the hard-margin QP is infeasible and the solver fails; the degree-2 polynomial and RBF kernels separate the training set perfectly. Here `sigma` is the RBF bandwidth, $K(x,z) = \exp(-\|x-z\|^2 / 2\sigma^2)$ (scikit-learn's `gamma` equals $1/2\sigma^2$).

## HW

Try to implement the SVM dual with the kernel for soft margin (version5.py) using the same structure as version4.py, but incorporating slack variables in the dual formulation.
