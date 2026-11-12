'''version3: hard-margin SVM solved in the dual with cvxopt.

What is NEW compared to version1 (same data, same classifier)
- Instead of w and b we solve for one Lagrange multiplier alpha_i per point:
      max_alpha  sum_i alpha_i - 1/2 sum_{i,j} alpha_i alpha_j y_i y_j x_i^T x_j
      subject to alpha_i >= 0,  sum_i alpha_i y_i = 0
  cvxopt minimises, so we negate:  P_ij = y_i y_j x_i^T x_j,  q = -1,
  G = -I, h = 0 (alpha >= 0), A = y^T, b = 0 (equality constraint).
- Complementary slackness: alpha_i > 0 only for points ON the margin. These
  are the support vectors (circled in the plot); every other point could be
  deleted without changing the classifier.
- w = sum_i alpha_i y_i x_i, and for any support vector s,
  y_s (w^T x_s + b) = 1, so b = y_s - w^T x_s. We average over all support
  vectors for numerical stability.
- The data enters only through dot products x_i^T x_j. That is what
  version4 exploits with kernels.
'''
import numpy as np
from cvxopt import matrix, solvers

from utils import (get_classification_dataset, train_test_split,
                   plot_decision_boundary, set_solver_options, accuracy)


class SVM:
    def __init__(self, sv_tol=1e-4):
        self.alphas = None
        self.X = None
        self.y = None
        self.b = None
        self.sv_tol = sv_tol  # alphas below this are treated as exactly 0

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0
        n = X.shape[0]
        # objective: P_ij = y_i y_j x_i^T x_j  (the Gram matrix weighted by labels)
        P = (y @ y.T) * (X @ X.T)
        q = -np.ones((n, 1), dtype=np.double)
        # inequality constraints: -alpha_i <= 0
        G = -np.eye(n, dtype=np.double)
        h = np.zeros((n, 1), dtype=np.double)
        # equality constraint: sum_i alpha_i y_i = 0
        A = y.T
        b = np.zeros((1, 1))

        set_solver_options()
        solver = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h),
                            matrix(A), matrix(b))
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return

        # keep only the support vectors (alpha_i > 0)
        alphas = np.array(solver['x']).ravel()
        idxs = np.where(alphas > self.sv_tol)[0]
        self.alphas = alphas[idxs]
        self.X = X[idxs]
        self.y = y[idxs].ravel()

        # b = y_s - sum_j alpha_j y_j x_j^T x_s, averaged over support vectors s
        self.b = np.mean(self.y - (self.alphas * self.y) @ (self.X @ self.X.T))
        w = (self.alphas * self.y) @ self.X
        print(f'{len(idxs)} support vectors | w = {w} b = {self.b:.4f}')

    def predict(self, X, raw=False):
        assert self.alphas is not None, 'Model not trained yet'
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        # f(x) = sum_i alpha_i y_i x_i^T x + b, summed over support vectors only
        out = (X @ self.X.T) @ (self.alphas * self.y) + self.b
        return out if raw else np.sign(out)


if __name__ == '__main__':
    np.random.seed(2023)
    X, y = get_classification_dataset(var=0.3)
    y[y == 0] = -1
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM()
    clf.fit(X_train, y_train)
    print(f'Train accuracy: {accuracy(clf, X_train, y_train):.3f}')
    print(f'Test accuracy: {accuracy(clf, X_test, y_test):.3f}')
    plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True,
                           support_vectors=clf.X)
    plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
