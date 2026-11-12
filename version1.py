'''version1: hard-margin SVM, solved in the primal with cvxopt.

What this file teaches
- The maximum-margin classifier as a quadratic program (QP):
      min_{w, b}  1/2 ||w||^2   subject to   y_i (w^T x_i + b) >= 1  for all i
- How to write it in cvxopt's standard form
      min_z  1/2 z^T P z + q^T z   subject to   G z <= h
  with z = [w, b]:
      P = identity, except P[b, b] = 0 (the bias is not penalised)
      q = 0
      G = -y_i [x_i, 1]  (one row per sample; we flip >= into <=)
      h = -1
- The plot shows the decision boundary f(x) = 0 and the margin lines
  f(x) = +-1. No training point lies inside the margin.

The two blobs are well separated (var=0.3), so the constraints are feasible.
'''
import numpy as np
from cvxopt import matrix, solvers

from utils import (get_classification_dataset, train_test_split,
                   plot_decision_boundary, set_solver_options, accuracy)


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        # z = [w, b]: append a column of ones so that z^T [x, 1] = w^T x + b
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        y = y.reshape(-1, 1).astype(np.double)
        # objective: 1/2 ||w||^2 (the bias entry of P is zero)
        P = np.eye(X_aug.shape[1], dtype=np.double)
        P[-1, -1] = 0
        q = np.zeros((X_aug.shape[1], 1), dtype=np.double)
        # inequality constraints: -y_i [x_i, 1] z <= -1
        G = -y * X_aug
        h = -np.ones_like(y, dtype=np.double)

        set_solver_options()
        solver = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return

        z = np.array(solver['x'])
        self.w, self.b = z[:-1], z[-1]
        print('w =', self.w.ravel(), 'b =', self.b, 'margin =', 2 / np.linalg.norm(self.w))

    def predict(self, X, raw=False):
        assert self.w is not None, 'Model not trained yet'
        out = X @ self.w + self.b
        return out if raw else np.sign(out)


if __name__ == '__main__':
    np.random.seed(2023)
    X, y = get_classification_dataset(var=0.3)
    y[y == 0] = -1  # SVMs use labels in {-1, +1}
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM()
    clf.fit(X_train, y_train)
    print(f'Train accuracy: {accuracy(clf, X_train, y_train):.3f}')
    print(f'Test accuracy: {accuracy(clf, X_test, y_test):.3f}')
    plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True)
    plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
