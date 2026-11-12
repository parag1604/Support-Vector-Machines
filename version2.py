'''version2: soft-margin SVM (slack variables), solved in the primal with cvxopt.

What is NEW compared to version1
- The blobs now overlap (var=1.0), so the hard-margin QP is infeasible.
  Each point gets a slack xi_i >= 0 that lets it violate the margin, at a
  cost C per unit of violation:
      min_{w, b, xi}  1/2 ||w||^2 + C sum_i xi_i
      subject to      y_i (w^T x_i + b) >= 1 - xi_i,   xi_i >= 0
- In cvxopt form, z = [w, b, xi] (length D + 1 + N):
      P = diag(1, ..., 1, 0, 0, ..., 0)   (only w is penalised quadratically)
      q = [0, ..., 0, C, ..., C]          (the slacks are penalised linearly)
      G = [[-y_i x_i, -y_i, -e_i],        margin constraints
           [   0,       0,   -I ]]        xi >= 0
      h = [-1, ..., -1, 0, ..., 0]
- Small C tolerates many violations (wide margin); large C approaches the
  hard margin. Try --C 0.01 and --C 100.
'''
import argparse

import numpy as np
from cvxopt import matrix, solvers

from utils import (get_classification_dataset, train_test_split,
                   plot_decision_boundary, set_solver_options, accuracy)


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, c=1.0):
        n, d = X.shape
        y = y.reshape(-1, 1).astype(np.double)
        # objective: 1/2 ||w||^2 + c * sum(xi); entries for b and xi are zero in P
        P = np.zeros((d + 1 + n, d + 1 + n), dtype=np.double)
        P[:d, :d] = np.eye(d)
        q = np.zeros((d + 1 + n, 1), dtype=np.double)
        q[d + 1:] = c
        # inequality constraints
        # (a) margin: -y_i (w^T x_i + b) - xi_i <= -1
        G_margin = np.hstack([-y * X, -y, -np.eye(n)])
        # (b) slack positivity: -xi_i <= 0
        G_slack = np.hstack([np.zeros((n, d + 1)), -np.eye(n)])
        G = np.vstack([G_margin, G_slack])
        h = np.vstack([-np.ones((n, 1)), np.zeros((n, 1))])

        set_solver_options()
        solver = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return

        z = np.array(solver['x'])
        self.w, self.b, self.xi = z[:d], z[d], z[d + 1:].ravel()
        print('w =', self.w.ravel(), 'b =', self.b,
              '| points violating the margin (xi > 0):', int(np.sum(self.xi > 1e-6)))

    def predict(self, X, raw=False):
        assert self.w is not None, 'Model not trained yet'
        out = X @ self.w + self.b
        return out if raw else np.sign(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soft-margin primal SVM')
    parser.add_argument('--C', type=float, default=0.5, help='slack penalty')
    args = parser.parse_args()

    np.random.seed(2023)
    X, y = get_classification_dataset(var=1.0)
    y[y == 0] = -1
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM()
    clf.fit(X_train, y_train, c=args.C)
    print(f'Train accuracy: {accuracy(clf, X_train, y_train):.3f}')
    print(f'Test accuracy: {accuracy(clf, X_test, y_test):.3f}')
    plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True)
    plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
