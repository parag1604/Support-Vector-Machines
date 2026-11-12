'''version4: hard-margin SVM in the dual with a kernel.

What is NEW compared to version3
- The dual and the prediction rule use the data only through dot products
  x_i^T x_j. Replacing them by a kernel K(x_i, x_j) = phi(x_i)^T phi(x_j)
  trains a linear SVM in the feature space of phi without ever computing phi:
      linear: K(x, z) = x^T z
      poly:   K(x, z) = (x^T z + 1)^degree
      rbf:    K(x, z) = exp(-||x - z||^2 / (2 sigma^2))
- New data: class 1 lies outside the unit circle, class 0 inside. No line
  separates them, so --kernel linear makes the hard-margin QP infeasible.
  A degree-2 polynomial kernel can represent the circle exactly, and the RBF
  kernel can separate any finite set of distinct points.

Note: sigma here is the RBF bandwidth. scikit-learn's SVC parameterises the
same kernel as exp(-gamma ||x - z||^2), i.e. gamma = 1 / (2 sigma^2).
'''
import argparse

import numpy as np
from cvxopt import matrix, solvers

from utils import (get_classification_dataset2, train_test_split,
                   plot_decision_boundary, set_solver_options, accuracy)


class SVM:
    def __init__(self, kernel='linear', sigma=1.0, degree=2, sv_tol=1e-4):
        self.alphas = None
        self.X = None
        self.y = None
        self.b = None
        self.mode = kernel
        self.sigma = sigma
        self.degree = degree
        self.sv_tol = sv_tol

    def kernel(self, X1, X2):
        '''Kernel (Gram) matrix between the rows of X1 and the rows of X2.'''
        if self.mode == 'linear':
            return X1 @ X2.T
        elif self.mode == 'poly':
            return (X1 @ X2.T + 1) ** self.degree
        elif self.mode == 'rbf':
            sq_dists = (np.sum(X1 ** 2, axis=1)[:, None]
                        + np.sum(X2 ** 2, axis=1)[None, :] - 2 * X1 @ X2.T)
            return np.exp(-sq_dists / (2 * self.sigma ** 2))
        raise ValueError(f'Unknown kernel: {self.mode}')

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0
        n = X.shape[0]
        # objective: identical to version3 with x_i^T x_j -> K(x_i, x_j)
        P = (y @ y.T) * self.kernel(X, X)
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
            print('Optimization failed: the data is probably not separable '
                  'in the feature space of this kernel (try another kernel).')
            return False

        alphas = np.array(solver['x']).ravel()
        idxs = np.where(alphas > self.sv_tol)[0]
        self.alphas = alphas[idxs]
        self.X = X[idxs]
        self.y = y[idxs].ravel()

        # b = y_s - sum_j alpha_j y_j K(x_j, x_s), averaged over support vectors s
        self.b = np.mean(self.y - (self.alphas * self.y) @ self.kernel(self.X, self.X))
        print(f'{len(idxs)} support vectors | b = {self.b:.4f}')
        return True

    def predict(self, X, raw=False):
        assert self.alphas is not None, 'Model not trained yet'
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        out = self.kernel(X, self.X) @ (self.alphas * self.y) + self.b
        return out if raw else np.sign(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kernel SVM in the dual')
    parser.add_argument('--kernel', choices=['linear', 'poly', 'rbf'], default='rbf')
    parser.add_argument('--sigma', type=float, default=1.0, help='RBF bandwidth')
    parser.add_argument('--degree', type=int, default=2, help='polynomial degree')
    args = parser.parse_args()

    np.random.seed(2023)
    X, y = get_classification_dataset2()
    y[y == 0] = -1
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM(kernel=args.kernel, sigma=args.sigma, degree=args.degree)
    if clf.fit(X_train, y_train):
        print(f'Train accuracy: {accuracy(clf, X_train, y_train):.3f}')
        print(f'Test accuracy: {accuracy(clf, X_test, y_test):.3f}')
        plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True,
                               support_vectors=clf.X)
        plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
