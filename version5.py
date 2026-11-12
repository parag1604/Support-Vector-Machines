'''version5: soft-margin kernel SVM solved in the dual with cvxopt.

What is NEW compared to version4
- version4 is a hard-margin SVM: if the data cannot be separated in the
  feature space of the kernel, the QP is infeasible. version2 fixed this in
  the primal with slack variables xi_i >= 0 and a penalty C sum_i xi_i.
- Doing the same in the dual changes exactly one thing: the multipliers get
  an upper bound C (the "box constraint"):
      max_alpha  sum_i alpha_i - 1/2 sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
      subject to 0 <= alpha_i <= C,  sum_i alpha_i y_i = 0
  (Lagrangian: d/dxi_i [C xi_i - alpha_i xi_i - mu_i xi_i] = 0 gives
   alpha_i = C - mu_i with mu_i >= 0, so alpha_i <= C.)
  For cvxopt we stack two inequality blocks:
      G = [-I; I],  h = [0; C 1]   (i.e. -alpha <= 0 and alpha <= C)
- The KKT conditions now sort the points into three groups:
      alpha_i = 0       : outside the margin (correct side), xi_i = 0
      0 < alpha_i < C   : exactly ON the margin, y_i f(x_i) = 1, xi_i = 0
      alpha_i = C       : inside the margin or misclassified, xi_i >= 0
  Only the middle group satisfies y_s f(x_s) = 1 exactly, so the bias is
  computed from these "margin support vectors" only:
      b = y_s - sum_j alpha_j y_j K(x_j, x_s),   averaged over 0 < alpha_s < C
  (Using the alpha_i = C points too, as version4 would, biases b.)
- New data: the circle of version4 with noise on the radius, so the classes
  overlap and no kernel should fit the training set perfectly.

Note: sigma is the RBF bandwidth; scikit-learn's gamma = 1 / (2 sigma^2).
Run with --compare to check the solution against sklearn.svm.SVC.
'''
import argparse

import numpy as np
from cvxopt import matrix, solvers

from utils import (get_classification_dataset3, train_test_split,
                   plot_decision_boundary, set_solver_options, accuracy)


class SVM:
    def __init__(self, kernel='rbf', C=1.0, sigma=1.0, degree=2, sv_tol=1e-6):
        self.alphas = None
        self.X = None
        self.y = None
        self.b = None
        self.mode = kernel
        self.C = C
        self.sigma = sigma
        self.degree = degree
        self.sv_tol = sv_tol  # relative tolerance (times C) for alpha = 0 / alpha = C

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
        K = self.kernel(X, X)
        # objective: identical to version4
        P = (y @ y.T) * K
        q = -np.ones((n, 1), dtype=np.double)
        # NEW: box constraints 0 <= alpha_i <= C, as two stacked blocks
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.vstack([np.zeros((n, 1)), self.C * np.ones((n, 1))])
        # equality constraint: sum_i alpha_i y_i = 0
        A = y.T
        b = np.zeros((1, 1))

        set_solver_options(show_progress=False)
        solver = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h),
                            matrix(A), matrix(b))
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return False

        alphas = np.array(solver['x']).ravel()
        tol = self.sv_tol * self.C
        sv = alphas > tol                            # all support vectors
        margin_sv = sv & (alphas < self.C - tol)     # 0 < alpha < C: on the margin
        bound_sv = alphas >= self.C - tol            # alpha = C: inside / wrong side

        # NEW: b from the margin support vectors only (y_s f(x_s) = 1 holds there)
        ay = alphas[sv] * y[sv].ravel()
        f_no_b = K[np.ix_(margin_sv, sv)] @ ay
        if margin_sv.any():
            self.b = np.mean(y[margin_sv].ravel() - f_no_b)
        else:
            # degenerate case (every SV at the bound): fall back to all SVs
            print('Warning: no margin support vectors; b averaged over all SVs')
            self.b = np.mean(y[sv].ravel() - K[np.ix_(sv, sv)] @ ay)

        self.alphas = alphas[sv]
        self.X = X[sv]
        self.y = y[sv].ravel()
        self.n_margin, self.n_bound = int(margin_sv.sum()), int(bound_sv.sum())
        print(f'{sv.sum()} support vectors ({self.n_margin} on the margin, '
              f'{self.n_bound} at alpha = C) | b = {self.b:.4f}')
        return True

    def predict(self, X, raw=False):
        assert self.alphas is not None, 'Model not trained yet'
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        out = self.kernel(X, self.X) @ (self.alphas * self.y) + self.b
        return out if raw else np.sign(out)


def compare_with_sklearn(clf, X_train, y_train, X_test):
    '''Fit sklearn's SVC with the same kernel and C and compare.'''
    from sklearn.svm import SVC
    kw = dict(kernel=clf.mode, C=clf.C, tol=1e-6)
    if clf.mode == 'rbf':
        kw['gamma'] = 1.0 / (2 * clf.sigma ** 2)
    elif clf.mode == 'poly':
        kw.update(degree=clf.degree, gamma=1.0, coef0=1.0)
    ref = SVC(**kw).fit(X_train, y_train)
    for name, X in [('train', X_train), ('test', X_test)]:
        ours, theirs = clf.predict(X, raw=True), ref.decision_function(X)
        agree = np.mean(np.sign(ours) == ref.predict(X))
        print(f'sklearn [{name}]: prediction agreement {agree:.3f}, '
              f'max |decision difference| {np.max(np.abs(ours - theirs)):.2e}')
    print(f'sklearn: {ref.n_support_.sum()} support vectors, b = {ref.intercept_[0]:.4f} '
          f'(ours: {len(clf.alphas)}, b = {clf.b:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soft-margin kernel SVM in the dual')
    parser.add_argument('--kernel', choices=['linear', 'poly', 'rbf'], default='rbf')
    parser.add_argument('--C', type=float, default=1.0, help='slack penalty')
    parser.add_argument('--sigma', type=float, default=1.0, help='RBF bandwidth')
    parser.add_argument('--degree', type=int, default=2, help='polynomial degree')
    parser.add_argument('--compare', action='store_true',
                        help='compare against sklearn.svm.SVC')
    parser.add_argument('--save', type=str, default=None,
                        help='save the train plot to this path')
    args = parser.parse_args()

    np.random.seed(2023)
    X, y = get_classification_dataset3()
    y[y == 0] = -1
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVM(kernel=args.kernel, C=args.C, sigma=args.sigma, degree=args.degree)
    if clf.fit(X_train, y_train):
        print(f'Train accuracy: {accuracy(clf, X_train, y_train):.3f}')
        print(f'Test accuracy: {accuracy(clf, X_test, y_test):.3f}')
        if args.compare:
            compare_with_sklearn(clf, X_train, y_train, X_test)
        plot_decision_boundary(clf, X_train, y_train, margin=True,
                               partition=f'Train ({args.kernel}, C={args.C})',
                               support_vectors=clf.X, save_path=args.save)
        plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
