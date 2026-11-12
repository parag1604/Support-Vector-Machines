"""Shared helpers: toy datasets, a train/test split, plotting, and cvxopt options."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from cvxopt import solvers


def get_classification_dataset(
        var: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a dataset with two classes 1 and 0 (two Gaussian blobs).

    Args:
        var: used in the covariance of the gaussians (small var -> separable)

    Returns:
        X: the dataset
        y: the labels
    '''
    mean1 = [-2, 2]
    mean2 = [1, -1]
    cov = [[var, 0], [0, var]]
    cluster1 = np.random.multivariate_normal(mean1, cov, 100)
    cluster2 = np.random.multivariate_normal(mean2, cov, 100)
    X = np.vstack((cluster1, cluster2))
    y = np.concatenate((np.ones(100, dtype=int), np.zeros(100, dtype=int)))
    return X, y


def get_classification_dataset2() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a dataset with two classes 1 and 0: class 1 lies outside the unit
    circle and class 0 inside it, so no straight line can separate them.

    Returns:
        X: the dataset
        y: the labels
    '''
    X = np.random.randn(100, 2)
    y = np.zeros(100)
    y[np.sum(X ** 2, axis=1) > 1] = 1
    return X, y


def get_classification_dataset3(
        n: int = 200,
        noise: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Noisy version of get_classification_dataset2: the label is decided by the
    radius *plus Gaussian noise*, so the two classes overlap near the unit
    circle and no kernel should separate the training set perfectly.

    Args:
        n: number of points
        noise: std of the noise added to the radius before thresholding at 1

    Returns:
        X: the dataset
        y: the labels (1 outside, 0 inside)
    '''
    X = np.random.randn(n, 2)
    r = np.sqrt(np.sum(X ** 2, axis=1)) + noise * np.random.randn(n)
    y = (r > 1).astype(float)
    return X, y


def plot_classification_datapoints(
        X: np.ndarray,
        y: np.ndarray,
        x: Optional[Union[np.ndarray, list]] = None,
        circle: bool = False
) -> None:
    '''
    Plot the classification datapoints.

    Args:
        X: the dataset
        y: the labels
        x: the datapoint to plot
        circle: if True, plot a circle, else plot a dot
    '''
    cluster1 = X[y >= 0.5]
    cluster2 = X[y < 0.5]
    plt.figure(figsize=(8, 6))
    plt.scatter(cluster1[:, 0], cluster1[:, 1], color='green', label='Class 1', marker='+')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], color='red', label='Class 0', marker='x')
    if x is not None:
        if circle:
            plt.scatter(x[0], x[1], marker='o', facecolor='none', edgecolor='blue', s=100)
        else:
            plt.scatter(x[0], x[1], marker='.', color='blue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(
        clf: object,
        X: np.ndarray,
        Y: np.ndarray,
        partition: str = '',
        cmap: str = 'Paired_r',
        margin: bool = False,
        support_vectors: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
) -> None:
    '''
    Plot decision boundary of a classification model.

    Args:
        clf: model with a predict(X, raw=...) method
        X: input data
        Y: target data
        partition: partition of data (used in the title)
        cmap: colormap
        margin: if True, also shade the margin band -1 < f(x) < 1
        support_vectors: if given, circle these points
        save_path: if given, also save the figure there
    '''
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xs.ravel(), ys.ravel()]
    # one vectorised call on the whole grid instead of one call per pixel
    out = np.asarray(clf.predict(grid, raw=True)).ravel()
    if not margin:
        Z = out > 0
    else:
        # 4 regions: outside the margin on either side, and the two halves of
        # the margin band. The contour lines then trace f(x) = -1, 0, +1.
        Z = np.select([out > 1, out < -1, out < 0], [3, 2, 0], default=1)
    Z = Z.reshape(xs.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    if support_vectors is not None:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150,
                    facecolors='none', edgecolors='k', label='support vectors')
        plt.legend()
    plt.title('%s decision boundary' % partition)
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        seed=2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the dataset into train and test sets.

    Returns:
        X_train, y_train, X_test, y_test
    '''
    assert 0.0 <= test_size <= 1.0
    indices = np.arange(X.shape[0])
    np.random.RandomState(seed).shuffle(indices)
    n_test = int(test_size * X.shape[0])
    train_idx, test_idx = indices[n_test:], indices[:n_test]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def set_solver_options(show_progress: bool = True) -> None:
    '''Tight tolerances for cvxopt's QP solver (shared by all versions).'''
    solvers.options['show_progress'] = show_progress
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8


def accuracy(clf: object, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(clf.predict(X).ravel() == y))


if __name__ == "__main__":
    X, y = get_classification_dataset2()
    plot_classification_datapoints(X, y)
