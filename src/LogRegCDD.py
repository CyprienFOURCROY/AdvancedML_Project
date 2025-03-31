import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


class LogRegCDD:
    def __init__(
        self, alpha=1.0, n_lambda=100, lambda_min_ratio=1e-4, max_iter=100, tol=1e-6
    ):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, z, gamma):
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def _logistic_irls_weights(self, X, y, beta):
        z = X @ beta
        p = expit(z)
        w = p * (1 - p)
        z_tilde = z + (y - p) / np.clip(w, 1e-8, None)
        return z_tilde, w

    def fit(self, X, y):
        n, p = X.shape
        X = X - X.mean(axis=0)
        self.X_mean = X.mean(axis=0)
        self.coef_path_ = []
        self.lambda_max = np.max(np.abs(X.T @ (y - y.mean()))) / (n * self.alpha)
        self.lambdas_ = np.logspace(
            np.log10(self.lambda_max),
            np.log10(self.lambda_max * self.lambda_min_ratio),
            self.n_lambda,
        )[::-1]
        beta = np.zeros(p)
        for lam in self.lambdas_:
            for _ in range(self.max_iter):
                beta_old = beta.copy()
                z_tilde, w = self._logistic_irls_weights(X, y, beta)
                for j in range(p):
                    r_j = z_tilde - X @ beta + X[:, j] * beta[j]
                    zj = np.sum(w * X[:, j] * r_j)
                    pj = np.sum(w * X[:, j] ** 2)
                    if pj < 1e-8:
                        continue
                    if self.alpha == 0:
                        beta[j] = zj / (pj + lam)
                    else:
                        beta[j] = self._soft_threshold(zj / pj, lam * self.alpha) / (
                            1 + lam * (1 - self.alpha) / pj
                        )

                if np.linalg.norm(beta - beta_old, ord=1) < self.tol:
                    break
            self.coef_path_.append(beta.copy())
        self.coef_path_ = np.array(self.coef_path_)
        self.coef_ = self.coef_path_[-1]

    def validate(self, X_val, y_val, measure="logloss"):
        X_val = X_val - self.X_mean
        scores = []
        for beta in self.coef_path_:
            p = expit(X_val @ beta)
            if measure == "logloss":
                p = np.clip(p, 1e-15, 1 - 1e-15)
                score = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
            elif measure == "accuracy":
                score = np.mean((p > 0.5) == y_val)
            scores.append(score)
        self.validation_scores_ = np.array(scores)
        return self.validation_scores_

    def predict_proba(self, X):
        X = X - self.X_mean
        return expit(X @ self.coef_)

    def plot_score(self, measure="logloss"):
        if hasattr(self, "validation_scores_"):
            plt.plot(np.log10(self.lambdas_), self.validation_scores_)
            plt.xlabel("log10(lambda)")
            plt.ylabel(f"Validation {measure}")
            plt.title("Validation Curve")
            plt.grid(True)
            plt.show()

    def plot_coef(self):
        plt.figure()
        for j in range(self.coef_path_.shape[1]):
            plt.plot(np.log10(self.lambdas_), self.coef_path_[:, j])
        plt.xlabel("log10(lambda)")
        plt.ylabel("Coefficients")
        plt.title("Coefficient Paths")
        plt.grid(True)
        plt.show()
