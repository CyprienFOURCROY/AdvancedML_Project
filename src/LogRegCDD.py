import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from typing import Tuple, Union


class LogRegCDD:
    """
    Logistic Regression with Cyclic Coordinate Descent (CCD) and Elastic Net Regularization.
    
    Parameters:
        alpha (float): Mixing parameter for Elastic Net (0 = Ridge, 1 = Lasso, between = Elastic Net).
        n_lambda (int): Number of regularization parameters to explore.
        lambda_min_ratio (float): Minimum lambda as a fraction of the maximum lambda.
        max_iter (int): Maximum number of iterations for coordinate descent.
        tol (float): Convergence tolerance.
    """
    def __init__(
        self, alpha: float=1.0, n_lambda: int=100, lambda_min_ratio: float=1e-4, max_iter: int=100, tol: float=1e-6
    ) -> None:
        """
        Initializes the LogRegCDD model with parameters.

        Parameters:
            alpha (float): Mixing parameter between L1 (Lasso) and L2 (Ridge) regularization.
            n_lambda (int): Number of lambda values in the regularization path.
            lambda_min_ratio (float): Ratio of the smallest to largest lambda value.
            max_iter (int): Maximum number of iterations for coordinate descent.
            tol (float): Convergence tolerance for coordinate descent.
        """
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, z: Union[float, np.ndarray], gamma: float) -> float:
        """
        Applies the soft-thresholding operator, used in Lasso regularization.
        
        Parameters:
            z (float or array): Input value(s).
            gamma (float): Threshold parameter.
        
        Returns:
            Transformed value(s) after soft-thresholding.
        """
        return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

    def _logistic_irls_weights(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the working response and weights for Iteratively Reweighted Least Squares (IRLS).
        
        Parameters:
            X (ndarray): Feature matrix.
            y (ndarray): Binary response variable.
            beta (ndarray): Current coefficient estimates.
        
        Returns:
            z_tilde (ndarray): Adjusted response values.
            w (ndarray): Weights for weighted least squares.
        """
        z = X @ beta
        p = expit(z)
        w = p * (1 - p)
        z_tilde = z + (y - p) / np.clip(w, 1e-8, None)
        return z_tilde, w

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic regression model using Cyclic Coordinate Descent with Elastic Net regularization.
        
        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Binary response vector of shape (n_samples,).
        """
        n, p = X.shape
        X = X - X.mean(axis=0)
        self.X_mean = X.mean(axis=0)
        self.coef_path_ = []
        if self.alpha == 0:
    
            self.lambda_max = 1.0  
        else:
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

    def validate(self, X_val: np.ndarray, y_val: np.ndarray, measure: str="logloss") -> np.ndarray:
        """
        Validates the model using a given performance measure.
        
        Parameters:
            X_val (ndarray): Validation feature matrix.
            y_val (ndarray): Validation labels.
            measure (str): Performance metric ('logloss' or 'accuracy').
        
        Returns:
        - scores (ndarray): Computed validation scores across lambda values.
        """
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities for input features.
        
        Parameters:
            X (ndarray): Input feature matrix.
        
        Returns:
            Probabilities of the positive class.
        """
        X = X - self.X_mean
        return expit(X @ self.coef_)

    def plot_score(self, measure: str="logloss") -> None:
        """
        Plots validation scores across lambda values.
        
        Parameters:
            measure (str): Performance metric ('logloss' or 'accuracy').
        """
        if hasattr(self, "validation_scores_"):
            plt.plot(np.log10(self.lambdas_), self.validation_scores_)
            plt.xlabel("log10(lambda)")
            plt.ylabel(f"Validation {measure}")
            plt.title("Validation Curve")
            plt.grid(True)
            plt.show()

    def plot_coef(self) -> None:
        """
        Plots the coefficient paths across lambda values.
        """
        plt.figure()
        for j in range(self.coef_path_.shape[1]):
            plt.plot(np.log10(self.lambdas_), self.coef_path_[:, j])
        plt.xlabel("log10(lambda)")
        plt.ylabel("Coefficients")
        plt.title("Coefficient Paths")
        plt.grid(True)
        plt.show()
