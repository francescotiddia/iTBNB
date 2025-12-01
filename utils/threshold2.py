import warnings
import numpy as np
from utils.confusion import all_metrics
from sklearn.base import BaseEstimator
import pandas as pd
from utils.validation import check_y


class ThresholdOptimizer(BaseEstimator):
    """
        Cross-validated threshold optimizer for the Threshold-based Naïve Bayes (Tb-NB) classifier.

        The `ThresholdOptimizer` class estimates the optimal decision threshold (τ)
        that minimizes Type I or Type II errors, or maximizes classification accuracy,
        using K-fold cross-validation. It can be applied to any estimator implementing
        a `fit(X, y)` and `predict_scores(X)` interface.

        Attributes
        ----------
        estimator_class : class
            Estimator class implementing `fit(X, y)` and `predict_scores(X)`.
         fit_params:
            Arguments passed to .fit(X, y) methods inside cross-validation loops
        """
    available_metrics = ['precision',
                         'recall',
                         'specificity',
                         'fpr',
                         'f1',
                         'mcc',
                         'misclassification_error',
                         'fnr',
                         'accuracy']

    def __init__(self, tau_grid, estimator_class, estimator_params=None, fit_params=None, K=5, random_state=42,
                 validate_inputs=True):
        self.estimator_class = estimator_class
        self.K = K
        self.random_state = random_state
        self.fit_params = fit_params
        self.estimator_params = estimator_params
        self.tau_grid = tau_grid
        self.validate_inputs = validate_inputs
        self._is_fitted = False

    def _best(self, mat, maximize=True):
        if mat is None or self.tau_grid is None:
            return None
        mean = np.mean(mat, axis=0)
        idx = np.nanargmax(mean) if maximize else np.nanargmin(mean)
        return self.tau_grid_[idx]

    def _generate_folds(self, X, y, K, random_state=None):
        """
                Create K random (non-stratified) cross-validation folds.

                Parameters
                ----------
                X : ndarray of shape (n_samples, n_features)
                    Feature matrix.
                y : ndarray of shape (n_samples,)
                    Binary target labels.
                K : int
                    Number of folds to generate.
                random_state : int, default=42
                    Random seed for reproducibility.

                Returns
                -------
                folds : list of ndarray
                    List containing index arrays for each test fold.
                """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must share the same length.")
        rng = np.random.default_rng(random_state)
        ind = np.arange(X.shape[0])
        rng.shuffle(ind)
        self.folds_ = np.array_split(ind, K)
        return self.folds_

    def fit(self, X, y):
        """
        Perform K-fold cross-validation to estimate Type I/II errors
        for each candidate threshold τ.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Binary class labels

        Returns
        -------
        self : ThresholdOptimizer
            The fitted optimizer containing cross-validated error matrices.
        """
        if self.validate_inputs:
            y = check_y(y)
        self._generate_folds(X, y, self.K, random_state=self.random_state)
        self.fit_params_, self.tau_grid_ = self._validate_input()

        n_folds = len(self.folds_)
        n_tau = len(self.tau_grid_)

        self.metric_mats_ = {}
        for m in self.available_metrics:
            self.metric_mats_[m] = np.zeros((n_folds, n_tau), dtype=np.float32)

        all_idx = np.arange(X.shape[0])

        for k, test_idx in enumerate(self.folds_):
            train_mask = np.ones(X.shape[0], dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_idx[train_mask]
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if self.estimator_params is None:
                estimator_params = {}
            else:
                estimator_params = self.estimator_params
            try:
                model = self.estimator_class(**estimator_params)
            except TypeError as e:
                raise TypeError(f"Invalid estimator_params passed to estimator_class: {e}")

            try:
                model.fit(X_train, y_train, **self.fit_params_)
            except TypeError as e:
                raise TypeError(f"Invalid fit_params passed to model.fit: {e}")
            scores = self._get_scores(model, X_test)

            TP, TN, FP, FN = self.vectorized_confusion_counts(scores, y_test, self.tau_grid_)

            for i, tau in enumerate(self.tau_grid_):
                metrics = all_metrics(TP[i], FP[i], TN[i], FN[i])
                for m in self.available_metrics:
                    self.metric_mats_[m][k, i] = metrics[m]

        self._is_fitted = True
        self._compute_taus()
        return self


    @staticmethod
    def _get_scores(model, X):
        """
        Try in order: predict_scores → predict_proba → decision_function.
        Returns a 1D array of scores suitable for thresholding.
        """

        if hasattr(model, "predict_scores"):
            try:
                scores = model.predict_scores(X)
                return np.asarray(scores).ravel()
            except Exception:
                pass

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X))

                if proba.ndim == 2:
                    if proba.shape[1] != 2:
                        raise ValueError(
                            f"predict_proba must return an array of shape (n_samples, 2) "
                            f"for binary classification, but got shape {proba.shape}"
                        )
                    return proba[:, 1]
                elif proba.ndim == 1:
                    return proba

                else:
                    raise ValueError(
                        f"predict_proba returned an array with invalid shape {proba.shape}"
                    )
            except Exception:
                pass

        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
                return np.asarray(scores).ravel()
            except Exception:
                pass

        raise AttributeError(
            "The model does not provide any usable scoring method among: "
            "predict_scores, predict_proba (validated), decision_function."
        )

    @staticmethod
    def vectorized_confusion_counts(scores, y, tau_grid):
        """
        Compute TP, TN, FP, FN for all thresholds τ in a vectorized way.
        Optimized for very large datasets.
        """

        sort_idx = np.argsort(scores)
        scores_sorted = scores[sort_idx]
        y_sorted = y[sort_idx]
        y_int = (y_sorted == 1).astype(np.int8)

        cum_pos = np.cumsum(y_int)
        cum_neg = np.arange(1, len(y_int) + 1) - cum_pos

        total_pos = cum_pos[-1]
        total_neg = cum_neg[-1]

        split_idx = np.searchsorted(scores_sorted, tau_grid, side='left')
        split_idx = np.clip(split_idx, 0, len(scores_sorted))

        FN = np.where(split_idx > 0, cum_pos[split_idx - 1], 0)
        TN = np.where(split_idx > 0, cum_neg[split_idx - 1], 0)
        TP = total_pos - FN
        FP = total_neg - TN

        return TP, TN, FP, FN

    def _validate_input(self):

        if self.fit_params is None:
            fit_params = {}
        else:
            fit_params = self.fit_params

        if self.tau_grid is None:
            warnings.warn("tau_grid was not provided. Defaulting to [-3,3].")
            tau_grid = np.arange(-3, 3, 0.1)
        else:
            tau_grid = np.sort(np.asarray(self.tau_grid))

        if self.K < 2:
            raise ValueError("K must be >= 2.")

        return fit_params, tau_grid

    def _compute_taus(self):
        self.best_tau_ = {}
        for m in self.available_metrics:
            mat = self.metric_mats_[m]
            maximize = not (m in ["fnr", "fpr", "me"])
            self.best_tau_[m] = self._best(mat, maximize=maximize)
        fpr_mat = self.metric_mats_["fpr"]
        fnr_mat = self.metric_mats_["fnr"]
        self.best_tau_["balanced_error"] = self.tau_grid_[np.argmin(np.mean(fpr_mat + fnr_mat, axis=0))]
        return self

    def summary(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")

        records = []

        for m in self.available_metrics + ["balanced_error"]:
            tau_best = self.best_tau_[m]
            idx = np.argmin(np.abs(self.tau_grid_ - tau_best))

            if m == "balanced_error":
                mat = (self.metric_mats_["fpr"] + self.metric_mats_["fnr"]) / 2
            else:
                mat = self.metric_mats_[m]

            mean_at_best = np.mean(mat[:, idx])
            std_at_best = np.std(mat[:, idx])

            records.append({
                "metric": m,
                "tau_best": tau_best,
                "mean_at_best": mean_at_best,
                "std_at_best": std_at_best,
            })

        return pd.DataFrame(records)



