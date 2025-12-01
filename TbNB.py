import pickle
import warnings

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from utils.decision import *
from utils.threshold import ThresholdOptimizer
from utils.validation import _validate_predict_inputs, _validate_fit_inputs, check_priors


class TbNB(ClassifierMixin, BaseEstimator):
    """
        Threshold-based Naïve Bayes (Tb-NB) classifier.

        Parameters
        ----------
        fit_prior : bool, default=True
            Whether to estimate class prior probabilities from the data.
        class_prior : array-like of shape (2,), optional
            Predefined class priors. Used only if `fit_prior=False`.
        alpha : float, default=1
            Additive (Laplace) smoothing parameter.
        iterative : bool, default=False
            Whether to enable iterative thresholding (iTb-NB).

        References
        ----------
        Romano, M., Zammarchi, G., & Conversano, C. (2024).
            *Iterative Threshold-Based Naïve Bayes Classifier*.
            Statistical Methods & Applications, 33, 235–265.
            https://doi.org/10.1007/s10260-023-00721-1
        """

    AVAILABLE_CRITERIA = ['precision',
                          'recall',
                          'specificity',
                          'fpr',
                          'f1',
                          'mcc',
                          'misclassification_error',
                          'fnr',
                          'accuracy',
                          'balanced_error']

    # TODO : controllare che misclassification_error sia giusto
    def __init__(self, fit_prior=True, class_prior=None, alpha=1, iterative=False, optimize_threshold=True,
                 criterion="balanced_error",
                 tau_grid="observed",
                 K=5,
                 n_tau=50,
                 random_state=42,
                 p_iter=0.2,
                 s_iter=20,
                 ):
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.alpha = alpha
        self.iterative = iterative
        self.optimize_threshold = optimize_threshold
        self.criterion = criterion
        self.tau_grid = tau_grid
        self.K = K
        self.n_tau = n_tau
        self.random_state = random_state
        self.p_iter, self.s_iter = p_iter, s_iter

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = False
        tags.input_tags.sparse = True
        tags.target_tags.required = True
        tags.classifier_tags.poor_score = True
        return tags

    def best_tau(self, metric):
        if not hasattr(self, "optimizer_"):
            raise RuntimeError("Threshold has not been optimized yet. Call fit(..., optimize_threshold=True).")
        return self.optimizer_.best_tau_[metric]

    @property
    def lw_present_(self):
        return self.log_conditional_pos_[1] - self.log_conditional_pos_[0]

    @property
    def lw_absent_(self):
        return self.log_conditional_neg_[1] - self.log_conditional_neg_[0]

    @property
    def lambda_scores_(self):
        return self.lw_present_ + self.lw_absent_

    @staticmethod
    def _estimate_prior(Y: np.ndarray):
        """
        Estimate class occurrence counts and prior probabilities.

        Parameters
        ----------
        Y : one-hot encoded ndarray of shape (n_samples, 2)

        Returns
        -------
        class_occurrences : ndarray of shape (2,)
            Count of samples per class.
        class_prior : ndarray of shape (2,)
            Empirical class prior probabilities.
        """
        class_occurrences = Y.sum(axis=0)
        n = class_occurrences.sum()
        class_prior = class_occurrences / n
        return class_occurrences, class_prior

    @staticmethod
    def _estimate_feature_counts(X: csr_matrix, Y: np.ndarray):
        """
        Compute feature counts conditional on class.

        Parameters
        ----------
        X : csr_matrix of shape (n_samples, n_features)
            Binary Bag-of-Words matrix (presence/absence of terms).
        Y : ndarray of shape (n_samples, 2)
            Binary indicator matrix of class labels.

        Returns
        -------
        feature_counts : ndarray of shape (2, n_features)
            Count of each feature across positive and negative samples.
        """

        feature_counts = Y.T @ X
        smooth_fc = np.asarray(feature_counts)
        return smooth_fc

    # TODO: verificare che alpha = 1 sia corretto

    def _estimate_log_conditional_probabilities(self, feature_counts, alpha=1.0):
        """
        Estimate log-conditional probabilities with Laplace smoothing.

        For each class, computes:
        P(w | class) = (count + α) / (N_class + 2α)

        The logarithms of both the presence and absence likelihoods are returned,
        which are later used to compute log-likelihood ratios.

        Parameters
        ----------
        feature_counts : ndarray of shape (2, n_features)
            Class-specific feature occurrence counts.
        alpha : float, default=1.0
            Smoothing parameter to prevent zero probabilities.

        Returns
        -------
        log_likelihood_pres : ndarray of shape (2, n_features)
            Log-probabilities of word presence given each class.
        log_likelihood_abs : ndarray of shape (2, n_features)
            Log-probabilities of word absence given each class.
        """

        pres_counts = feature_counts
        abs_counts = self.class_occurrences_[:, None] - feature_counts

        pres_likelihood = (pres_counts + alpha) / (self.class_occurrences_[:, None] + 2 * alpha)
        abs_likelihood = (abs_counts + alpha) / (self.class_occurrences_[:, None] + 2 * alpha)

        log_likelihood_pres = np.log(pres_likelihood)
        log_likelihood_abs = np.log(abs_likelihood)

        return log_likelihood_pres, log_likelihood_abs

    def fit(
            self,
            X,
            y

    ):
        """
        Fit the Tb-NB classifier to training data.

        Parameters
        ----------
        X : matrix of shape (n_samples, n_features)
            Binary Bag-of-Words matrix.
        y : array-like of shape (n_samples,)
             Binary class labels (0 or 1).
        optimize_threshold : bool, default=True
            Whether to estimate an optimal decision threshold τ via cross-validation.
        criterion : {"precision", "recall", "specificity", "fpr", "f1",
             "mcc", "me", "fnr", "accuracy", "balanced_error"}, default="balanced_error"
            Metric used to select the optimal threshold.
        tau_grid : {"observed", tuple of (low, high)}, default="observed"
            Range or method for generating candidate thresholds.
        K : int, default=5
            Number of folds for cross-validation threshold optimization.
        n_tau : int, default=50
            Number of threshold candidates to evaluate.
        random_state: int, default=42
            Seed used for train-test split.

        Returns
        -------
        self : TbNB
            Fitted classifier.

        """

        if self.criterion not in self.AVAILABLE_CRITERIA:
            raise ValueError("{} is not an available criterion.".format(self.criterion))

        result = _validate_fit_inputs(self, X, y)
        X = result[0]
        y_data = result[1]

        if y_data is None:
            return self

        y, classes = y_data

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

        alpha = self.alpha

        Y = np.column_stack((1 - y, y))

        if self.fit_prior:
            self.class_occurrences_, self.class_prior_ = self._estimate_prior(Y)
        else:
            check_priors(self.class_prior)
            self.class_occurrences_ = Y.sum(axis=0)
            self.class_prior_ = self.class_prior

        self.feature_counts_ = self._estimate_feature_counts(X, Y)
        self.log_conditional_pos_, self.log_conditional_neg_ = \
            self._estimate_log_conditional_probabilities(self.feature_counts_, alpha)

        scores = self.predict_scores(X)

        if isinstance(self.tau_grid, str) and self.tau_grid == "observed":
            low, high = np.percentile(scores, [1, 99])
            tau_grid = np.linspace(low, high, self.n_tau)
        elif isinstance(self.tau_grid, tuple) and len(self.tau_grid) == 2:
            tau_grid = np.linspace(self.tau_grid[0], self.tau_grid[1], self.n_tau)

        if self.optimize_threshold:
            optimizer = ThresholdOptimizer(
                tau_grid=tau_grid,
                estimator_class=TbNB,
                estimator_params={
                    "fit_prior": self.fit_prior,
                    "class_prior": self.class_prior,
                    "alpha": self.alpha,
                    "iterative": False,
                    "optimize_threshold": False, 
                    "criterion": self.criterion,
                    "tau_grid": self.tau_grid,
                    "K": self.K,
                    "n_tau": self.n_tau,
                    "random_state": self.random_state,
                },
                K=self.K,
                random_state=self.random_state,
            )

            optimizer.fit(X, y)
            self.optimizer_ = optimizer
            self.threshold_ = self.optimizer_.best_tau_[self.criterion]

        if self.iterative:
            self.decisions_ = iterate_threshold(scores, y, self.threshold_, p=self.p_iter, s=self.s_iter)

        return self

    # TODO: ora come ora non puoi modificare i parametri della iterazione

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def predict_scores(self, X: csr_matrix):
        Lw = self.lw_present_
        Lnw = self.lw_absent_

        delta = Lw - Lnw
        bias = np.sum(Lnw)

        scores = X @ delta
        scores = np.asarray(scores).ravel() + bias

        return scores

    def predict(self, X, threshold=None):

        X = _validate_predict_inputs(self, X)

        scores = self.predict_scores(X)

        tau = threshold if threshold is not None else getattr(self, "threshold_", None)
        if tau is None:
            raise ValueError(
                "No threshold specified. Fit must run threshold optimization or pass threshold=..."
            )

        pred = (scores >= tau).astype(int)

        if self.iterative:
            if not hasattr(self, "decisions_"):
                warnings.warn("Iterative mode enabled but no decisions_ found.")
                return pred
            pred_int = predict_from_decisions(scores, self.decisions_, default_threshold=tau).astype(int)
        else:
            pred_int = pred
        return self.classes_[pred_int]

# TODO: migliorare i nomi delle variabili e dei metodi

# TODO: verificare che funzioni con altre label
