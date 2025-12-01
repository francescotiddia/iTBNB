import numpy as np
from scipy.stats import gaussian_kde
from dataclasses import dataclass


@dataclass
class Decision:
    iteration: int
    start: float
    end: float
    tau: float
    x_max_pos: float
    x_max_neg: float
    direction: str


def _estimate_sigma(X, tau, p):
    """
        Estimate the uncertainty width (σ) around a threshold τ.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Observed scores or decision values.
        tau : float
            Current threshold estimate.
        p : float
            Quantile used to determine σ (typically between 0 and 1).

        Returns
        -------
        sigma : float
            Estimated uncertainty width σ defining the neighborhood around τ.
        """
    distances = np.abs(X - tau)
    sigma = np.quantile(distances, p)
    return sigma


def _intersect_threshold(scores_pos, scores_neg):
    """
                Estimate the intersection point (new threshold) between two class densities.

                Parameters
                ----------
                scores_pos : array-like of shape (n_pos,)
                    Scores associated with positive class instances.
                scores_neg : array-like of shape (n_neg,)
                    Scores associated with negative class instances.

                Returns
                -------
                tau : float
                    Intersection point of the two class densities (refined threshold).
                x_max_pos : float
                    Score corresponding to the argmax of the positive class density.
                x_max_neg : float
                    Score corresponding to the argmax of the negative class density.

                Raises
                ------
                InterruptedError
                    If the two densities do not intersect (no sign change in their difference).
                """
    f_pos = gaussian_kde(scores_pos)
    f_neg = gaussian_kde(scores_neg)
    low = min(scores_pos.min(), scores_neg.min())
    high = max(scores_pos.max(), scores_neg.max())
    x = np.linspace(low, high, 1000)
    pdf_pos = f_pos(x)
    pdf_neg = f_neg(x)
    delta = pdf_pos - pdf_neg
    sign_change = np.any(np.sign(delta[:-1]) != np.sign(delta[1:]))
    if not sign_change:
        raise InterruptedError
    tau = x[np.argmin(np.abs(delta))]
    x_max_pos = x[np.argmax(pdf_pos)]
    x_max_neg = x[np.argmax(pdf_neg)]

    return tau, x_max_pos, x_max_neg


def iterate_threshold(X, Y, tau, p=0.2, s=20, i=0, epsilon=1e-3):
    """
        Iteratively refine the decision threshold based on uncertain observations.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Decision scores (Λ values) for all observations.
        Y : array-like of shape (n_samples,)
            Binary class labels (0 = negative, 1 = positive).
        tau : float
            Initial threshold (typically from Tb-NB).
        p : float, default=0.2
            Proportion of cases around τ to consider for refinement.
        s : int, default=20
            Minimum number of cases required for a valid iteration.
        i : int, default=0
            Initial iteration counter (used for recursive or chained calls).
        epsilon : float, default=1e-3
            Minimum distance between class density maxima before convergence.

        Returns
        -------
        decisions : list of Decision
            List of `Decision` objects, each describing an iteration and its parameters.

        """
    decisions = []
    x = X.copy()

    while True:
        i += 1
        sigma = _estimate_sigma(x, tau, p)
        mask = (X > (tau - sigma)) & (X < (tau + sigma))
        omega_x, omega_y = X[mask], Y[mask]

        if omega_x.shape[0] < s:
            break

        x_pos = omega_x[omega_y == 1]
        x_neg = omega_x[omega_y == 0]

        if len(x_pos) < 2 or len(x_neg) < 2:
            break

        try:
            tau, x_max_pos, x_max_neg = _intersect_threshold(x_pos, x_neg)
        except InterruptedError:
            break
        except ValueError as e:
            break

        if abs(x_max_pos - x_max_neg) < epsilon:
            break
        elif x_max_pos > x_max_neg:
            direction_i = "r"
        else:
            direction_i = "l"

        decisions.append(
            Decision(i, omega_x.min(), omega_x.max(), tau, x_max_pos, x_max_neg, direction_i)
        )

        x = omega_x

    return decisions


def predict_from_decisions(X, decisions, default_threshold=0.0):
    """
        Generate class predictions using a sequence of refined decision rules.


        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Decision scores to classify.
        decisions : list of Decision
            Sequence of iterative threshold refinement steps.
        default_threshold : float, default=0.0
            Base threshold used for classification outside any refined regions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1) after applying all iterative decisions.
        """

    X = np.asarray(X)

    y_pred = (X > default_threshold).astype(int)

    decisions = sorted(decisions, key=lambda d: d.iteration)

    for d in decisions:
        mask = (X >= d.start) & (X <= d.end)
        if not np.any(mask):
            continue

        if d.direction == "r":
            y_pred[mask & (X > d.tau)] = 1
            y_pred[mask & (X <= d.tau)] = 0
        else:
            y_pred[mask & (X < d.tau)] = 1
            y_pred[mask & (X >= d.tau)] = 0

    return y_pred
