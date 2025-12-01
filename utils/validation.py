import numpy as np
from scipy.sparse import csr_matrix, issparse
import warnings
from typing import Any
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions."""
    pass


def try_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def encode_labels(y):
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("y must contain exactly two classes" if len(classes) > 1
                         else "y contains only one class"
                         )
    mapping = {cls: i for i, cls in enumerate(classes)}
    encoded = np.array([mapping[val] for val in y])

    return encoded, classes


def check_y(y, *, return_classes=False, allow_none=False):
    """
    Validate y for binary classification.

    - Accepts strings or numbers
    - Converts labels to {0, 1} using sklearn-style LabelEncoder
    - Warns if column vector passed
    - Ensures 1-D target vector
    """

    if y is None:
        if allow_none:
            return None
        raise ValueError("y cannot be None")

    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,), for example using ravel().",
            DataConversionWarning,
        )
        y = y.ravel()

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")

    y_enc, classes = encode_labels(y)

    if return_classes:
        return y_enc, classes
    return y_enc


def check_priors(priors):
    """
    Validate custom class priors for binary NB.
    """
    if priors is None:
        raise ValueError("fit_prior was set to False but no class priors were provided")

    priors = np.asarray(priors, dtype=float)

    if len(priors) != 2:
        raise ValueError("class_prior must have length 2")

    if not np.isclose(priors.sum(), 1.0):
        raise ValueError("class_prior must sum to 1.")

    return priors

def check_x(
        X: Any,
        *,
        expected_n_features=None,
        dtype=np.int8,
        allow_object=False,
):
    """
    Validate and standardize input features for Tb-Naive Bayes.
    """

    if issparse(X):
        X = X.tocsr()

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[0] == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        if X.shape[1] == 0:
            raise ValueError(
                f"Found array with 0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )

        if not np.all(np.isfinite(X.data)):
            raise ValueError("X contains NaN or infinite values.")

        X.data = (X.data > 0).astype(dtype)

        if expected_n_features is not None and X.shape[1] != expected_n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but estimator is expecting "
                f"{expected_n_features} features as input"
            )
        return X

    try:
        X = np.asarray(X)
    except Exception:
        raise TypeError(f"X must be array-like, got {type(X)}")

    if X.ndim == 1 or X.ndim == 0:
        raise ValueError(
            "Expected 2D array, got 1D array instead. "
            "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
            "or array.reshape(1, -1) if it contains a single sample."
        )

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    if X.shape[0] == 0:
        raise ValueError(
            f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
        )
    if X.shape[1] == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
        )

    if np.iscomplexobj(X):
        raise ValueError("Complex data not supported")

    if X.dtype == object:
        try:
            X = X.astype(float)
        except ValueError:
            vf = np.vectorize(try_float, otypes=[float])
            X = vf(X)

    if not np.issubdtype(X.dtype, np.number):
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError("X must contain numeric values, got non-numeric dtype.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    X = csr_matrix((X > 0).astype(dtype))

    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but estimator is expecting "
            f"{expected_n_features} features as input"
        )

    return X


def _validate_fit_inputs(estimator, X, y):
    """
        Validate inputs for estimator.fit(X, y) in a binary classifier.

        Parameters
        ----------
        estimator : object
            Estimator instance.
        X : array-like or sparse matrix
            Training data.
        y : array-like or None
            Target labels.

        Returns
        -------
        X_out : csr_matrix
        y_out : tuple or None
            (encoded_y, classes) or None if y=None.

        Raises
        ------
        ValueError
            For invalid shapes, empty data, or non-binary targets.
        """

    X = check_x(X)

    # allowing y to be None is mainly for sklearn compliance
    if y is None:
        estimator.n_features_in_ = X.shape[1]
        return X, None

    y = np.asarray(y)

    y_type = type_of_target(y, input_name="y", raise_unknown=True)
    if y_type != "binary":
        raise ValueError(
            f"Only binary classification is supported. The type of the target is {y_type}."
        )

    y, classes = check_y(y, return_classes=True)

    return X, (y, classes)


def _validate_predict_inputs(estimator, X):
    """
        Validate inputs for estimator.predict(X).

        Parameters
        ----------
        estimator : object
            Fitted estimator.
        X : array-like or sparse matrix
            Input data.

        Returns
        -------
        X_out : csr_matrix
            Validated and transformed input matrix.

        Raises
        ------
        ValueError
            If estimator is not fitted, X is invalid, or has wrong feature size.
        """

    check_is_fitted(estimator)

    X = check_x(X, expected_n_features=estimator.n_features_in_)

    return X
