if issparse(X):
    X = X.tocsr()

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if X.shape[0] == 0:
        raise ValueError(f"Found array with 0 samples (shape={X.shape}).")

    if X.shape[1] == 0:
        raise ValueError(f"Found array with 0 features (shape={X.shape}).")

    if not np.all(np.isfinite(X.data)):
        raise ValueError("X contains NaN or infinite values.")

    # SE I DATI SONO CONTINUI → BINARIZZA
    if not np.issubdtype(X.data.dtype, np.integer):
        X.data = (X.data > 0).astype(dtype)
    else:
        # SE SONO INTERI → COUNT MODE
        if X.data.min() < 0:
            raise ValueError("Negative counts not allowed.")
        X.data = X.data.astype(dtype)

    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but estimator expects {expected_n_features}."
        )

    return X



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
