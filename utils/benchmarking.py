import time

from sklearn.metrics import accuracy_score, f1_score

from TbNB import TbNB


def run_experiment(iterative_flag, x_train, y_train, x_test, y_test):
    clf = TbNB(iterative=iterative_flag)

    t0 = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(x_test)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    n_iter = len(clf.decisions_) if iterative_flag and hasattr(clf, "decisions_") else 0

    return {
        "Iterative": iterative_flag,
        "Accuracy": acc,
        "F1-score": f1,
        "Train Time (s)": train_time,
        "Predict Time (s)": pred_time,
        "Iterations": n_iter,
    }


def evaluate_model(clf, X_train, y_train, X_test, y_test, name, pos_label=1):
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(X_test)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label=pos_label)


    return {
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1,
        "Train Time (s)": train_time,
        "Predict Time (s)": pred_time
    }
