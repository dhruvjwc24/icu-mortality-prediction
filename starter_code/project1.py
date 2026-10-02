"""
EECS 445 Fall 2026

This script contains most of the work for the project. You will need to fill in every TODO comment.
"""

import random
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score ,confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import helper

__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)


def generate_feature_vector(df: pd.DataFrame, challenge: bool = False) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: DataFrame with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    df_replaced = df.replace(-1, np.nan)

    feature_dict = {}

    if challenge:
        time_df = df_replaced["Time"].str.split(":", expand=True).astype(int)
        hrs, mins = time_df[0], time_df[1]
        ordered_df = df_replaced.assign(minutes=hrs*60+mins).sort_values("minutes", kind="stable")

        for sv in static_variables:
            matching_series = ordered_df.loc[ordered_df["Variable"] == sv, "Value"].dropna()
            feature_dict[sv] = next(iter(matching_series), np.nan)

        for tv in timeseries_variables:
            matching_series = ordered_df.loc[ordered_df["Variable"] == tv, "Value"].dropna()
            feature_dict[f"min_{tv}"] = np.nan if matching_series.empty else matching_series.min()
            feature_dict[f"med_{tv}"] = np.nan if matching_series.empty else matching_series.median()
            feature_dict[f"max_{tv}"] = np.nan if matching_series.empty else matching_series.max()
            feature_dict[f"std_{tv}"] = np.nan if matching_series.empty else matching_series.std()
            feature_dict[f"last_{tv}"] = np.nan if matching_series.empty else matching_series.iloc[-1]
            feature_dict[f"count_{tv}"] = matching_series.size

    else:
        for sv in static_variables:
            matching_series = df_replaced.loc[df_replaced["Variable"] == sv, "Value"].dropna()
            feature_dict[sv] = next(iter(matching_series), np.nan)
        for tv in timeseries_variables:
            matching_series = df_replaced.loc[df_replaced["Variable"] == tv, "Value"].dropna()
            feature_dict[f"max_{tv}"] = np.nan if matching_series.empty else matching_series.max()

    return feature_dict


def impute_missing_values(X: npt.NDArray, challenge: bool = False) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: (n, d) feature matrix, which could contain missing values

    Returns:
        X: (n, d) feature matrix, without missing values
    """
    if challenge: return X

    for colNum in range(X.shape[1]):
        col = X[:, colNum]
        colMean = np.nanmean(col)

        col[np.isnan(col)] = colMean

    return X


def normalize_feature_matrix(X: npt.NDArray, challenge: bool = False) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (n, d) feature matrix

    Returns:
        X: (n, d) feature matrix with values that are normalized per column
    """
    # X_min = np.nanmin(X, axis=0, keepdims=True)
    # X_max = np.nanmax(X, axis=0, keepdims=True)

    # return (X - X_min) / (X_max - X_min)

    # OR

    if challenge: return X

    scaler = MinMaxScaler()
    return scaler.fit_transform(X)
    

def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: The name of the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel: The name of the Kernel used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but recommended): implement function based on docstring

    if loss == "logistic":
        raise NotImplementedError()
    elif loss == "squared_error":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown loss function: {loss}")


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy",
    bootstrap: bool = False,
) -> float | tuple[float, float, float]:
    """
    Calculates the performance metric as evaluated on the true labels y_true versus the predicted scores from
    clf_trained and X. Returns single sample performance if bootstrap is False, otherwise returns the median
    and the empirical 95% confidence interval. You may want to implement an additional helper function to
    reduce code redundancy.

    Args:
        clf_trained: a fitted sklearn estimator
        X: (n, d) feature matrix
        y_true: (n, ) vector of labels in {+1, -1}
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1_score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
        bootstrap: whether to use bootstrap sampling for performance estimation

    Returns:
        If bootstrap is False, returns the performance for the specific metric. If bootstrap is True, returns
        the median and the empirical 95% confidence interval.
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if bootstrap:
        n = len(y_true)
        bootstrapped_performances = []
        for _ in range(1000):
            indices = np.random.randint(low=0, high=n, size=n)
            bootstrapped_performances.append(performance(clf_trained, X[indices], y_true[indices], metric, False))
        low, med, high = np.percentile(bootstrapped_performances, [2.5, 50, 97.5])
        return med, low, high

    else:
        if hasattr(clf_trained, "decision_function"): # type LogisticRegression()
            y_score = clf_trained.decision_function(X)
        else: # type KernelRidge (no decision_function(), just use predict())
            y_score = clf_trained.predict(X)

        if metric in ["auroc", "average_precision"]:
            match metric:
                case "auroc":
                    return roc_auc_score(y_true, y_score)
                case "average_precision":
                    return average_precision_score(y_true, y_score)
        else:
            y_pred = np.where(y_score >= 0, 1, -1)
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            match metric:
                case "accuracy": # basic how many write / total num
                    return (TP + TN) / (TP + TN + FP + FN)
                case "precision": # of those pred pos, how many acc positive (used when FP is costly)
                    return 0 if TP + FP == 0 else (TP) / (TP + FP)
                case "f1_score": # good balance between precision and recall
                    return 0 if TP + FP + FN == 0 else (2 * TP) / (2 * TP + FP + FN)
                case "sensitivity": # aka recall, prop of actual pos cases mod correctly classifies (used when FN is costly)
                    return 0 if TP + FN == 0 else (TP) / (TP + FN)
                case "specificity":
                    return 0 if TN + FP == 0 else (TN) / (TN + FP)


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy"
                other options: "precision", "f1_score", "auroc", "average_precision",
                "sensitivity", and "specificity")

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """

    skf = StratifiedKFold(n_splits=k)
    min_perf = float("inf")
    max_perf = float("-inf")
    running_perf = 0

    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        # fit model over train data
        clf.fit(X_train, y_train)

        perf = performance(clf, X_val, y_val, metric, False)

        min_perf = min(min_perf, perf)
        max_perf = max(max_perf, perf)
        running_perf += perf

    avg_perf = running_perf / k
    return (avg_perf, min_perf, max_perf)


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of true labels in {+1, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default="accuracy",
                other options: "precision", "f1_score", "auroc", "average_precision", "sensitivity",
                and "specificity")
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """

    best_cv_performance = float("-inf")
    best_candidate = ()
    for C in C_range:
        for penalty in penalties:
            l1_ratio = 0 if penalty == "l2" else 1
            clf_candidate = LogisticRegression(l1_ratio=l1_ratio, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            clf_cv_performance = cv_performance(clf_candidate, X, y, metric, k=k)
            if clf_cv_performance[0] > best_cv_performance: 
                best_candidate = (C, penalty)
                best_cv_performance = clf_cv_performance[0]

    return best_candidate


def select_param_RBF(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    gamma_range: list[float],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of binary labels {1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy",
                other options: "precision", "f1_score", "auroc", "average_precision",
                "sensitivity", and "specificity")
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """

    best_cv_performance = float("-inf")
    best_candidate = ()
    for C in C_range:
        for gamma in gamma_range:
            clf_candidate = KernelRidge(alpha=1/(2*C), kernel="rbf", gamma=gamma)
            clf_cv_performance = cv_performance(clf_candidate, X, y, metric, k=k)
            if clf_cv_performance[0] > best_cv_performance: 
                best_candidate = (C, gamma)
                best_cv_performance = clf_cv_performance[0]

    return best_candidate


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            clf = LogisticRegression(
                l1_ratio=1 if penalty == "l1" else 0,
                C=C,
                solver="liblinear",
                fit_intercept=False,
                random_state=seed,
            )

            clf.fit(X, y)
            w = clf.coef_[0]

            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()

def q1d(X, feature_names, seed, path="output/q1d.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    pd.DataFrame({
        "Feature": feature_names,
        "Mean Value": X.mean(axis=0).round(4),
        "Interquartile Range": (q75 - q25).round(4),
    }).to_csv(path, index=False)

def q2c(X, y, metrics, seed, path="output/q2c.csv"):
    C_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    penalties = ["l1", "l2"]

    rows = []
    for metric in metrics:
        best_C, best_penalty = select_param_logreg(
            X=X, y=y, C_range=C_range, penalties=penalties, metric=metric, k=5
        )

        clf = LogisticRegression(
            l1_ratio=1 if best_penalty == "l1" else 0,
            C=best_C,
            solver="liblinear",
            fit_intercept=False,
            random_state=seed,
        )
        mean, low, high = cv_performance(clf, X, y, metric=metric, k=5)
        rows.append({
            "Performance Measure": metric,
            "C": best_C,
            "Penalty": best_penalty,
            "Mean (Min, Max) CV Performance": f"{mean:.4f} ({low:.4f}, {high:.4f})",
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def q2d(X_train, y_train, X_test, y_test, metrics, seed, path="output/q2d.csv"):
    C_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    penalties = ["l1", "l2"]

    rows = []
    best_C, best_penalty = select_param_logreg(
        X=X_train, y=y_train, C_range=C_range, penalties=penalties, metric="auroc", k=5
    )

    clf = LogisticRegression(
        l1_ratio=1 if best_penalty == "l1" else 0,
        C=best_C,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed,
    )

    clf.fit(X_train, y_train)

    for metric in metrics:
        med, low, high = performance(clf, X_test, y_test, metric, bootstrap=True)
        rows.append({
            "Performance Measure": metric,
            "C": best_C,
            "Penalty": best_penalty,
            "Median": f"{med:.4f}",
            "95% CI": f"({low:.4f}, {high:.4f})",
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def q2e(X, y):
    C_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    plot_weight(X, y, C_range, ["l1", "l2"])

def q2f(X, y, feature_names, path="output/q2f.csv"):
    clf = LogisticRegression(
        l1_ratio=1,
        C=1,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed,
    )

    clf.fit(X, y)

    df = pd.DataFrame({
        "Coefficient": clf.coef_[0],
        "Feature Name": feature_names,
    }).sort_values("Coefficient", ascending=False)

    df = pd.concat([df.head(4), df.tail(4)])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")

def q3p2b(X_train, y_train, X_test, y_test, metrics, seed, path="output/q3p2b.csv"):
    
    rows = []
    clf = LogisticRegression(
        l1_ratio=0,
        C=1,
        solver="liblinear",
        fit_intercept=False,
        class_weight={-1: 1, 1: 50},
        random_state=seed,
    )

    clf.fit(X_train, y_train)

    for metric in metrics:
        med, low, high = performance(clf, X_test, y_test, metric, bootstrap=True)
        rows.append({
            "Performance Measure": metric,
            "Median": f"{med:.4f}",
            "95% CI": f"({low:.4f}, {high:.4f})",
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def q3p2d(X_train, y_train, X_test, y_test, seed, path="output/q3p2d.png"):

    weights = [{-1: 1, 1: 1}, {-1: 1, 1: 5}]
    for weight in weights:
        clf = LogisticRegression(
            l1_ratio=0,
            C=1,
            solver="liblinear",
            fit_intercept=False,
            class_weight=weight,
            random_state=seed,
        )

        clf.fit(X_train, y_train)

        y_score = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)

        plt.plot(fpr, tpr, label=f"Wn={weight[-1]} | Wp={weight[1]}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(path, dpi=200)
    plt.close()

def q4p1b(X_train, y_train, X_test, y_test, metrics, seed, path="output/q4p1b.csv"):
    rows = []

    clf_lr = LogisticRegression(
        l1_ratio=0,
        C=1,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed,
    )

    clf_rr = KernelRidge(
        alpha=1/2,
        kernel="linear",
    )

    clf_lr.fit(X_train, y_train)
    clf_rr.fit(X_train, y_train)

    for metric in metrics:
        lr_med, lr_low, lr_high = performance(clf_lr, X_test, y_test, metric, bootstrap=True)
        rr_med, rr_low, rr_high = performance(clf_rr, X_test, y_test, metric, bootstrap=True)

        rows.append({
            "Metric": metric,
            "lr_Median": f"{lr_med:.4f}",
            "lr_95% CI": f"({lr_low:.4f}, {lr_high:.4f})",
            "rr_Median": f"{rr_med:.4f}",
            "rr_95% CI": f"({rr_low:.4f}, {rr_high:.4f})",
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def q4p2b(X_train, y_train, metric="auroc", path="output/q4p2b.csv"):
    C = 1
    gamma_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    rows = []

    for gamma in gamma_range:
        clf = KernelRidge(
            alpha=1/(2*C),
            kernel="rbf",
            gamma=gamma,
        )

        med, low, high = cv_performance(clf, X_train, y_train, metric, k=5)
        rows.append({
            "gamma": gamma,
            "Mean (Min, Max)": f"{med:.4f} ({low:.4f}, {high:.4f})",
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def q4p2c(X_train, y_train, X_test, y_test, metrics, path="output/q4p2c.csv"):
    C_range = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    gamma_range = [1e-2, 1e-1, 1e0, 1e1]

    best_C, best_gamma = select_param_RBF(X_train, y_train, C_range, gamma_range, metric="auroc", k=5)

    clf = KernelRidge(
        alpha = 1/(2*best_C),
        kernel="rbf",
        gamma=best_gamma,
    )

    clf.fit(X_train, y_train)

    rows = []
    for metric in metrics:
        med, low, high = performance(clf, X_test, y_test, metric, bootstrap=True)
        rows.append({
            "C": best_C,
            "gamma": best_gamma,
            "Performance Measure": metric,
            "Median": f"{med:.4f}",
            "95% CI": f"({low:.4f}, {high:.4f})"
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    



def main():
    print(f"Using Seed = {seed}")
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data(debug=False)
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    ## Question 1d
    # q1d(X_train, feature_names, seed=seed)

    ## Question 2c
    # q2c(X_train, y_train, metrics, seed=seed)

    ## Question 2d
    # q2d(X_train, y_train, X_test, y_test, metrics, seed)

    ## Question 2e
    # q2e(X_train, y_train)

    ## Question 2f
    # q2f(X_train, y_train, feature_names)

    ## Question 3.2 (b)
    # q3p2b(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, metrics=metrics, seed=seed)

    ## Question 3.2 (d)
    # q3p2d(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed)

    ## Question 4.1 (b)
    # q4p1b(X_train, y_train, X_test, y_test, metrics, seed)

    ## Question 4.2 (b)
    # q4p2b(X_train, y_train, metric="auroc")

    ## Question 4.2 (c)
    # q4p2c(X_train, y_train, X_test, y_test, metrics)

    X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()

    flow = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("stdscale", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", random_state=seed)),
    ])

    candidates = {
        "clf__C": [0.01, 0.03, 0.1, 0.3],
        "clf__l1_ratio": [0, 1],
        "clf__class_weight": [{-1: 1, 1: 2}, {-1: 1, 1: 4}]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gsearch = GridSearchCV(estimator=flow, param_grid=candidates, scoring="roc_auc", cv=skf, n_jobs=-1)
    gsearch.fit(X_challenge, y_challenge)
    clf = gsearch.best_estimator_

    print(f"optimal params: {gsearch.best_params_} & best CV AUROC: {gsearch.best_score_:.4}")
    print(confusion_matrix(y_challenge, clf.predict(X_challenge)))

    y_score = clf.decision_function(X_heldout)
    y_pred = clf.predict(X_heldout).astype(int)
    helper.save_challenge_predictions(y_pred, y_score, 'DHRUVJWC')


if __name__ == "__main__":
    main()
