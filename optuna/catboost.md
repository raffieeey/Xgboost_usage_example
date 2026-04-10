# Optuna + CatBoost Cheat Sheet

## Table of Contents

- [1. Suggest Methods (Hyperparameter Sampling)](#1-suggest-methods-hyperparameter-sampling)
- [2. When to Use `log=True`](#2-when-to-use-logtrue)
- [3. CatBoost Hyperparameter Search Space](#3-catboost-hyperparameter-search-space)
- [4. Full CatBoost + Optuna Example (Classification)](#4-full-catboost--optuna-example-classification)
- [5. Full CatBoost + Optuna Example (Regression)](#5-full-catboost--optuna-example-regression)
- [6. Pruning vs Early Stopping](#6-pruning-vs-early-stopping)
- [7. Study Configuration](#7-study-configuration)
- [8. Multi-Objective Optimization](#8-multi-objective-optimization)
- [9. Analyzing Results](#9-analyzing-results)
- [10. Visualization](#10-visualization)
- [11. Saving & Resuming Studies](#11-saving--resuming-studies)
- [12. Samplers](#12-samplers)
- [13. Conditional Hyperparameters](#13-conditional-hyperparameters)
- [14. Common Pitfalls](#14-common-pitfalls)
- [15. Recommended Default Tuning Recipe](#15-recommended-default-tuning-recipe)
- [16. Quick Reference Table](#16-quick-reference-table)

---

## 1. Suggest Methods (Hyperparameter Sampling)

```python
# Uniform float (continuous)
trial.suggest_float("dropout", 0.1, 0.5)

# Float with step
trial.suggest_float("momentum", 0.5, 0.99, step=0.01)

# Log-uniform float (spans orders of magnitude)
trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

# Integer
trial.suggest_int("depth", 3, 10)

# Integer with step
trial.suggest_int("batch_size", 16, 128, step=16)

# Log-scale integer
trial.suggest_int("n_estimators", 100, 10000, log=True)

# Categorical (pick from a list)
trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
```

---

## 2. When to Use `log=True`

Use `log=True` when a parameter spans **multiple orders of magnitude**. Without it, sampling is uniform and most values cluster near the top of the range.

| Parameter            | Range Example     | Use `log=True`? | Why                                      |
|----------------------|-------------------|-----------------|------------------------------------------|
| `learning_rate`      | 1e-5 → 1e-1      | Yes             | 0.001 vs 0.01 often matters as much as 0.01 vs 0.1 |
| `l2_leaf_reg`        | 1e-3 → 100       | Yes             | Spans 5 orders of magnitude              |
| `random_strength`    | 1e-3 → 10        | Yes             | Spans 4 orders of magnitude              |
| `bagging_temperature`| 0.0 → 10.0       | No              | Linear range, differences are proportional |
| `depth`              | 3 → 10           | No              | Small integer range                      |
| `iterations`         | 100 → 10000      | Yes             | 100 vs 1000 often matters more than 9000 vs 10000 |
| `border_count`       | 32 → 255         | No              | Roughly same order of magnitude          |

**Rule of thumb:** if `max / min > 100`, use `log=True`.

---

## 3. CatBoost Hyperparameter Search Space

### Core Parameters (Safe Defaults vs Wide Exploration)

| Parameter             | Safe Default Range        | Wide Exploration Range    | `log=True`? |
|-----------------------|---------------------------|---------------------------|-------------|
| `iterations`          | 500 – 2000                | 300 – 5000                | Yes         |
| `learning_rate`       | 0.01 – 0.2               | 1e-4 – 0.3               | Yes         |
| `depth`               | 4 – 8                    | 3 – 10                   | No          |
| `l2_leaf_reg`         | 1 – 10                   | 1e-3 – 100               | Yes         |
| `random_strength`     | 0.1 – 5                  | 1e-3 – 10                | Yes         |
| `bagging_temperature` | 0.0 – 5.0                | 0.0 – 10.0               | No          |
| `border_count`        | 64 – 255                 | 32 – 255                 | No          |
| `min_data_in_leaf`    | 1 – 30                   | 1 – 100                  | Yes         |

Start with the safe defaults for 50–100 trials. Switch to wide exploration if you have budget for 200+ trials or if safe defaults are plateauing.

```python
# Safe default search space
params = {
    "iterations":          trial.suggest_int("iterations", 500, 2000, log=True),
    "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    "depth":               trial.suggest_int("depth", 4, 8),
    "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
    "random_strength":     trial.suggest_float("random_strength", 0.1, 5, log=True),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    "border_count":        trial.suggest_int("border_count", 64, 255),
    "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 30, log=True),
}

# Wide exploration search space (more trials needed)
params = {
    "iterations":          trial.suggest_int("iterations", 300, 5000, log=True),
    "learning_rate":       trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
    "depth":               trial.suggest_int("depth", 3, 10),
    "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 100, log=True),
    "random_strength":     trial.suggest_float("random_strength", 1e-3, 10, log=True),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
    "border_count":        trial.suggest_int("border_count", 32, 255),
    "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 100, log=True),
}
```

### Additional / Advanced Parameters

```python
# RSM (column sampling ratio — like colsample_bytree in XGBoost)
params["rsm"] = trial.suggest_float("rsm", 0.5, 1.0)

# Bootstrap type — controls what bagging params are available (see Section 13)
# Do NOT set bagging_temperature or subsample unconditionally;
# gate them based on bootstrap_type.
params["bootstrap_type"] = trial.suggest_categorical("bootstrap_type",
                               ["Bayesian", "Bernoulli", "MVS"])

# Leaf estimation method
params["leaf_estimation_method"] = trial.suggest_categorical(
    "leaf_estimation_method", ["Newton", "Gradient"]
)

# Leaf estimation iterations
params["leaf_estimation_iterations"] = trial.suggest_int(
    "leaf_estimation_iterations", 1, 10
)

# Score function (for choosing splits)
params["score_function"] = trial.suggest_categorical(
    "score_function", ["Cosine", "L2"]
)

# Grow policy — min_data_in_leaf and max_leaves are conditional (see Section 13)
params["grow_policy"] = trial.suggest_categorical("grow_policy",
                            ["SymmetricTree", "Depthwise", "Lossguide"])
```

### What Each Parameter Does

| Parameter               | Effect                                                    |
|-------------------------|-----------------------------------------------------------|
| `iterations`            | Number of boosting rounds (trees)                         |
| `learning_rate`         | Step size shrinkage — lower = slower but more accurate    |
| `depth`                 | Max tree depth — higher = more complex, risk of overfit   |
| `l2_leaf_reg`           | L2 regularization on leaf values — higher = more regular  |
| `random_strength`       | Randomness of score splitting — higher = more random      |
| `bagging_temperature`   | Controls intensity of Bayesian bootstrap (only with `Bayesian` bootstrap) |
| `border_count`          | Number of splits for numerical features                   |
| `grow_policy`           | Tree growing strategy                                     |
| `min_data_in_leaf`      | Min samples in a leaf — higher = more conservative        |
| `subsample`             | Fraction of data used per tree (only with `Bernoulli`/`MVS` bootstrap) |
| `rsm`                   | Fraction of features used per tree                        |

---

## 4. Full CatBoost + Optuna Example (Classification)

```python
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cat_features = []  # indices of categorical columns, if any

def objective(trial):
    params = {
        "iterations":          trial.suggest_int("iterations", 500, 2000, log=True),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":               trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "random_strength":     trial.suggest_float("random_strength", 0.1, 5, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count":        trial.suggest_int("border_count", 64, 255),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 30, log=True),

        # Fixed params
        "loss_function":       "Logloss",
        "eval_metric":         "F1",
        "task_type":           "CPU",       # or "GPU"
        "verbose":             0,
        "random_seed":         42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        train_pool = Pool(X[train_idx], y[train_idx], cat_features=cat_features)
        val_pool   = Pool(X[val_idx],   y[val_idx],   cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=0,
        )

        preds = model.predict(X[val_idx])
        scores.append(f1_score(y[val_idx], preds))

    return np.mean(scores)


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best F1:     {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# After study: refit best params on full training data
# best_model = CatBoostClassifier(**study.best_params)
# best_model.fit(Pool(X_train_full, y_train_full, cat_features=cat_features))
```

> **CV + early stopping protocol:**
> When using early stopping inside CV folds, the validation fold within each split is used to decide when to stop boosting — this is fine. But keep a separate held-out test set (or use nested CV) for final evaluation. After the study, refit the best params on the full training set with a fresh validation split for early stopping.

> **Categorical features note:**
> CatBoost handles categorical features natively via `cat_features`. Pass column indices (or names) to `Pool()`. Do not one-hot encode columns you want CatBoost to handle — just declare them as categorical.

---

## 5. Full CatBoost + Optuna Example (Regression)

```python
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def objective(trial):
    params = {
        "iterations":          trial.suggest_int("iterations", 500, 2000, log=True),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":               trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "random_strength":     trial.suggest_float("random_strength", 0.1, 5, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count":        trial.suggest_int("border_count", 64, 255),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 30, log=True),
        "rsm":                 trial.suggest_float("rsm", 0.5, 1.0),

        "loss_function":       "RMSE",
        "eval_metric":         "RMSE",
        "verbose":             0,
        "random_seed":         42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        train_pool = Pool(X[train_idx], y[train_idx])
        val_pool   = Pool(X[val_idx],   y[val_idx])

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)

        preds = model.predict(X[val_idx])
        scores.append(mean_squared_error(y[val_idx], preds, squared=False))

    return np.mean(scores)  # minimize RMSE


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

---

## 6. Pruning vs Early Stopping

These are two different mechanisms that are often confused:

- **Early stopping** shortens a single model's training. It stops adding boosting rounds when the validation metric plateaus within one trial.
- **Pruning** stops entire Optuna trials. It kills an unpromising hyperparameter configuration before it finishes training, based on how it compares to other trials so far.

You should use both: early stopping inside each trial (to avoid wasting rounds), and pruning across trials (to avoid wasting entire runs).

### CatBoostPruningCallback

```python
from optuna.integration import CatBoostPruningCallback

def objective(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 500, 2000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":         trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "eval_metric":   "Logloss",
        "verbose":       0,
    }

    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val, y_val)

    # Pruning callback — stops bad *trials* across the study
    pruning_callback = CatBoostPruningCallback(trial, "Logloss")

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,   # stops bad *rounds* within this trial
        callbacks=[pruning_callback],
        verbose=0,
    )

    # Must call this to tell Optuna pruning is done
    pruning_callback.check_pruned()

    return model.get_best_score()["validation"]["Logloss"]


study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
)
study.optimize(objective, n_trials=200)
```

### Available Pruners

```python
optuna.pruners.MedianPruner(n_warmup_steps=10)       # Prune if below median of completed trials
optuna.pruners.PercentilePruner(25.0)                 # Prune if in bottom 25%
optuna.pruners.HyperbandPruner()                      # Successive halving
optuna.pruners.ThresholdPruner(upper=0.9)             # Prune if metric > threshold
optuna.pruners.NopPruner()                            # Disable pruning
```

---

## 7. Study Configuration

```python
# Maximize a metric (e.g., accuracy, F1)
study = optuna.create_study(direction="maximize")

# Minimize a metric (e.g., RMSE, Logloss)
study = optuna.create_study(direction="minimize")

# Run optimization
study.optimize(objective, n_trials=100)          # fixed number of trials
study.optimize(objective, timeout=3600)          # time limit in seconds
study.optimize(objective, n_trials=100, n_jobs=4)  # parallel trials

# Enqueue known good params to start with (warm-start)
study.enqueue_trial({
    "learning_rate": 0.03,
    "depth": 6,
    "iterations": 1000,
    "l2_leaf_reg": 3.0,
})
study.optimize(objective, n_trials=100)
```

---

## 8. Multi-Objective Optimization

Optimize multiple metrics simultaneously (e.g., accuracy vs training time).

Use multi-objective only when you truly care about a Pareto tradeoff (e.g., you need to deploy under latency constraints). Otherwise, optimizing a single metric and logging runtime separately is usually simpler and more interpretable.

```python
def objective(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":         trial.suggest_int("depth", 4, 8),
        "verbose":       0,
    }

    model = CatBoostClassifier(**params)

    import time
    start = time.time()
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
    train_time = time.time() - start

    accuracy = model.score(X_val, y_val)

    return accuracy, train_time  # maximize accuracy, minimize time


study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(objective, n_trials=100)

# Get Pareto-optimal trials
for trial in study.best_trials:
    print(f"Accuracy: {trial.values[0]:.4f}, Time: {trial.values[1]:.1f}s")
```

---

## 9. Analyzing Results

```python
# Best trial
print(study.best_value)
print(study.best_params)
print(study.best_trial)

# All trials as a DataFrame
df = study.trials_dataframe()
df.sort_values("value", ascending=False).head(10)

# Access individual trial info
for trial in study.trials:
    print(f"Trial {trial.number}: value={trial.value}, params={trial.params}")

# Check trial states
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print(f"Completed: {len(completed)}, Pruned: {len(pruned)}")
```

---

## 10. Visualization

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
    plot_edf,
)

# Optimization history (score over trials)
plot_optimization_history(study).show()

# Which params matter most
plot_param_importances(study).show()

# Parallel coordinate plot (see param interactions)
plot_parallel_coordinate(study).show()

# Contour plot (2D interaction between two params)
plot_contour(study, params=["learning_rate", "depth"]).show()

# Slice plot (each param vs objective)
plot_slice(study).show()

# Empirical distribution function
plot_edf(study).show()
```

---

## 11. Saving & Resuming Studies

```python
# Save to SQLite (persistent across runs)
study = optuna.create_study(
    study_name="catboost_tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True,  # resume if study already exists
)
study.optimize(objective, n_trials=50)

# Resume later — same code picks up where it left off
study = optuna.create_study(
    study_name="catboost_tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)  # runs 50 MORE trials

# Load a study without running more trials
study = optuna.load_study(
    study_name="catboost_tuning",
    storage="sqlite:///optuna_study.db",
)
print(study.best_params)
```

---

## 12. Samplers

```python
# TPE (default — usually best)
sampler = optuna.samplers.TPESampler(seed=42)

# Random search baseline
sampler = optuna.samplers.RandomSampler(seed=42)

# CMA-ES (good for continuous params)
sampler = optuna.samplers.CmaEsSampler(seed=42)

# Grid search (exhaustive, requires search_space)
sampler = optuna.samplers.GridSampler({
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
})

# GP (Gaussian Process — good for expensive objectives, few trials)
sampler = optuna.samplers.GPSampler(seed=42)

# Use a sampler
study = optuna.create_study(direction="maximize", sampler=sampler)
```

### Which Sampler to Use

| Scenario                          | Sampler       |
|-----------------------------------|---------------|
| General purpose (default)         | `TPESampler`  |
| Expensive objective, < 50 trials  | `GPSampler`   |
| All continuous params             | `CmaEsSampler`|
| Need reproducible baseline        | `RandomSampler`|
| Small discrete search space       | `GridSampler` |

---

## 13. Conditional Hyperparameters

Bootstrap-specific parameters, grow policy parameters, and other options that only apply under certain conditions must be gated with `if` statements. Never include them in a flat search dict unconditionally.

```python
def objective(trial):
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
    )

    params = {
        "bootstrap_type": bootstrap_type,
        "depth":          trial.suggest_int("depth", 4, 8),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "verbose":        0,
    }

    # Conditional params based on bootstrap type
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 5.0
        )
    elif bootstrap_type == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    elif bootstrap_type == "MVS":
        params["subsample"] = trial.suggest_float("subsample_mvs", 0.5, 1.0)

    # grow_policy-dependent params
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params["grow_policy"] = grow_policy

    if grow_policy in ["Depthwise", "Lossguide"]:
        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf", 1, 100, log=True
        )
    if grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

    # ... train and return metric
```

---

## 14. Common Pitfalls

**1. Forgetting `log=True` on learning rate and regularization**
```python
# BAD — 99% of samples will be between 0.01 and 0.1, ignoring 1e-5 to 1e-3
trial.suggest_float("lr", 1e-5, 0.1)

# GOOD
trial.suggest_float("lr", 1e-5, 0.1, log=True)
```

**2. Using the same parameter name twice**
```python
# BAD — Optuna will raise an error
trial.suggest_int("min_data_in_leaf", 1, 50)
trial.suggest_int("min_data_in_leaf", 1, 100)  # error: inconsistent range

# GOOD — use unique names for conditional params
trial.suggest_int("min_data_in_leaf_sym", 1, 50)
trial.suggest_int("min_data_in_leaf_loss", 1, 100)
```

**3. Not using early stopping**
```python
# BAD — wastes time on full iterations
model.fit(train_pool)

# GOOD — stops when validation metric plateaus
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
```

**4. Confusing early stopping with pruning**

Early stopping stops boosting rounds within one model. Pruning stops entire Optuna trials across the search. Use both — they are complementary, not alternatives.

**5. Including bootstrap/grow-policy params unconditionally**

`bagging_temperature` only applies when `bootstrap_type="Bayesian"`. `subsample` only applies with `Bernoulli` or `MVS`. `min_data_in_leaf` behaves differently across grow policies. Always gate these conditionally (see Section 13).

**6. Too few trials**

With very few trials (< 20), you will get limited benefit from adaptive search over random. 50–100 trials is a solid starting point for TPE. 200+ trials with pruning gives thorough exploration.

**7. Not setting `verbose=0`**

Your terminal will be flooded. Always set `verbose=0` in CatBoost params.

**8. Not seeding for reproducibility**
```python
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=42),
)
# Also set random_seed in CatBoost params
params["random_seed"] = 42
```

**9. Assuming GPU is always faster**

GPU acceleration (`task_type="GPU"`) helps on large datasets but can be slower on small ones due to data transfer overhead. Benchmark before committing.

```python
params["task_type"] = "GPU"
```

**10. Using `random_seed` vs `random_state`**

CatBoost uses `random_seed`, not `random_state` (which is the sklearn convention). `StratifiedKFold` uses `random_state`. Don't mix them up.

---

## 15. Recommended Default Tuning Recipe

An opinionated starting point for tabular classification or regression:

```python
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import numpy as np

# 1. Hold out a final test set BEFORE tuning
# X_train, X_test, y_train, y_test = train_test_split(...)

cat_features = []  # indices of categorical columns

# 2. Define objective with CV + early stopping
def objective(trial):
    params = {
        "iterations":          3000,  # high cap — early stopping will cut it
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth":               trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "random_strength":     trial.suggest_float("random_strength", 0.1, 5, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count":        trial.suggest_int("border_count", 64, 255),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 30, log=True),
        "verbose":             0,
        "random_seed":         42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        train_pool = Pool(X_train[train_idx], y_train[train_idx], cat_features=cat_features)
        val_pool   = Pool(X_train[val_idx],   y_train[val_idx],   cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=0,
        )

        preds = model.predict(X_train[val_idx])
        # Use AUC for imbalanced binary; F1 if thresholded performance matters
        scores.append(f1_score(y_train[val_idx], preds))

    return np.mean(scores)

# 3. Run study
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 4. Refit best params on full training data
best_params = study.best_params
best_params.update({
    "iterations": 3000,
    "verbose": 0,
    "random_seed": 42,
})

# Use a fresh validation split for early stopping during final refit
X_fit, X_es, y_fit, y_es = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
final_model = CatBoostClassifier(**best_params)
final_model.fit(
    Pool(X_fit, y_fit, cat_features=cat_features),
    eval_set=Pool(X_es, y_es, cat_features=cat_features),
    early_stopping_rounds=50,
    verbose=0,
)

# 5. Evaluate on held-out test set
test_preds = final_model.predict(X_test)
test_f1 = f1_score(y_test, test_preds)
print(f"Test F1: {test_f1:.4f}")
```

### Checklist

- Sampler: `TPESampler(seed=42)`
- Trials: 50–100 for safe ranges, 200+ for wide exploration
- CV: `StratifiedKFold(5)` for classification, `KFold(5)` for regression
- Metric: AUC for imbalanced binary, F1 if threshold matters, RMSE for regression
- Early stopping: always, `early_stopping_rounds=50`
- Categorical features: pass indices to `Pool()`, do not one-hot encode
- GPU: only if dataset is large enough to benefit (benchmark first)
- Final refit: refit best params on full training data with a fresh validation split
- Test set: never seen during tuning or training — evaluate only at the end

---

## 16. Quick Reference Table

| Task                          | Code                                                          |
|-------------------------------|---------------------------------------------------------------|
| Log-scale float               | `trial.suggest_float("lr", 1e-5, 0.1, log=True)`             |
| Uniform float                 | `trial.suggest_float("drop", 0.1, 0.5)`                      |
| Integer                       | `trial.suggest_int("depth", 3, 10)`                          |
| Log integer                   | `trial.suggest_int("iters", 100, 5000, log=True)`            |
| Categorical                   | `trial.suggest_categorical("opt", ["a", "b"])`               |
| Maximize                      | `create_study(direction="maximize")`                          |
| Minimize                      | `create_study(direction="minimize")`                          |
| Multi-objective               | `create_study(directions=["maximize", "minimize"])`           |
| Persist to DB                 | `create_study(storage="sqlite:///study.db")`                  |
| Resume study                  | `create_study(..., load_if_exists=True)`                      |
| Warm-start                    | `study.enqueue_trial({"lr": 0.03, ...})`                     |
| Param importance              | `plot_param_importances(study)`                               |
| Get best                      | `study.best_params`, `study.best_value`                       |
| All results as df             | `study.trials_dataframe()`                                    |
| Prune bad trials              | `CatBoostPruningCallback(trial, "Logloss")`                  |
| Parallel                      | `study.optimize(obj, n_trials=100, n_jobs=4)`                 |
| Silence logs                  | `optuna.logging.set_verbosity(optuna.logging.WARNING)`        |
| GPU training                  | `params["task_type"] = "GPU"`                                 |
 
