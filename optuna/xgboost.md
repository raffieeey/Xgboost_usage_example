# Optuna + XGBoost Cheat Sheet

## Table of Contents

- [1. Suggest Methods (Hyperparameter Sampling)](#1-suggest-methods-hyperparameter-sampling)
- [2. When to Use `log=True`](#2-when-to-use-logtrue)
- [3. XGBoost Hyperparameter Search Space](#3-xgboost-hyperparameter-search-space)
- [4. Full XGBoost + Optuna Example (Classification)](#4-full-xgboost--optuna-example-classification)
- [5. Full XGBoost + Optuna Example (Regression)](#5-full-xgboost--optuna-example-regression)
- [6. Pruning vs Early Stopping](#6-pruning-vs-early-stopping)
- [7. Study Configuration](#7-study-configuration)
- [8. Multi-Objective Optimization](#8-multi-objective-optimization)
- [9. Analyzing Results](#9-analyzing-results)
- [10. Visualization](#10-visualization)
- [11. Saving & Resuming Studies](#11-saving--resuming-studies)
- [12. Samplers](#12-samplers)
- [13. Conditional Hyperparameters](#13-conditional-hyperparameters)
- [14. XGBoost Sklearn API vs Native API](#14-xgboost-sklearn-api-vs-native-api)
- [15. Common Pitfalls](#15-common-pitfalls)
- [16. Recommended Default Tuning Recipe](#16-recommended-default-tuning-recipe)
- [17. Quick Reference Table](#17-quick-reference-table)

---

## 1. Suggest Methods (Hyperparameter Sampling)

```python
# Uniform float (continuous)
trial.suggest_float("colsample_bytree", 0.5, 1.0)

# Float with step
trial.suggest_float("subsample", 0.5, 1.0, step=0.05)

# Log-uniform float (spans orders of magnitude)
trial.suggest_float("learning_rate", 1e-5, 0.3, log=True)

# Integer
trial.suggest_int("max_depth", 3, 12)

# Integer with step
trial.suggest_int("batch_size", 16, 128, step=16)

# Log-scale integer
trial.suggest_int("n_estimators", 100, 10000, log=True)

# Categorical (pick from a list)
trial.suggest_categorical("booster", ["gbtree", "dart"])
```

---

## 2. When to Use `log=True`

Use `log=True` when a parameter spans **multiple orders of magnitude**. Without it, sampling is uniform and most values cluster near the top of the range.

| Parameter              | Range Example     | Use `log=True`? | Why                                       |
|------------------------|-------------------|-----------------|-------------------------------------------|
| `learning_rate`        | 1e-5 → 0.3       | Yes             | 0.001 vs 0.01 often matters as much as 0.01 vs 0.1 |
| `reg_alpha`            | 1e-8 → 100       | Yes             | Spans 10 orders of magnitude              |
| `reg_lambda`           | 1e-8 → 100       | Yes             | Spans 10 orders of magnitude              |
| `min_child_weight`     | 1e-3 → 100       | Yes             | Spans 5 orders of magnitude               |
| `gamma`                | 1e-8 → 10        | Yes             | Spans 9 orders of magnitude               |
| `subsample`            | 0.5 → 1.0        | No              | Small linear range                        |
| `colsample_bytree`     | 0.5 → 1.0        | No              | Small linear range                        |
| `max_depth`            | 3 → 12           | No              | Small integer range                       |
| `n_estimators`         | 100 → 10000      | Yes             | 100 vs 1000 often matters more than 9000 vs 10000 |

**Rule of thumb:** if `max / min > 100`, use `log=True`.

---

## 3. XGBoost Hyperparameter Search Space

### Core Parameters (Safe Defaults vs Wide Exploration)

| Parameter           | Safe Default Range        | Wide Exploration Range    | `log=True`? |
|---------------------|---------------------------|---------------------------|-------------|
| `n_estimators`      | 300 – 2000                | 100 – 5000                | Yes         |
| `learning_rate`     | 0.01 – 0.2               | 1e-4 – 0.3               | Yes         |
| `max_depth`         | 4 – 8                    | 3 – 12                   | No          |
| `min_child_weight`  | 1 – 20                   | 1e-3 – 100               | Yes         |
| `gamma`             | 0 – 5                    | 1e-8 – 10                | Yes (wide)  |
| `subsample`         | 0.6 – 1.0                | 0.5 – 1.0                | No          |
| `colsample_bytree`  | 0.6 – 1.0                | 0.5 – 1.0                | No          |
| `reg_alpha`         | 1e-3 – 10                | 1e-8 – 100               | Yes         |
| `reg_lambda`        | 1e-3 – 10                | 1e-8 – 100               | Yes         |

Start with the safe defaults for 50–100 trials. Switch to wide exploration if you have budget for 200+ trials or if safe defaults are plateauing.

```python
# Safe default search space
params = {
    "n_estimators":      trial.suggest_int("n_estimators", 300, 2000, log=True),
    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    "max_depth":         trial.suggest_int("max_depth", 4, 8),
    "min_child_weight":  trial.suggest_float("min_child_weight", 1, 20, log=True),
    "gamma":             trial.suggest_float("gamma", 0.01, 5.0, log=True),
    "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
}

# Wide exploration search space (more trials needed)
params = {
    "n_estimators":      trial.suggest_int("n_estimators", 100, 5000, log=True),
    "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
    "max_depth":         trial.suggest_int("max_depth", 3, 12),
    "min_child_weight":  trial.suggest_float("min_child_weight", 1e-3, 100, log=True),
    "gamma":             trial.suggest_float("gamma", 1e-8, 10, log=True),
    "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
    "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
}
```

### Additional / Advanced Parameters

```python
# Column sampling per level and per node
params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
params["colsample_bynode"]  = trial.suggest_float("colsample_bynode", 0.5, 1.0)

# Max leaves (only when grow_policy="lossguide")
params["max_leaves"] = trial.suggest_int("max_leaves", 0, 256)

# Max bin (histogram-based splits)
params["max_bin"] = trial.suggest_int("max_bin", 128, 512, step=32)

# Booster type
params["booster"] = trial.suggest_categorical("booster", ["gbtree", "dart"])

# Tree method
params["tree_method"] = trial.suggest_categorical(
    "tree_method", ["auto", "hist", "approx"]
)

# Grow policy
params["grow_policy"] = trial.suggest_categorical(
    "grow_policy", ["depthwise", "lossguide"]
)

# Scale pos weight (for imbalanced classification)
params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 10.0)

# DART-specific params (only when booster="dart" — see Section 13)
# Do NOT include these in a flat search dict; gate them conditionally.
# params["sample_type"]    = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
# params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
# params["rate_drop"]      = trial.suggest_float("rate_drop", 0.0, 0.5)
# params["skip_drop"]      = trial.suggest_float("skip_drop", 0.0, 0.5)
```

### What Each Parameter Does

| Parameter            | Effect                                                         |
|----------------------|----------------------------------------------------------------|
| `n_estimators`       | Number of boosting rounds (trees)                              |
| `learning_rate`      | Step size shrinkage — lower = slower but more accurate         |
| `max_depth`          | Max tree depth — higher = more complex, risk of overfit        |
| `min_child_weight`   | Min sum of instance weight in a child — higher = conservative  |
| `gamma`              | Min loss reduction to make a split — higher = fewer splits     |
| `subsample`          | Fraction of rows sampled per tree                              |
| `colsample_bytree`   | Fraction of features sampled per tree                          |
| `colsample_bylevel`  | Fraction of features sampled per depth level                   |
| `colsample_bynode`   | Fraction of features sampled per split                         |
| `reg_alpha`          | L1 regularization on leaf weights (sparsity)                   |
| `reg_lambda`         | L2 regularization on leaf weights (smoothing)                  |
| `max_leaves`         | Max leaf nodes (only for `lossguide` grow policy)              |
| `scale_pos_weight`   | Balances positive/negative weights for imbalanced data         |
| `booster`            | `gbtree` (trees), `dart` (dropout trees), `gblinear` (linear) |

---

## 4. Full XGBoost + Optuna Example (Classification)

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000, log=True),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma":             trial.suggest_float("gamma", 0.01, 5.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),

        # Fixed params
        "objective":         "binary:logistic",
        "eval_metric":       "logloss",
        "tree_method":       "hist",
        "device":            "cpu",         # or "cuda" for GPU
        "verbosity":         0,
        "random_state":      42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
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
# best_model = xgb.XGBClassifier(**study.best_params)
# best_model.fit(X_train_full, y_train_full)
```

> **CV + early stopping protocol:**
> When using early stopping inside CV folds, the validation fold within each split is used to decide when to stop boosting — this is fine. But keep a separate held-out test set (or use nested CV) for final evaluation. After the study, refit the best params on the full training set with a fresh validation split for early stopping.

> **`enable_categorical` note:**
> Set `enable_categorical=True` only when your categorical columns are correctly typed as `pandas.Categorical` and your XGBoost version and tree method support it. Do not set it blindly on NumPy arrays or incorrectly encoded data.

---

## 5. Full XGBoost + Optuna Example (Regression)

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 2000, log=True),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma":             trial.suggest_float("gamma", 0.01, 5.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),

        "objective":         "reg:squarederror",
        "eval_metric":       "rmse",
        "tree_method":       "hist",
        "verbosity":         0,
        "random_state":      42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )

        preds = model.predict(X[val_idx])
        scores.append(mean_squared_error(y[val_idx], preds, squared=False))

    return np.mean(scores)


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

### XGBoostPruningCallback (Native API)

```python
from optuna.integration import XGBoostPruningCallback

def objective(trial):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma":            trial.suggest_float("gamma", 0.01, 5.0, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "objective":        "binary:logistic",
        "eval_metric":      "logloss",
        "verbosity":        0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)

    # Pruning callback — stops bad *trials* across the study
    pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, "validation")],
        early_stopping_rounds=50,   # stops bad *rounds* within this trial
        callbacks=[pruning_callback],
        verbose_eval=False,
    )

    preds = bst.predict(dval)
    return float(np.mean((preds > 0.5) == y_val))


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
)
study.optimize(objective, n_trials=200)
```

### Sklearn API with Early Stopping Only (simpler, no pruning)

```python
def objective(trial):
    params = {
        "n_estimators":     3000,  # high cap, early stopping will cut it
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "early_stopping_rounds": 50,
        "verbosity":        0,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model.score(X_val, y_val)
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

# Minimize a metric (e.g., RMSE, logloss)
study = optuna.create_study(direction="minimize")

# Run optimization
study.optimize(objective, n_trials=100)              # fixed trial count
study.optimize(objective, timeout=3600)              # time limit in seconds
study.optimize(objective, n_trials=100, n_jobs=4)    # parallel trials

# Enqueue known good params to start with (warm-start)
study.enqueue_trial({
    "learning_rate": 0.03,
    "max_depth": 6,
    "n_estimators": 1000,
    "reg_lambda": 1.0,
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
        "n_estimators":  trial.suggest_int("n_estimators", 100, 3000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":     trial.suggest_int("max_depth", 4, 8),
        "verbosity":     0,
    }

    import time
    model = xgb.XGBClassifier(**params)

    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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
plot_contour(study, params=["learning_rate", "max_depth"]).show()

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
    study_name="xgboost_tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)

# Resume later — same code picks up where it left off
study = optuna.create_study(
    study_name="xgboost_tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)  # runs 50 MORE trials

# Load without running more trials
study = optuna.load_study(
    study_name="xgboost_tuning",
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
    "max_depth": [4, 6, 8, 10],
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

DART-specific parameters, grow policy parameters, and other options that only apply under certain conditions must be gated with `if` statements. Never include them in a flat search dict unconditionally.

```python
def objective(trial):
    booster = trial.suggest_categorical("booster", ["gbtree", "dart"])

    params = {
        "booster":        booster,
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "reg_alpha":      trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":     trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "objective":      "binary:logistic",
        "verbosity":      0,
    }

    # gbtree and dart share tree params
    if booster in ["gbtree", "dart"]:
        params["max_depth"]        = trial.suggest_int("max_depth", 4, 8)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 1, 20, log=True)
        params["gamma"]            = trial.suggest_float("gamma", 0.01, 5.0, log=True)
        params["subsample"]        = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)

    # DART-specific dropout params — only when booster="dart"
    if booster == "dart":
        params["sample_type"]    = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"]      = trial.suggest_float("rate_drop", 0.0, 0.5)
        params["skip_drop"]      = trial.suggest_float("skip_drop", 0.0, 0.5)

    # Grow policy — max_leaves only applies to lossguide
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params["grow_policy"] = grow_policy

    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256)

    # ... train and return metric
```

---

## 14. XGBoost Sklearn API vs Native API

### Sklearn API (simpler, recommended for most use cases)

```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    early_stopping_rounds=50,
    verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
preds = model.predict(X_val)
acc   = model.score(X_val, y_val)
```

### Native API (more control, needed for pruning callback)

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)

params = {
    "learning_rate": 0.03,
    "max_depth": 6,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

preds = bst.predict(dval)
```

### Key Differences

| Feature                | Sklearn API              | Native API                    |
|------------------------|--------------------------|-------------------------------|
| Iterations param       | `n_estimators`           | `num_boost_round`             |
| Early stopping         | `early_stopping_rounds=` | `early_stopping_rounds=`      |
| Data format            | numpy / pandas           | `xgb.DMatrix`                 |
| Pruning callback       | Not supported natively   | `XGBoostPruningCallback`      |
| Feature importance     | `model.feature_importances_` | `bst.get_score()`        |
| `.predict()` output    | Labels (0/1)             | Probabilities                 |

---

## 15. Common Pitfalls

**1. Forgetting `log=True` on learning rate and regularization**
```python
# BAD — 99% of samples will be in [0.01, 0.1], ignoring small values
trial.suggest_float("learning_rate", 1e-5, 0.1)
trial.suggest_float("reg_alpha", 1e-8, 100)

# GOOD
trial.suggest_float("learning_rate", 1e-5, 0.1, log=True)
trial.suggest_float("reg_alpha", 1e-8, 100, log=True)
```

**2. Using the same parameter name twice**
```python
# BAD — Optuna will raise an error
trial.suggest_int("max_depth", 3, 10)
trial.suggest_int("max_depth", 3, 12)  # error: inconsistent range

# GOOD — use unique names for conditional params
trial.suggest_int("max_depth_gbtree", 3, 10)
trial.suggest_int("max_depth_dart", 3, 12)
```

**3. Not using early stopping**
```python
# BAD — wastes time training full n_estimators
model = xgb.XGBClassifier(n_estimators=5000)
model.fit(X_train, y_train)

# GOOD — stops when validation metric plateaus
model = xgb.XGBClassifier(n_estimators=5000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

**4. Confusing early stopping with pruning**

Early stopping stops boosting rounds within one model. Pruning stops entire Optuna trials across the search. Use both — they are complementary, not alternatives.

**5. Mixing up native vs sklearn early stopping**
```python
# Native API — early_stopping_rounds is a kwarg of xgb.train()
bst = xgb.train(params, dtrain, num_boost_round=5000,
                 evals=[(dval, "val")], early_stopping_rounds=50)

# Sklearn API — early_stopping_rounds is a constructor param (XGBoost >= 2.0)
model = xgb.XGBClassifier(n_estimators=5000, early_stopping_rounds=50)
```

**6. Too few trials**

With very few trials (< 20), you will get limited benefit from adaptive search over random. 50–100 trials is a solid starting point for TPE. 200+ trials with pruning gives thorough exploration.

**7. Not setting verbosity**
```python
# Silence XGBoost
params["verbosity"] = 0

# Silence Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

**8. Not seeding for reproducibility**
```python
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=42),
)
# Also set random_state in XGBoost params
params["random_state"] = 42
```

**9. Assuming GPU is always faster**

GPU acceleration (`device="cuda"` or `tree_method="gpu_hist"` on older versions) helps on large datasets but can actually be slower on small ones due to data transfer overhead. Benchmark before committing.

```python
params["device"] = "cuda"       # XGBoost >= 2.0
# or
params["tree_method"] = "gpu_hist"  # older XGBoost versions
```

**10. Including DART params unconditionally**

DART parameters (`rate_drop`, `skip_drop`, `sample_type`, `normalize_type`) only apply when `booster="dart"`. Including them in a flat search dict wastes trials and can cause unexpected behavior. Always gate them conditionally (see Section 13).

---

## 16. Recommended Default Tuning Recipe

An opinionated starting point for tabular classification or regression:

```python
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

# 1. Hold out a final test set BEFORE tuning
# X_train, X_test, y_train, y_test = train_test_split(...)

# 2. Define objective with CV + early stopping
def objective(trial):
    params = {
        "n_estimators":      3000,  # high cap — early stopping will cut it
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma":             trial.suggest_float("gamma", 0.01, 5.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "tree_method":       "hist",
        "verbosity":         0,
        "random_state":      42,
        "early_stopping_rounds": 50,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train[train_idx], y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        # Use AUC for imbalanced binary; F1 if thresholded performance matters
        scores.append(model.score(X_train[val_idx], y_train[val_idx]))

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
    "n_estimators": 3000,
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": 42,
    "early_stopping_rounds": 50,
})

# Use a fresh validation split for early stopping during final refit
X_fit, X_es, y_fit, y_es = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

# 5. Evaluate on held-out test set
test_score = final_model.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Checklist

- Sampler: `TPESampler(seed=42)`
- Trials: 50–100 for safe ranges, 200+ for wide exploration
- CV: `StratifiedKFold(5)` for classification, `KFold(5)` for regression
- Metric: AUC for imbalanced binary, F1 if threshold matters, RMSE for regression
- Early stopping: always, `early_stopping_rounds=50`
- Tree method: start with `hist`
- GPU: only if dataset is large enough to benefit (benchmark first)
- Final refit: refit best params on full training data with a fresh validation split
- Test set: never seen during tuning or training — evaluate only at the end

---

## 17. Quick Reference Table

| Task                          | Code                                                          |
|-------------------------------|---------------------------------------------------------------|
| Log-scale float               | `trial.suggest_float("lr", 1e-5, 0.3, log=True)`             |
| Uniform float                 | `trial.suggest_float("sub", 0.5, 1.0)`                       |
| Integer                       | `trial.suggest_int("depth", 3, 12)`                          |
| Log integer                   | `trial.suggest_int("iters", 100, 5000, log=True)`            |
| Categorical                   | `trial.suggest_categorical("booster", ["gbtree", "dart"])`   |
| Maximize                      | `create_study(direction="maximize")`                          |
| Minimize                      | `create_study(direction="minimize")`                          |
| Multi-objective               | `create_study(directions=["maximize", "minimize"])`           |
| Persist to DB                 | `create_study(storage="sqlite:///study.db")`                  |
| Resume study                  | `create_study(..., load_if_exists=True)`                      |
| Warm-start                    | `study.enqueue_trial({"lr": 0.03, ...})`                     |
| Param importance              | `plot_param_importances(study)`                               |
| Get best                      | `study.best_params`, `study.best_value`                       |
| All results as df             | `study.trials_dataframe()`                                    |
| Prune bad trials              | `XGBoostPruningCallback(trial, "validation-logloss")`         |
| Parallel                      | `study.optimize(obj, n_trials=100, n_jobs=4)`                 |
| Silence logs                  | `optuna.logging.set_verbosity(optuna.logging.WARNING)`        |
| GPU training                  | `params["device"] = "cuda"`                                   |
