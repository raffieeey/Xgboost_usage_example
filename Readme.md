# XGBoost Python Usage Examples

A comprehensive collection of code snippets for XGBoost in Python.

---

## Table of Contents

- [Regression](#regression)
- [Train on GPU](#train-on-gpu)
- [Binary Classification](#binary-classification)
- [Multiclass Classification](#multiclass-classification)
- [Get the Best Result for Each Metric](#get-the-best-result-for-each-metric)
- [Get the Identifier of the Iteration with the Best Result](#get-the-identifier-of-the-iteration-with-the-best-result)
- [Load the Dataset from list, ndarray, pandas.DataFrame, pandas.Series](#load-the-dataset-from-list-ndarray-pandasdataframe-pandasseries)
- [Load the Dataset from a File](#load-the-dataset-from-a-file)
- [Load the Dataset from Sparse Python Data](#load-the-dataset-from-sparse-python-data)
- [Get a Slice of a DMatrix](#get-a-slice-of-a-dmatrix)
- [CV](#cv)
- [Using Sample Weights](#using-sample-weights)
- [Using Best Model](#using-best-model)
- [Load the Model from a File](#load-the-model-from-a-file)
- [Using staged_predict](#using-staged_predict)
- [Using Pre-training Results (Baseline)](#using-pre-training-results-baseline)
- [Training Continuation](#training-continuation)
- [Batch Training](#batch-training)
- [Exporting the Model](#exporting-the-model)
- [Feature Importance Calculation](#feature-importance-calculation)
- [User-defined Loss Function](#user-defined-loss-function)
- [User-defined Metric for Overfitting Detector and Best Model Selection](#user-defined-metric-for-overfitting-detector-and-best-model-selection)
- [SHAP Values](#shap-values)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Ranking](#ranking)
- [Categorical Features (Native Support)](#categorical-features-native-support)

---

## Regression

```python
from xgboost import XGBRegressor

# Initialize data
train_data = [[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]]
eval_data = [[2, 4, 6, 8], [1, 4, 50, 60]]
train_labels = [10, 20, 30]

# Initialize XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit model
model.fit(train_data, train_labels)

# Get predictions
predictions = model.predict(eval_data)
print(predictions)
```

---

## Train on GPU

```python
from xgboost import XGBClassifier

# Initialize data
train_data = [[0, 3], [4, 1], [8, 1], [9, 1]]
train_labels = [0, 0, 1, 1]

# Initialize classifier with GPU support
model = XGBClassifier(
    tree_method='hist',
    device='cuda',  # Use 'cuda' for GPU, 'cpu' for CPU
    n_estimators=100,
    learning_rate=0.1
)

# Fit model
model.fit(train_data, train_labels)
```

> **Note:** For XGBoost versions < 2.0, use `tree_method='gpu_hist'` instead.

---

## Binary Classification

```python
from xgboost import XGBClassifier

train_data = [[0, 3], [4, 1], [8, 1], [9, 1]]
train_labels = [0, 0, 1, 1]
eval_data = [[2, 1], [3, 1], [9, 0], [5, 3]]

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    objective='binary:logistic'
)

model.fit(train_data, train_labels)

# Get predicted classes
predictions = model.predict(eval_data)

# Get predicted probabilities
probabilities = model.predict_proba(eval_data)
print(predictions)
print(probabilities)
```

---

## Multiclass Classification

```python
from xgboost import XGBClassifier
import sklearn.datasets

iris = sklearn.datasets.load_iris()

model = XGBClassifier(
    n_estimators=100,
    objective='multi:softmax',
    num_class=3
)

model.fit(iris.data, iris.target)
predictions = model.predict(iris.data[:5])
print(predictions)
```

To get probabilities instead of class labels:

```python
model = XGBClassifier(
    n_estimators=100,
    objective='multi:softprob',
    num_class=3
)

model.fit(iris.data, iris.target)
probabilities = model.predict_proba(iris.data[:5])
print(probabilities)
```

---

## Get the Best Result for Each Metric

```python
import xgboost as xgb

train_data = [[0, 3], [4, 1], [8, 1], [9, 1]]
train_labels = [0, 0, 1, 1]
eval_data = [[2, 1], [3, 1], [9, 0], [5, 3]]
eval_labels = [0, 1, 1, 0]

dtrain = xgb.DMatrix(train_data, label=train_labels)
deval = xgb.DMatrix(eval_data, label=eval_labels)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc']
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (deval, 'eval')],
    evals_result=evals_result,
    verbose_eval=False
)

# Access results
print("Best AUC:", max(evals_result['eval']['auc']))
print("Best LogLoss:", min(evals_result['eval']['logloss']))
```

---

## Get the Identifier of the Iteration with the Best Result

```python
import xgboost as xgb

dtrain = xgb.DMatrix(train_data, label=train_labels)
deval = xgb.DMatrix(eval_data, label=eval_labels)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(deval, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=False
)

print("Best iteration:", model.best_iteration)
print("Best score:", model.best_score)
```

---

## Load the Dataset from list, ndarray, pandas.DataFrame, pandas.Series

**From Python list:**

```python
import xgboost as xgb

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
labels = [0, 1, 0]

dtrain = xgb.DMatrix(data, label=labels)
```

**From NumPy ndarray:**

```python
import xgboost as xgb
import numpy as np

data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, 100)

dtrain = xgb.DMatrix(data, label=labels)
```

**From pandas DataFrame:**

```python
import xgboost as xgb
import pandas as pd

data = pd.DataFrame({
    'feature_1': [1, 2, 3],
    'feature_2': [4, 5, 6]
})
labels = pd.Series([0, 1, 0])

dtrain = xgb.DMatrix(data, label=labels)
print(dtrain.feature_names)
```

---

## Load the Dataset from a File

**LibSVM format:**

```python
import xgboost as xgb

dtrain = xgb.DMatrix('train.libsvm')
```

**CSV format:**

```python
import xgboost as xgb

# label_column specifies which column is the label (0-indexed)
dtrain = xgb.DMatrix('train.csv?format=csv&label_column=0')
```

**Binary DMatrix format:**

```python
import xgboost as xgb

# Save
dtrain.save_binary('train.buffer')

# Load
dtrain_loaded = xgb.DMatrix('train.buffer')
```

---

## Load the Dataset from Sparse Python Data

```python
import xgboost as xgb
from scipy import sparse
import numpy as np

# Create sparse CSR matrix
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_matrix = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

labels = np.array([0, 1, 0])

dtrain = xgb.DMatrix(csr_matrix, label=labels)
print(f"Shape: ({dtrain.num_row()}, {dtrain.num_col()})")
```

---

## Get a Slice of a DMatrix

```python
import xgboost as xgb
import numpy as np

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
dtrain = xgb.DMatrix(X, label=y)

# Slice by row indices
indices = [0, 5, 10, 15, 20]
dslice = dtrain.slice(indices)

print(f"Original: ({dtrain.num_row()}, {dtrain.num_col()})")
print(f"Sliced: ({dslice.num_row()}, {dslice.num_col()})")
```

---

## CV

```python
import xgboost as xgb
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
dtrain = xgb.DMatrix(X, label=y)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3
}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    stratified=True,
    early_stopping_rounds=10,
    seed=42
)

print("Best AUC:", cv_results['test-auc-mean'].max())
print("Best iteration:", cv_results['test-auc-mean'].idxmax())
```

---

## Using Sample Weights

```python
import xgboost as xgb
import numpy as np

X = np.random.rand(100, 10)
y = np.array([0] * 90 + [1] * 10)  # Imbalanced

# Create weights (higher weight for minority class)
weights = np.ones(100)
weights[y == 1] = 9  # 9x weight for minority

dtrain = xgb.DMatrix(X, label=y, weight=weights)

params = {'objective': 'binary:logistic', 'max_depth': 3}
model = xgb.train(params, dtrain, num_boost_round=100)
```

**Alternative using `scale_pos_weight`:**

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=9,  # ratio of negative to positive samples
    n_estimators=100
)
model.fit(X, y)
```

---

## Using Best Model

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Best iteration:", model.best_iteration)
print("Best score:", model.best_score)
```

---

## Load the Model from a File

**Save and load model:**

```python
import xgboost as xgb

# Save model
model.save_model('model.json')  # JSON format (recommended)
model.save_model('model.ubj')   # Binary JSON format (faster)

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('model.json')
```

**Using sklearn API:**

```python
from xgboost import XGBClassifier

# Save
model.save_model('model.json')

# Load
loaded_model = XGBClassifier()
loaded_model.load_model('model.json')
```

**Using pickle:**

```python
import pickle

# Save (preserves all Python attributes)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

---

## Using staged_predict

```python
import xgboost as xgb

# Get predictions at specific iteration
predictions = model.predict(dtest, iteration_range=(0, 50))  # First 50 trees

# Get predictions at different stages
for n_trees in [10, 25, 50, 100]:
    pred = model.predict(dtest, iteration_range=(0, n_trees))
    print(f"Trees: {n_trees}, Predictions: {pred[:3]}")
```

---

## Using Pre-training Results (Baseline)

```python
import xgboost as xgb
import numpy as np

# Base predictions (log-odds for classification)
base_margin = np.array([0.5, -0.5, 0.3, -0.3])

dtrain = xgb.DMatrix(train_data, label=train_labels)
dtrain.set_base_margin(base_margin)

params = {'objective': 'binary:logistic', 'max_depth': 3}
model = xgb.train(params, dtrain, num_boost_round=100)
```

---

## Training Continuation

```python
import xgboost as xgb

# Initial training
model = xgb.train(params, dtrain, num_boost_round=50)
print("Trees after initial training:", model.num_boosted_rounds())

# Continue training
model_continued = xgb.train(
    params,
    dtrain,
    num_boost_round=50,
    xgb_model=model  # Pass existing model
)
print("Trees after continuation:", model_continued.num_boosted_rounds())
```

**Continue from saved file:**

```python
model.save_model('checkpoint.json')

model_continued = xgb.train(
    params,
    dtrain,
    num_boost_round=50,
    xgb_model='checkpoint.json'
)
```

---

## Batch Training

```python
import xgboost as xgb

model = None

for batch_idx in range(5):
    # Get batch data
    X_batch, y_batch = get_batch(batch_idx)  # Your data loading function
    dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
    
    # Train/continue training
    model = xgb.train(
        params,
        dtrain_batch,
        num_boost_round=10,
        xgb_model=model
    )
    print(f"Batch {batch_idx + 1}: Total trees = {model.num_boosted_rounds()}")
```

---

## Exporting the Model

**Export to JSON:**

```python
model.save_model('model.json')

# Get model dump as text
dump = model.get_dump(dump_format='text')
print(dump[0])  # First tree

# Get model dump as JSON
dump_json = model.get_dump(dump_format='json')
```

**Export to ONNX:**

```python
from onnxmltools import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 10]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

---

## Feature Importance Calculation

```python
import xgboost as xgb

# Get importance by different types
importance_weight = model.get_score(importance_type='weight')   # Frequency
importance_gain = model.get_score(importance_type='gain')       # Average gain
importance_cover = model.get_score(importance_type='cover')     # Average coverage

print("Top features by gain:")
for feat, score in sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feat}: {score:.2f}")
```

**Using sklearn API:**

```python
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100)
model.fit(X, y)

# feature_importances_ uses 'gain' by default
importances = model.feature_importances_
print(importances)
```

**Plot feature importance:**

```python
import xgboost as xgb

xgb.plot_importance(model, importance_type='gain', max_num_features=10)
```

---

## User-defined Loss Function

```python
import xgboost as xgb
import numpy as np

def custom_huber_loss(predt, dtrain, delta=1.0):
    """Custom Huber Loss objective function."""
    y = dtrain.get_label()
    residual = predt - y
    
    # Gradient
    grad = np.where(
        np.abs(residual) <= delta,
        residual,
        delta * np.sign(residual)
    )
    
    # Hessian
    hess = np.where(
        np.abs(residual) <= delta,
        np.ones_like(residual),
        np.zeros_like(residual)
    )
    hess = np.maximum(hess, 1e-6)  # Numerical stability
    
    return grad, hess

params = {'max_depth': 3, 'learning_rate': 0.1}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=custom_huber_loss
)
```

---

## User-defined Metric for Overfitting Detector and Best Model Selection

```python
import xgboost as xgb
import numpy as np

def custom_f1_metric(predt, dtrain):
    """Custom F1 Score metric."""
    y = dtrain.get_label()
    pred_labels = (predt > 0.5).astype(int)
    
    tp = np.sum((pred_labels == 1) & (y == 1))
    fp = np.sum((pred_labels == 1) & (y == 0))
    fn = np.sum((pred_labels == 0) & (y == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return 'f1', f1

params = {
    'objective': 'binary:logistic',
    'disable_default_eval_metric': 1
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(deval, 'eval')],
    custom_metric=custom_f1_metric,
    early_stopping_rounds=10
)
```

---

## SHAP Values

**Using XGBoost built-in:**

```python
import xgboost as xgb

# Get SHAP values (contributions)
shap_values = model.predict(dtest, pred_contribs=True)
print("SHAP values shape:", shap_values.shape)  # (n_samples, n_features + 1)

# Get SHAP interaction values
shap_interactions = model.predict(dtest, pred_interactions=True)
```

**Using SHAP library:**

```python
import shap
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Single prediction explanation
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0]
))
```

---

## Hyperparameter Tuning

**Grid Search:**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
}

model = XGBClassifier(objective='binary:logistic')

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

**Optuna:**

```python
import optuna
import xgboost as xgb

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    cv_results = xgb.cv(
        params, dtrain,
        num_boost_round=100,
        nfold=5,
        metrics='auc',
        early_stopping_rounds=10,
        seed=42
    )
    
    return cv_results['test-auc-mean'].max()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)
```

---

## Ranking

```python
import xgboost as xgb
import numpy as np

# Ranking data: documents grouped by queries
X = np.random.rand(100, 10)  # 100 documents, 10 features
y = np.random.randint(0, 5, 100)  # Relevance labels (0-4)
groups = [10] * 10  # 10 queries, 10 docs each

dtrain = xgb.DMatrix(X, label=y)
dtrain.set_group(groups)

params = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg',
    'max_depth': 4
}

model = xgb.train(params, dtrain, num_boost_round=100)
```

**Using sklearn API:**

```python
from xgboost import XGBRanker

ranker = XGBRanker(
    objective='rank:ndcg',
    n_estimators=100,
    max_depth=4
)

ranker.fit(X_train, y_train, group=groups_train)
scores = ranker.predict(X_test)
```

---

## Categorical Features (Native Support)

> **Note:** Native categorical support was added in XGBoost 1.5. Requires `tree_method='hist'` or `device='cuda'`.

**Using sklearn API (recommended):**

```python
import pandas as pd
from xgboost import XGBClassifier

# Create DataFrame with categorical columns
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'price': [10.0, 15.0, 20.0, 12.0, 8.0],
    'label': [0, 1, 1, 0, 0]
})

# Convert to 'category' dtype
categorical_columns = ['color', 'size']
for col in categorical_columns:
    data[col] = data[col].astype('category')

X = data.drop('label', axis=1)
y = data['label']

# Enable categorical support
model = XGBClassifier(
    enable_categorical=True,
    tree_method='hist'
)
model.fit(X, y)

# Prediction (ensure same category dtype)
new_data = pd.DataFrame({
    'color': pd.Categorical(['green', 'red']),
    'size': pd.Categorical(['medium', 'large']),
    'price': [18.0, 25.0]
})
predictions = model.predict(new_data)
```

**Using native API with DMatrix:**

```python
import xgboost as xgb
import pandas as pd

# Prepare categorical DataFrame
df = pd.DataFrame({
    'cat_feature': pd.Categorical(['a', 'b', 'c', 'a', 'b']),
    'num_feature': [1.0, 2.0, 3.0, 4.0, 5.0]
})
labels = [0, 1, 1, 0, 1]

# Create DMatrix with enable_categorical
dtrain = xgb.DMatrix(df, label=labels, enable_categorical=True)

params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'max_depth': 3
}

model = xgb.train(params, dtrain, num_boost_round=100)
```

**Specifying feature types manually:**

```python
import xgboost as xgb
import numpy as np

# When using arrays, specify feature_types
X = np.array([[0, 1.5], [1, 2.5], [2, 3.5], [0, 4.5]])  # First col is categorical (encoded)
y = [0, 1, 1, 0]

dtrain = xgb.DMatrix(X, label=y, feature_types=['c', 'q'])  # 'c' = categorical, 'q' = quantitative
dtrain.set_info(enable_categorical=True)

# Or with sklearn API
from xgboost import XGBClassifier

model = XGBClassifier(
    feature_types=['c', 'q'],
    enable_categorical=True,
    tree_method='hist'
)
```

**Controlling categorical split strategy:**

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    enable_categorical=True,
    tree_method='hist',
    max_cat_to_onehot=4,  # Use one-hot encoding if categories <= 4, else partition
    max_cat_threshold=64  # Max categories for partition-based split
)
```

**With mixed numeric and categorical features:**

```python
import pandas as pd
from xgboost import XGBRegressor

df = pd.DataFrame({
    'city': pd.Categorical(['NYC', 'LA', 'Chicago', 'NYC', 'LA']),
    'bedrooms': [2, 3, 1, 4, 2],
    'sqft': [1000, 1500, 800, 2000, 1200],
    'price': [500000, 700000, 300000, 900000, 600000]
})

X = df.drop('price', axis=1)
y = df['price']

model = XGBRegressor(
    enable_categorical=True,
    tree_method='hist',
    n_estimators=100
)
model.fit(X, y)
```

---

## Requirements

```bash
pip install xgboost numpy pandas scipy scikit-learn matplotlib shap optuna
```

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Python API Reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
