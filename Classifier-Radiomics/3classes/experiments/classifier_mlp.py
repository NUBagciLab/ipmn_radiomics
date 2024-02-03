import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('uvscaling_t.csv')
data.drop('patient', axis=1, inplace=True)
# data.drop('DL', axis=1, inplace=True)
X = data.drop('label', axis=1).values
y = data['label'].values

# Define weights for each class
weights = np.ones_like(y)

# Define 10-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize arrays to store results
acc_scores = np.zeros(10)
prec_scores = np.zeros((10, 3))
rec_scores = np.zeros((10, 3))
mean_fpr = np.linspace(0, 1, 100)
tprs = []
auc_scores = []

# Loop over folds
for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    print("Fold ", i+1)

    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w_train, w_test = weights[train_index], weights[test_index]

    # Apply MLP to training data
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # Fit MLP model on training data and get embeddings
    mlp.fit(X_train, y_train)
    X_train_mlp = mlp.predict(X_train)
    X_test_mlp = mlp.predict(X_test)

    # Combine embeddings with original features
    X_train_combined = np.concatenate((X_train, X_train_mlp.reshape(-1, 1)), axis=1)
    X_test_combined = np.concatenate((X_test, X_test_mlp.reshape(-1, 1)), axis=1)

    # Define XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        max_depth=4,
        n_estimators=140,
        learning_rate=0.1,
        random_state=42
    )

    # Fit model on combined data
    model.fit(X_train_combined, y_train, sample_weight=w_train)

    # Predict on test data
    y_pred = model.predict(X_test_combined)

    # Calculate evaluation metrics
    acc_scores[i] = accuracy_score(y_test, y_pred)
    prec_scores[i] = precision_score(y_test, y_pred, average=None, zero_division=1)
    rec_scores[i] = recall_score(y_test, y_pred, average=None, zero_division=1)

    print(f'accuracy={acc_scores[i]}')
    print(f'precision={prec_scores[i]}')
    print(f'recall={rec_scores[i]}')

    # Calculate ROC curve and AUC score
    y_proba = model.predict_proba(X_test_combined)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    print('AUC score: ', roc_auc)

# # Print overall results
print('Accuracy: ', np.mean(acc_scores))
print('Accuracy STD: ',np.std(acc_scores))
print('Precision: ', np.mean(prec_scores, axis=0))
print('Recall: ', np.mean(rec_scores, axis=0))
print('AUC scores:',np.mean(auc_scores), '+-', np.std(auc_scores))