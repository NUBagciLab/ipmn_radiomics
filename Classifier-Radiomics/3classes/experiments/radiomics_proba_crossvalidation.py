import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from joblib import dump

# Read data
data = pd.read_csv('df_trainset_processed.csv')
# labels = pd.read_csv('df_all.csv')

X = data.drop(['label','patient'], axis=1).values
y = data['label'].values
patient_ids = data['patient'].values

# Define k for k-fold cross-validation
k = 10
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize arrays to store results and the best model
best_model = None
best_patient_ids = None
best_auc = -np.inf
acc_scores = []
auc_scores = []
prec_scores = []
rec_scores = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
probas = []

# Perform k-fold cross-validation
for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    patient_id_test = patient_ids[test_index]

    # Train XGBoost classifier
    # model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y)), random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3,max_depth=4, n_estimators=140,learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    probas.append(pd.DataFrame(y_proba, columns=['class'+str(i) for i in range(len(np.unique(y)))], index=patient_id_test))

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    # Update best model if current model is better
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_patient_ids = patient_id_test

    # Append scores
    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    auc_scores.append(auc)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))

    print(f'Fold {i+1} - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, AUC: {auc}')

# Print overall results
print('Accuracy: ', np.mean(acc_scores), '+-', np.std(acc_scores))
print('Precision: ', np.mean(prec_scores), '+-', np.std(prec_scores))
print('Recall: ', np.mean(rec_scores), '+-', np.std(rec_scores))
print('AUC: ', np.mean(auc_scores), '+-', np.std(auc_scores))

# Save the best model
dump(best_model, 'best_model.joblib')

# SHAP values and plot
plt.figure()
explainer = shap.Explainer(best_model)
shap_values = explainer(X)
shap.summary_plot(shap_values, feature_names=data.columns[2:], plot_type="bar")
plt.savefig('shap_plot.png')

# Save ROC curve
mean_tpr = np.mean(tprs, axis=0)
plt.figure()
plt.plot(mean_fpr, mean_tpr, label='Mean ROC')
plt.legend()
plt.savefig('ROC_curve.png')

# Save probability vectors to CSV
probas_df = pd.concat(probas)
probas_df.to_csv('probas.csv')

# print("Patient IDs for the best model:")
# print(best_patient_ids)
