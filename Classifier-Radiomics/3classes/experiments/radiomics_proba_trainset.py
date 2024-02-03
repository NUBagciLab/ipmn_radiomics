import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
# 1. Read file
df = pd.read_csv('df_trainset_processed.csv')
df_test = pd.read_csv('df_testset_processed.csv')

# 2. Preprocess data
X = df.drop(['patient', 'label'], axis=1)  # Features
y = df['label']  # Target

X_test = df_test.drop(['patient', 'label'], axis=1)
y_test = df_test['label']

# 3. Train the XGBoost model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3,max_depth=4, n_estimators=140,learning_rate=0.1, random_state=42)  # For multiclass classification
eval_set = [(X, y), (X_test, y_test)]
model.fit(X, y, eval_metric=["mlogloss", "merror"], eval_set=eval_set, verbose=True)

# 4. Store the trained model
joblib.dump(model, 'trained_model.pkl')

# 5. Classification probability vectors
y_proba = model.predict_proba(X)
y_test_proba = model.predict_proba(X_test)

# 6. Prepare dataframe to save into CSV
proba_df = pd.DataFrame(y_proba, columns=[f'class{i}' for i in range(len(y.unique()))])
output_df = pd.concat([df[['patient', 'label']], proba_df], axis=1)

proba_test_df = pd.DataFrame(y_test_proba, columns=[f'class{i}' for i in range(len(y_test.unique()))])
output_test_df= pd.concat([df_test[['patient', 'label']], proba_test_df], axis=1)

# 8. Print out accuracy, AUC, precision, recall
y_pred = model.predict(X)
y_test_pred = model.predict(X_test)

# 7. Save probability vectors along with patient id, label into a CSV file
preds_df = pd.DataFrame(y_pred, columns=['preds'])
preds_test_df = pd.DataFrame(y_test_pred, columns=['preds'])

output_df = pd.concat([output_df, preds_df], axis=1)
output_test_df = pd.concat([output_test_df, preds_test_df], axis=1)

output_df.to_csv('radiomics_train_proba.csv', index=False)
output_test_df.to_csv('radiomics_test_proba.csv', index=False)


print(f"Train Accuracy: {accuracy_score(y, y_pred)}")
print(f"Train Precision: {precision_score(y, y_pred, average='macro')}")
print(f"Train Recall: {recall_score(y, y_pred, average='macro')}")
print(f"Train AUC: {roc_auc_score(y, y_proba, multi_class='ovr')}")

print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
print(f"Test Precision: {precision_score(y_test, y_test_pred, average='macro')}")
print(f"Test Recall: {recall_score(y_test, y_test_pred, average='macro')}")
print(f"Test AUC: {roc_auc_score(y_test, y_test_proba, multi_class='ovr')}")
