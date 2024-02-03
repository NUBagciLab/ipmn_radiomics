import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

# load data
data = pd.read_csv('uvscaling_t.csv')
data.drop('patient',axis=1,inplace=True)
# separate features and labels
X = data.drop('label', axis=1)
y = data['label']

# define the model pipeline
model = Pipeline(steps=[('feature_selection', RFE(XGBClassifier())),
                        ('classification', XGBClassifier())])

# define the cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# define weights for the classes
weights = [1, 1, 2, 2] # assuming class 2 and 3 have half number of data points

# evaluate model with cross-validation
accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
precision_scores = cross_val_score(model, X, y, scoring='precision_weighted', cv=cv, n_jobs=-1)
recall_scores = cross_val_score(model, X, y, scoring='recall_weighted', cv=cv, n_jobs=-1)
roc_auc_scores = cross_val_score(model, X, y, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1)

# print the mean and standard deviation of the scores
print('Accuracy: {:.3f} ({:.3f})'.format(np.mean(accuracy_scores), np.std(accuracy_scores)))
print('Precision: {:.3f} ({:.3f})'.format(np.mean(precision_scores), np.std(precision_scores)))
print('Recall: {:.3f} ({:.3f})'.format(np.mean(recall_scores), np.std(recall_scores)))
print('ROC AUC: {:.3f} ({:.3f})'.format(np.mean(roc_auc_scores), np.std(roc_auc_scores)))
