import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold,learning_curve
import matplotlib.pyplot as plt
import shap
import sys

# Load data
data = pd.read_csv('df_all.csv')
data.drop('patient',axis=1,inplace=True)
# data.drop('DL',axis=1,inplace=True)
X = data.drop('label', axis=1).values
y = data['label'].values

# Define weights for each class
weights = np.ones_like(y)
# weights[y==2] = 2.0 # double weight for class 2
# weights[y==3] = 2.0 # double weight for class 3

# Define k-fold cross-validation
k = 4
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize arrays to store results
acc_scores = np.zeros(k)
prec_scores = np.zeros((k,3))
rec_scores = np.zeros((k,3))
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

    # Define XGBoost model
    # model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, scale_pos_weight=weights[0]/weights[2], random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3,max_depth=4, n_estimators=140,learning_rate=0.1, random_state=42)

    # Fit model on training data
    model.fit(X_train, y_train, sample_weight=w_train)
    
    ## Plot learning curve after each fold
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=k, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('p1_Learning Curve.png')

    ##

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    acc_scores[i] = accuracy_score(y_test, y_pred)
    print("y_test:",y_test,"y_pred:",y_pred)
    prec_scores[i] = precision_score(y_test, y_pred, average=None,zero_division=1)
    rec_scores[i] = recall_score(y_test, y_pred, average=None,zero_division=1)

    print(f'accuracy={acc_scores[i]}')
    print(f'precision={prec_scores[i]}')
    print(f'recall={rec_scores[i]}')
    # Calculate ROC curve and AUC score
    y_proba = model.predict_proba(X_test)
    np.set_printoptions(threshold=sys.maxsize)
    print('Probability vectors: ')
    print(y_proba.shape)
    print(y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    print('AUC score: ', roc_auc)

    ## Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    ## Plot SHAP values for a single prediction (e.g., the first prediction)
    shap.initjs()
    plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test[0,:])
    shap.save_html(str(i+1)+' fold: shap.html',plot)
    ##

# Print overall results
print('Accuracy: ', np.mean(acc_scores), "+-",np.std(acc_scores) )
print('Precision: ', np.mean(prec_scores, axis=0),"+-",np.std(prec_scores) )
print('Recall: ', np.mean(rec_scores, axis=0),"+-",np.std(rec_scores) )

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = %0.2f)' % mean_auc, color='b', linestyle=':')
plt.legend()
plt.savefig('p3_ROC_curves.png')
