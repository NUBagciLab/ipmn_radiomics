import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import glob

# Get a list of all .csv files
files = glob.glob('/data/Classifier-combine/proba_results/*.csv')  # Modify the path if needed
for file in files:
    # Read the data
    half_idx = 24
    df = pd.read_csv(file)[half_idx:]

    # Ground truth
    y_true = df['label']

    # Model predictions
    y_pred = df['preds']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Calculate AUC
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    y_pred_prob = df[['class0', 'class1', 'class2']].values
    auc = roc_auc_score(y_true_bin, y_pred_prob, average='macro', multi_class='ovr')
    
    print(f'File: {file}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print('-'*50)
