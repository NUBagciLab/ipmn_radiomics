import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from scipy.stats import mode
'''
cv - cross validation
This version of grid search Splits the data into two halves.
'''


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum

pd1 = pd.read_csv('/Classifier-combine/proba_results/AlexNet_test_proba.csv') #the 5 deep learning models
pd2 = pd.read_csv('/Classifier-combine/proba_results/DenseNet_test_proba.csv')
pd3 = pd.read_csv('/Classifier-combine/proba_results/Mobile_test_proba.csv')
pd4 = pd.read_csv('/Classifier-combine/proba_results/ResNet18_test_proba.csv')
pd5 = pd.read_csv('/Classifier-combine/proba_results/VisionTransformer_test_proba.csv')
pdr = pd.read_csv('/Classifier-combine/proba_results/radiomics_test_proba.csv') #the radiomics model

pd1.drop('preds',axis=1,inplace=True)
pd2.drop('preds',axis=1,inplace=True)
pd3.drop('preds',axis=1,inplace=True)
pd4.drop('preds',axis=1,inplace=True)
pd5.drop('preds',axis=1,inplace=True)
pdr.drop('preds',axis=1,inplace=True)

# Load all prediction sheets
prob_sheets = [pd1, pd2, pd3, pd4, pd5, pdr]

# Split the data into two halves
half_idx = len(prob_sheets[0]) // 2
prob_sheets_test = [df[:half_idx] for df in prob_sheets]
prob_sheets_train = [df[half_idx:] for df in prob_sheets]

a_values = np.linspace(0, 1, 11)  # Adjust the range and number of steps as needed
b_values = np.linspace(0, 1, 11)  # Adjust the range and number of steps as needed
c_values= np.linspace(0, 1, 11)

i = 0
method = 'wa'
if method == 'linear': # (optional) linear layer on the probabilities, not selected in the paper.
    for df_dl in prob_sheets_train[:-1]:  # Iterate over deep learning models for training set
        best_a = None
        best_b = None
        best_c = None
        best_accuracy = 0
        for a in a_values:
            for b in b_values:
                for c in c_values:
                    combined_proba = []
                    for (idx1, row1), (idx2, row2) in zip(df_dl.iterrows(), prob_sheets_train[-1].iterrows()):  
                        r1 = a*row1[2]+(1-a)*row2[2]
                        r2 = b*row1[3]+(1-b)*row2[3]
                        r3 = c*row1[4]+(1-c)*row2[4]
                        combined_proba.append([r1,r2,r3])
                    fused_pred = [np.argmax(cp) for cp in combined_proba]
                    accuracy = accuracy_score(df_dl['label'], fused_pred)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_a = a
                        best_b = b
                        best_c = c
        print('for model ', i+1)
        print(f"Best a: {best_a}, Best b: {best_b},Best c: {best_c} Accuracy: {best_accuracy}")

        # Apply best parameters on the test set
        df_dl_test = prob_sheets_test[i]
        combined_proba_test = []
        for (idx1, row1), (idx2, row2) in zip(df_dl_test.iterrows(), prob_sheets_test[-1].iterrows()):  
            r1_test = best_a*row1[2]+(1-best_a)*row2[2]
            r2_test = best_b*row1[3]+(1-best_b)*row2[3]
            r3_test = best_c*row1[4]+(1-best_c)*row2[4]
            combined_proba_test.append([r1_test,r2_test,r3_test])
        fused_pred_test = [np.argmax(cp) for cp in combined_proba_test]
        accuracy_test = accuracy_score(df_dl_test['label'], fused_pred_test)
        precision_test = precision_score(df_dl_test['label'], fused_pred_test, average='macro')
        recall_test = recall_score(df_dl_test['label'], fused_pred_test, average='macro')
        combined_proba_test = np.array([softmax(cp) for cp in combined_proba_test])
        auc_test = roc_auc_score(df_dl_test['label'], combined_proba_test, multi_class='ovr')

        print('for model ', i+1, ' on test set')
        print(f"Test Accuracy: {accuracy_test}")
        print(f"Test AUC: {auc_test}")
        print(f"Test Precision: {precision_test}")
        print(f"Test Recall: {recall_test}")
        # print('combined proba of testset:\n',combined_proba_test)
        i += 1



elif method == 'wa': #weight averaging 
    k_values = np.linspace(0, 1, 11)
    t_values = np.linspace(0, 1, 11)
    # k_values = [0.4]
    # t_values = [0.8]
    for df_dl in prob_sheets[:-1]:  # Iterate over deep learning models
        best_k = None
        best_t = None
        best_accuracy = 0

        for k in k_values:
            for t in t_values:
                combined_proba = []
                # Iterate over rows in both DataFrames
                for (idx1, row1), (idx2, row2) in zip(df_dl.iterrows(), prob_sheets[-1].iterrows()):  
                    if max(row2[2:]) < t :
                        r1 = k * row1[2] + (1 - k) * row2[2]
                        r2 = k * row1[3] + (1 - k) * row2[3]
                        r3 = k * row1[4] + (1 - k) * row2[4]
                        # combined_proba.append(k * row1[2:] + (1 - k) * row2[2:])
                        combined_proba.append([r1,r2,r3])
                    else:
                        # combined_proba.append(row2[2:])
                        combined_proba.append([row2[2],row2[3],row2[4]])
                fused_pred = [np.argmax(cp) for cp in combined_proba]  # Determine class by max probability
                accuracy = accuracy_score(df_dl['label'], fused_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k
                    best_t = t
        print('for model ', i+1)
        print(f"Best k: {best_k}, Best t: {best_t}, Accuracy: {best_accuracy}")

        # Apply best parameters on the test set
        df_dl_test = prob_sheets_test[i]
        combined_proba_test = []
        for (idx1, row1), (idx2, row2) in zip(df_dl_test.iterrows(), prob_sheets_test[-1].iterrows()):  
            if max(row2[2:]) < best_t :
                r1_test = best_k * row1[2] + (1 - best_k) * row2[2]
                r2_test = best_k * row1[3] + (1 - best_k) * row2[3]
                r3_test = best_k * row1[4] + (1 - best_k) * row2[4]
            else:
                # combined_proba.append(row2[2:])
                r1_test = row2[2]
                r2_test = row2[3]
                r3_test = row2[4]
            combined_proba_test.append([r1_test,r2_test,r3_test])
        fused_pred_test = [np.argmax(cp) for cp in combined_proba_test]
        accuracy_test = accuracy_score(df_dl_test['label'], fused_pred_test)
        precision_test = precision_score(df_dl_test['label'], fused_pred_test, average='macro')
        recall_test = recall_score(df_dl_test['label'], fused_pred_test, average='macro')
        combined_proba_test = np.array([softmax(cp) for cp in combined_proba_test])
        auc_test = roc_auc_score(df_dl_test['label'], combined_proba_test, multi_class='ovr')

        print('for model ', i+1, ' on test set')
        print(f"Test Accuracy: {accuracy_test}")
        print(f"Test AUC: {auc_test}")
        print(f"Test Precision: {precision_test}")
        print(f"Test Recall: {recall_test}")
        i+=1
