import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score, precision_score, recall_score
from scipy.stats import mode

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo / expo_sum

pd1 = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/proba_results/AlexNet_test_proba.csv')
pd2 = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/proba_results/DenseNet_test_proba.csv')
pd3 = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/proba_results/Mobile_test_proba.csv')
pd4 = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/proba_results/ResNet18_test_proba.csv')
pd5 = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/proba_results/VisionTransformer_test_proba.csv')
pdr = pd.read_csv('/home/lys6602/Radiomics/Classifier-combine/radiomics_test_proba.csv')

pd1.drop('preds',axis=1,inplace=True)
pd2.drop('preds',axis=1,inplace=True)
pd3.drop('preds',axis=1,inplace=True)
pd4.drop('preds',axis=1,inplace=True)
pd5.drop('preds',axis=1,inplace=True)
pdr.drop('preds',axis=1,inplace=True)


# Load all prediction sheets
prob_sheets = [pd1,pd2,pd3,pd4,pd5,pdr]

a_values = np.linspace(0, 1, 11)  # Adjust the range and number of steps as needed
b_values = np.linspace(0, 1, 11)  # Adjust the range and number of steps as needed
c_values= np.linspace(0, 1, 11)

i=0
method = 'wa'
if method == 'linear':# (optional) linear layer on the probabilities, not selected in the paper.
    for df_dl in prob_sheets[:-1]:  # Iterate over deep learning models
        best_a = None
        best_b = None
        best_c = None
        best_accuracy = 0
        for a in a_values:
            for b in b_values:
                for c in c_values:
                    combined_proba = []
                    # Iterate over rows in both DataFrames
                    for (idx1, row1), (idx2, row2) in zip(df_dl.iterrows(), prob_sheets[-1].iterrows()):  
                        r1 = a*row1[2]+(1-a)*row2[2]
                        r2 = b*row1[3]+(1-b)*row2[3]
                        r3 = c*row1[4]+(1-c)*row2[4]
                        combined_proba.append([r1,r2,r3])
                    fused_pred = [np.argmax(cp) for cp in combined_proba]  # Determine class by max probability
                    accuracy = accuracy_score(df_dl['label'], fused_pred)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_a = a
                        best_b = b
                        best_c = c
                        
                        precision = precision_score(df_dl['label'], fused_pred, average='macro')
                        recall = recall_score(df_dl['label'], fused_pred, average='macro')
                        combined_proba = np.array([softmax(cp) for cp in combined_proba])
                        auc = roc_auc_score(df_dl['label'], combined_proba, multi_class='ovr')
                print(f"a= {a}, b={b},c={c}, accuracy = {accuracy}")
        print('for model ', i+1)
        print(f"Best a: {best_a}, Best b: {best_b},Best c: {best_c}")
        print(f"Accuracy: {best_accuracy}")

        print(f"AUC: {auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        i+=1

elif method == 'wa':#weight averaging
    # k_values = np.linspace(0, 1, 11)
    # t_values = np.linspace(0, 1, 11)
    k_values = [0.4]
    t_values = [0.8]
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

                    precision = precision_score(df_dl['label'], fused_pred, average='macro')
                    recall = recall_score(df_dl['label'], fused_pred, average='macro')
                    combined_proba = np.array([softmax(cp) for cp in combined_proba])
                    auc = roc_auc_score(df_dl['label'], combined_proba, multi_class='ovr')
                print(f"k= {k}, t={t}, accuracy = {accuracy}")
        print('for model ', i+1)
        print(f"Best k: {best_k}, Best t: {best_t}, Accuracy: {best_accuracy}")
        print(f"AUC: {auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        i+=1
