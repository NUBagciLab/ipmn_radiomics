
# Radiomics Boosts Deep Learning Model for IPMN Classification


**Authors:**

- Lanhong Yao<sup>1</sup>, Zheyuan Zhang<sup>1</sup>, Ugur Demir<sup>1</sup>, Elif Keles<sup>1</sup>, Camila Vendrami<sup>1</sup>, Emil Agarunov<sup>2</sup>, Candice Bolan<sup>3</sup>, Ivo Schoots<sup>4</sup>, Marc Bruno<sup>4</sup>, Rajesh Keswani<sup>1</sup>, Frank Miller<sup>1</sup>, Tamas Gonda<sup>2</sup>, Cemal Yazici<sup>5</sup>, Temel Tirkes<sup>6</sup>, Michael Wallace<sup>7</sup>, Concetto Spampinato<sup>8</sup>, and Ulas Bagci<sup>1</sup>⋆

**Affiliations:** 

<sup>1</sup> Department of Radiology, Northwestern University, Chicago IL 60611, USA  
<sup>2</sup> NYU Langone Health, New York, NY 10016  
<sup>3</sup> Mayo Clinic, Rochester, MN 55905  
<sup>4</sup> Erasmus Medical Center, 3015 GD Rotterdam, Netherlands  
<sup>5</sup> University of Illinois Chicago, Chicago, IL 60607  
<sup>6</sup> Indiana University–Purdue University Indianapolis, Indianapolis, IN 46202  
<sup>7</sup> Sheikh Shakhbout Medical City, 11001, Abu Dhabi, United Arab Emirates  
<sup>8</sup> University of Catania, 95124 Catania CT, Italy


- [Link to Paper](https://arxiv.org/pdf/2309.05857.pdf)
- Diagram [Diagram](/img/IPMN_classification.png)


## Abstract
Intraductal Papillary Mucinous Neoplasm (IPMN) cysts are pre-malignant pancreas lesions, and they can progress into pancreatic cancer. Therefore, detecting and stratifying their risk level is of ultimate importance for effective treatment planning and disease control. However, this is a highly challenging task because of the diverse and irregular shape, texture, and size of the IPMN cysts as well as the pancreas. 

In this study, we propose a novel computer-aided diagnosis pipeline for IPMN risk classification from multi-contrast MRI scans. Our proposed analysis framework includes an efficient volumetric self-adapting segmentation strategy for pancreas delineation, followed by a newly designed deep learning-based classification scheme with a radiomics-based predictive approach. We test our proposed decision-fusion model in multi-center data sets of 246 multi-contrast MRI scans and obtain superior performance to the state of the art (SOTA) in this field. 

Our ablation studies demonstrate the significance of both radiomics and deep learning modules for achieving the new SOTA performance compared to international guidelines and published studies (81.9% vs 61.3% in accuracy). Our findings have important implications for clinical decision-making. In a series of rigorous experiments on multi-center data sets (246 MRI scans from five centers), we achieved unprecedented performance (81.9% accuracy).


## Environment
Ensure you have Python and necessary packages installed by running::
```bash
pip install -r requirements.txt
```


## Steps
The following steps provide a general idea on how to use the code, which may be modified upon your .csv structure. 

1. Dataset Preparation
The dataset used in this project has the following structure, in which `label.csv` contains IPMN classification info for each case.
```bash
-- data
   |-- t1
   |   |-- raw
   |   |-- preprocessed
   |-- t2
   |   |-- raw
   |   |-- preprocessed
   |-- t1_segmentation
   |   |-- raw
   |   |-- preprocessed
   |-- t2_segmentation
   |   |-- raw
   |   |-- preprocessed
   |-- label.csv
```

2. Data Preprocessing
Navigate to `/mri_preprocessing`, adjust `src_dir` and `dst_dir` paths accordingly (eg. /data/t1/raw, /data/t1/preprocessed), and execute:
```bash
python perform_preprocess.py
```


3. Deep Learning Classifier
For training the model within `/Classifier-Transformer-IPMN/Transformer-IPMN`, follow the detailed instructions provided in the seperate README file. Ensure the dataset is divided appropriately into training and testing sets.

Switch the model and model type to use different deep learning models for training. Save the models as .bin file.
After training, run `GET_VAL_PROB.py` and it should read the model files to print out predicted probabilities for the testset, of each model. 
This file `validation_proba_example.csv` gives an example of the structure of how these data should be saved for further processing. 


4. Classifier-Radiomics
Modify `image_dir` and `mask_dir` in `/Classifier-Radiomics` and execute:
```bash
python radiomics_extraction.py
```

After getting the tabular data on volume and radiomics, perform data normalization: ln(x + 1) transformation and unit variance scaling. The output logs in jupyter notebook provide an example of how the inputs should look like.

Train the processed data on `classification_radiomics.ipynb` under `/3classes` or `/4classes`, based on project needs. For example, in this case, there are 4 classification labels for IPMN, 0-normal, 1-low risk, 2-high risk, 3-cancer, and we group label 2 and 3 into label 2, which result in choice of `/3classes`. 


5. Fusion Classifier
After obtaining predictions and probabilities from both classifiers, use `/Classifier-combine` for final prediction integration and evaluation.

Files under `/proba_results` provide an example of how the inputs should look like. Find the best parameters of combination with `grid_search_cv.py`. Use the parameters on testset and evaluate the results with `eval_results.py`. 

## Citations
Please cite our work using the following bibtex entry:
```bibtex
@inproceedings{yao2023radiomics,
  title={Radiomics Boosts Deep Learning Model for IPMN Classification},
  author={Yao, Lanhong and Zhang, Zheyuan and Demir, Ugur and Keles, Elif and Vendrami, Camila and Agarunov, Emil and Bolan, Candice and Schoots, Ivo and Bruno, Marc and Keswani, Rajesh and others},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={134--143},
  year={2023},
  organization={Springer}
}
```
