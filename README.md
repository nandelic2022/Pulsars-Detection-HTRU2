# Pulsars-Detection-HTRU2

This repository provides the code developed and used in the paper "Improvement of Pulsars Detection Using Dataset Balancing Methods and Symbolic Classification Ensemble". The dataset used in this research is a publicly available dataset HTRU2 which you can download at UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/372/htru2 or Kaggle https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2. 



The HTRU2.csv consists of the following columns: 
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

The first 8 variables are input variables and the last one 9. class is the target variable. One of the problems with this dataset is that it contains a large number of outliers so in this research the Winsorizer function was used from the feature_engine library (capping_method = IQR, tail=both) for capping the outliers. For more information about feature_engine library and winsorizer function and parameters please visit: https://feature-engine.trainindata.com/en/latest/api_doc/outliers/Winsorizer.html#feature_engine.outliers.Winsorizer . The dataset consists of 17898 samples, where 1639 samples are pulsars and 16259 are non-pulsars. This means that the dataset is highly imbalanced so the application of dataset-balancing techniques was required. On the dataset, oversampling and undersampling techniques were applied to obtain balanced dataset variations which were later used in GPSC to obtain symbolic expressions (mathematical equations) for the detection of pulsars. The library that was used for oversampling/undersampling techniques is the imblearn library (https://imbalanced-learn.org/stable/). 

To run the GPSC algorithm (GPSC_CV5_test.py) you need a Python 3.9 and install these Python libraries: 
1. numpy (pip install numpy)
2. gplearn 0.4.2 (pip install gplearn)
3. scikit-learn (pip install scikit-learn)
4. pandas (pip install pandas)

If you have installed the Anaconda distribution of Python then use the conda command in the anaconda prompt to install the aforementioned libraries.

When all libraries are installed using pip or conda command ensure that the dataset is in the same folder as the python script or provide an address where the dataset (.csv) is stored. Then in the Python script on command line 407 type in the address (if the dataset is in another folder) or simply the dataset name (if the .csv file is in the same folder as the python script) and run the Python script. 

Acknowledgement: If you use this code please cite the following article - Anđelić N. - Improvement of Pulsars Detection Using Dataset Balancing Methods and Symbolic Classification Ensemble, In progress 
