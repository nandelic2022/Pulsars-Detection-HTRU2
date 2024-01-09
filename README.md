# Pulsars-Detection-HTRU2

This repository provides the code developed and used in the paper "Improvement of Pulsars Detection Using Dataset Balancing Methods and Symbolic Classification Ensemble". The dataset used in this research is a publicly availabe dataset HTRU2 which you can download at UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/372/htru2 or Kaggle 


The HTRU2.csv consist of following columns: 
1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

The first 8 variables are input variables and the last one 9. class is the target variable. One of the problems with this dataset is that it contains a large number of outliers so in this research the Winsorizer function was used form feature_engine library (capping_method = IQR, tail=both) for capping the outliers. For more information about feature_engine library and winsorizer function and parameters please visit: https://feature-engine.trainindata.com/en/latest/api_doc/outliers/Winsorizer.html#feature_engine.outliers.Winsorizer . The dataset consist of 17898 samples, where 1639 samples are pulsars and 16259 are non-pulsars. This means that the dataset is highly imbalanced so the application of dataset balancing techniques was required. On the dataset oversampling and undersampling techniques were applied to obtain balanced dataset variations which were later used in GPSC to obtain symbolic expressions (mathematical equations) for detection of pulsars. The library which was used for oversampling/undersampling techniques is imblearn library (https://imbalanced-learn.org/stable/). 

The GPSC 

