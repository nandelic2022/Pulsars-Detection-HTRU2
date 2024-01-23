# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:41:57 2022

@author: Admin
"""
import re
import random 
import numpy as np 
import pandas as pd 
import gplearn.genetic
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
def generateGPParameters(): 
    parameters = [] 
    PopSize = random.randint(1000, 2000)
    noGen = random.randint(20,30)
    while True: 
        tourSize = random.randint(100,200)
        if tourSize < PopSize:
            break 
        else: 
            pass 
    treeDepth = (random.randint(3,7), random.randint(8,18))
    while True: 
        x = 0
        crosCoeff = random.uniform(0.01,1)
        pSubMute = random.uniform(0.8,1)
        pHoistMute = random.uniform(0.001,1)
        pPointMute = random.uniform(0.001,1)
        x = crosCoeff + pSubMute + pHoistMute + pPointMute 
        print("x = {}".format(x))
        if x >=0.999 and x <= 1: 
            print("Crossover = {}".format(crosCoeff))
            print("subtree mutation = {}".format(pSubMute))
            print("Hoist Mutation = {}".format(pHoistMute))
            print("Point Mutation = {}".format(pPointMute))
            break
        else: 
            pass
    stoppingCrit = random.randint(1,1000)/1000000.0
    maxSamples = random.uniform(0.7,1)
    constRange = (-random.uniform(0.01,10000), random.uniform(0.01,10000))
    parsimony = random.uniform(.1,1)/1000000.0
    parameters = [PopSize, \
                  noGen,\
                  tourSize,\
                  treeDepth,\
                  crosCoeff,\
                  pSubMute,\
                  pHoistMute,\
                  pPointMute,\
                  stoppingCrit,\
                  maxSamples,\
                  constRange,\
                  parsimony]
    print("Chosen Parameters = {}".format(parameters))
    file0.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(PopSize, \
                                                                           noGen,\
                                                                           tourSize,\
                                                                           treeDepth,\
                                                                           crosCoeff,\
                                                                           pSubMute,\
                                                                           pHoistMute,\
                                                                           pPointMute,\
                                                                           stoppingCrit,\
                                                                           maxSamples,\
                                                                           constRange,\
                                                                           parsimony))
    file1.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(PopSize, \
                                                                           noGen,\
                                                                           tourSize,\
                                                                           treeDepth,\
                                                                           crosCoeff,\
                                                                           pSubMute,\
                                                                           pHoistMute,\
                                                                           pPointMute,\
                                                                           stoppingCrit,\
                                                                           maxSamples,\
                                                                           constRange,\
                                                                           parsimony))
    file0.flush()
    file1.flush()
    return parameters
def log2(x):
    with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x) > 0.001, np.log2(np.abs(x)),0.)
def log10(x):
    with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x) > 0.001, np.log10(np.abs(x)),0.)
import scipy.special as sp 
def CubeRoot(x):
    return sp.cbrt(x)
gp_CRoot = make_function(function = CubeRoot, name = "CubeRoot",arity = 1)
gp_log2 = make_function(function = log2, name = "log2",arity = 1)
gp_log10 = make_function(function = log10, name = "log10",arity = 1)
def GeneticProgramming(genes,X_train,X_test, y_train, y_test):
    CleanFormulas = [] # ovdje se spremaju sirove formule iz GP-a 
    NumpyFormulas = [] # ovdje se spremaju formule s numpy funkcijama 
    ACC_TRAIN = []
    ACC_VALID = []
    ROC_AUC_TRAIN = []
    ROC_AUC_VALID = []
    PRECISION_TRAIN = []
    PRECISION_VALID = []
    RECALL_TRAIN = []
    RECALL_VALID = []
    F1_SCORE_TRAIN = []
    F1_SCORE_VALID = []
    est_gp = gplearn.genetic.SymbolicClassifier(population_size = genes[0], 
                                                generations = genes[1], 
                                                tournament_size = genes[2],
                                                init_depth = genes[3],
                                                p_crossover = genes[4],
                                                p_subtree_mutation= genes[5],
                                                p_hoist_mutation= genes[6], 
                                                p_point_mutation= genes[7],
                                                stopping_criteria = genes[8],
                                                max_samples = genes[9], 
                                                const_range=genes[10],
                                                parsimony_coefficient = genes[11],
                                                verbose = True,
                                                n_jobs=-1,
                                                function_set = ('add',
                                                                'sub', 
                                                                'mul', 
                                                                'div',
                                                                'sqrt',
                                                                'abs',
                                                                'log',
                                                                "max",
                                                                "min",
                                                                "sin",
                                                                "cos",
                                                                "tan",
                                                                gp_CRoot,
                                                                gp_log2,
                                                                gp_log10))
    k = 5 
    kf = KFold(n_splits = k, shuffle = True, random_state = 42)
    for train_index, test_index in kf.split(X_train):
        X_Train, X_Valid = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        y_Train, y_Valid = y_train.iloc[train_index], y_train.iloc[test_index]
        est_gp.fit(X_Train, y_Train)
        print("Formula = {}".format(est_gp._program))
        print("Formula Depth = {}".format(est_gp._program.depth_))
        print("Formula Length = {}".format(est_gp._program.length_))
        # Write formula into files 
        file0.write("Formula = {}, depth = {}, length = {}".format(est_gp._program, est_gp._program.depth_, est_gp._program.length_))
        file2.write("Formula = {}, depth = {}, length = {}\n".format(est_gp._program, est_gp._program.depth_, est_gp._program.length_))
        CleanFormulas.append([str(est_gp._program)]) 
        print("Current Clean Formula status = {}".format(len(CleanFormulas)))
        NumpyFormulas.append(processingFormulas([str(est_gp._program)]))
        print("Current Numpy Formula status = {}".format(len(NumpyFormulas)))
        print("Numpy formula list = {}".format(NumpyFormulas))
        #######################################################################
        # Obtain Scores on Train part of each CV
        #######################################################################
        ACCURACY_train = accuracy_score(y_Train, est_gp.predict(X_Train))
        ROCAUC_train = roc_auc_score(y_Train, est_gp.predict(X_Train))
        PRECISION_train = precision_score(y_Train, est_gp.predict(X_Train))
        RECALL_train = recall_score(y_Train, est_gp.predict(X_Train))
        F1_SCORE_train = f1_score(y_Train, est_gp.predict(X_Train))
        ACC_TRAIN.append(ACCURACY_train)
        ROC_AUC_TRAIN.append(ROCAUC_train)
        PRECISION_TRAIN.append(PRECISION_train)
        RECALL_TRAIN.append(RECALL_train)
        F1_SCORE_TRAIN.append(F1_SCORE_train)
        #######################################################################
        # Obtain Scores on Validation Fold in each CV 
        #######################################################################
        ACCURACY_valid = accuracy_score(y_Valid, est_gp.predict(X_Valid))
        ROCAUC_valid = roc_auc_score(y_Valid, est_gp.predict(X_Valid))
        PRECISION_valid = precision_score(y_Valid, est_gp.predict(X_Valid))
        RECALL_valid = recall_score(y_Valid, est_gp.predict(X_Valid))
        F1_SCORE_valid = f1_score(y_Valid, est_gp.predict(X_Valid))
        ACC_VALID.append(ACCURACY_valid)
        ROC_AUC_VALID.append(ROCAUC_valid)
        PRECISION_VALID.append(PRECISION_valid)
        RECALL_VALID.append(RECALL_valid)
        F1_SCORE_VALID.append(F1_SCORE_valid)
        print("ACC TRAIN = {}\n ACC Valid = {}".format(ACCURACY_train, ACCURACY_valid))
    ###########################################################################
    # Calculate average ROC_AUC_SCORE, Precision, Recall, F1_score 
    ###########################################################################
    file4.write("""ACCURACY_TRAIN_CV = {}\n ACCURACY_VALID = {}\n ROC_AUC_TRAIN_CV = {}\n ROC_AUC_VALID_CV = {}\n PRECISION_TRAIN_CV = {}\n PRECISION_VALID_CV = {}\n RECALL_TRAIN_CV = {}\n RECALL_VALID_CV = {}\n F1_SCORE_TRAIN_CV = {}\n F1_SCORE_VALID_CV = {}\n""".format(
                ACC_TRAIN,
                ACC_VALID,
                ROC_AUC_TRAIN, 
                ROC_AUC_VALID,
                PRECISION_TRAIN,
                PRECISION_VALID,
                RECALL_TRAIN,
                RECALL_VALID,
                F1_SCORE_TRAIN,
                F1_SCORE_VALID))
    MEAN_ACC = np.mean([np.mean(ACC_TRAIN), np.mean(ACC_VALID)])
    STD_ACC = np.std([np.mean(ACC_TRAIN), np.mean(ACC_VALID)])
    MEAN_ROCAUC = np.mean([np.mean(ROC_AUC_TRAIN), np.mean(ROC_AUC_VALID)])
    STD_ROCAUC = np.std([np.mean(ROC_AUC_TRAIN), np.mean(ROC_AUC_VALID)])
    MEAN_PRECISION = np.mean([np.mean(PRECISION_TRAIN), np.mean(PRECISION_VALID)])
    STD_PRECISION = np.std([np.mean(PRECISION_TRAIN), np.mean(PRECISION_VALID)])
    MEAN_RECALL = np.mean([np.mean(RECALL_TRAIN), np.mean(RECALL_VALID)])
    STD_RECALL = np.std([np.mean(RECALL_TRAIN), np.mean(RECALL_VALID)])
    MEAN_F1_SCORE = np.mean([np.mean(F1_SCORE_TRAIN), np.mean(F1_SCORE_VALID)])
    STD_F1_SCORE = np.std([np.mean(F1_SCORE_TRAIN), np.mean(F1_SCORE_VALID)])
    print("####################################################################")
    print("MEAN_ACC = {}".format(MEAN_ACC))
    print("STD_ACC = {}".format(STD_ACC))
    print("MEAN_ROC_AUC_CV = {}".format(MEAN_ROCAUC))
    print("STD_ROC_AUC_CV = {}".format(STD_ROCAUC))
    print("MEAN_PRECISION_CV = {}".format(MEAN_PRECISION))
    print("STD_PRECISION_CV = {}".format(STD_PRECISION))
    print("MEAN_RECALL_CV = {}".format(MEAN_RECALL))
    print("STD_RECALL_CV = {}".format(STD_RECALL))
    print("MEAN_F1_SCORE_CV = {}".format(MEAN_F1_SCORE))
    print("STD_F1_SCORE_CV = {}".format(STD_F1_SCORE))
    print("####################################################################")
    for i in range(len(NumpyFormulas)):
        file3.write("Clean Fomula Fold {} = {}\n".format(i, NumpyFormulas[i]))
    file4.write("""MEAN_ACC = {}\n STD_ACC = {}\n MEAN_ROC_AUC = {}\n STD_ROC_AUC = {}\n MEAN_PRECISION = {}\n STD_PRECISION = {}\n MEAN_RECALL = {}\n STD_RECALL = {}\n MEAN_F1_SCORE = {}\n STD_F1_SCORE = {}\n""".format(
        MEAN_ACC,
        STD_ACC,
        MEAN_ROCAUC,
        STD_ROCAUC,
        MEAN_PRECISION,
        STD_PRECISION,
        MEAN_RECALL,
        STD_RECALL,
        MEAN_F1_SCORE,
        STD_F1_SCORE))
    if all(x > 0.99 for x in (MEAN_ACC, MEAN_ROCAUC, MEAN_PRECISION, MEAN_RECALL, MEAN_F1_SCORE)):
        #############################################
        #Calculate the output of each formula 
        #############################################
        y_pred_train = [[] for i in range(len(NumpyFormulas))] 
        y_pred_test = [[] for i in range(len(NumpyFormulas))] 
        columns = ["X{}".format(i) for i in range(len(list(X_train.columns)))]
        X_train.columns = columns; X_test.columns = columns
        # print(Equations)
        NumpyFormulas_train = [NumpyFormulas[i][0] for i in range(len(NumpyFormulas))]
        NumpyFormulas_test = [NumpyFormulas[i][0] for i in range(len(NumpyFormulas))]
        print(type(NumpyFormulas_train[0]))
        # print(NumpyFormulas_train)
        for i in range(len(NumpyFormulas_train)):
            for j in range(len(columns)):
                pattern = re.compile(r'\b{}\b'.format(columns[j]))
                NumpyFormulas_train[i] = pattern.sub('X_train.loc[z, "{}"]'.format(columns[j]), NumpyFormulas_train[i])
                NumpyFormulas_test[i] = pattern.sub('X_test.loc[i, "{}"]'.format(columns[j]), NumpyFormulas_test[i])
        print(NumpyFormulas_train)
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        # print(X_train)
        def Sigmoid(x):
            return round(1/(1+np.exp(-x)),0)
        for z in range(len(X_train)):
            exec(NumpyFormulas_train[0])
            if z%100 == 0:
                print(f"Evaluating formulas on training dataset ... Currently {round(z/len(X_train)*100,4)}% completed")
                
            # print("i = {}".format(i))
            # result = [eval(NumpyFormulas_train[j]) for j in range(len(NumpyFormulas_train))]
            result = [eval(NumpyFormulas_train[0]),
                     eval(NumpyFormulas_train[1]),
                     eval(NumpyFormulas_train[2]),
                     eval(NumpyFormulas_train[3]),
                     eval(NumpyFormulas_train[4])]
            #print(result)
            res_sig = [Sigmoid(result[k]) for k in range(len(result))]
            for j in range(len(res_sig)):
                y_pred_train[j].append(res_sig[j])        

        for i in range(len(X_test)):
            if i%100 == 0:
                print(f"Evaluating formulas on testing dataset ... Currently {round(i/len(X_test)*100,4)}% completed")
                
            # print("i = {}".format(i))
            # result = [eval(NumpyFormulas_test[j]) for j in range(len(NumpyFormulas_test))]
            result = [eval(NumpyFormulas_test[0]),
                     eval(NumpyFormulas_test[1]),
                     eval(NumpyFormulas_test[2]),
                     eval(NumpyFormulas_test[3]),
                     eval(NumpyFormulas_test[4])]
            #print(result)
            res_sig = [Sigmoid(result[k]) for k in range(len(result))]
            for j in range(len(res_sig)):
                y_pred_test[j].append(res_sig[j])         
        ACCURACY_FINAL_TRAIN = np.mean([accuracy_score(y_train, y_pred_train[i]) for i in range(len(y_pred_train))])
        ACCURACY_FINAL_TEST = np.mean([accuracy_score(y_test, y_pred_test[i]) for i in range(len(y_pred_test))]) 
        ROC_AUC_FINAL_TRAIN = np.mean([roc_auc_score(y_train, y_pred_train[i]) for i in range(len(y_pred_train))])
        ROC_AUC_FINAL_TEST = np.mean([roc_auc_score(y_test, y_pred_test[i]) for i in range(len(y_pred_test))]) 
        PRECISION_FINAL_TRAIN = np.mean([precision_score(y_train, y_pred_train[i]) for i in range(len(y_pred_train))])
        PRECISION_FINAL_TEST = np.mean([precision_score(y_test, y_pred_test[i]) for i in range(len(y_pred_test))]) 
        RECALL_FINAL_TRAIN = np.mean([recall_score(y_train, y_pred_train[i]) for i in range(len(y_pred_train))])
        RECALL_FINAL_TEST = np.mean([recall_score(y_test, y_pred_test[i]) for i in range(len(y_pred_test))]) 
        F1_SCORE_FINAL_TRAIN = np.mean([f1_score(y_train, y_pred_train[i]) for i in range(len(y_pred_train))])
        F1_SCORE_FINAL_TEST = np.mean([f1_score(y_test, y_pred_test[i]) for i in range(len(y_pred_test))]) 
        FINAL_MEAN_ACC = np.mean([ACCURACY_FINAL_TRAIN, ACCURACY_FINAL_TEST])
        FINAL_STD_ACC = np.std([ACCURACY_FINAL_TRAIN, ACCURACY_FINAL_TEST])
        FINAL_MEAN_ROC_AUC = np.mean([ROC_AUC_FINAL_TRAIN, ROC_AUC_FINAL_TEST])
        FINAL_STD_ROC_AUC = np.std([ROC_AUC_FINAL_TRAIN, ROC_AUC_FINAL_TEST])
        FINAL_MEAN_PRECISION = np.mean([PRECISION_FINAL_TRAIN, PRECISION_FINAL_TEST])
        FINAL_STD_PRECISION = np.std([PRECISION_FINAL_TRAIN, PRECISION_FINAL_TEST])
        FINAL_MEAN_RECALL = np.mean([RECALL_FINAL_TRAIN, RECALL_FINAL_TEST])
        FINAL_STD_RECALL = np.std([RECALL_FINAL_TRAIN, RECALL_FINAL_TEST])
        FINAL_MEAN_F1_SCORE = np.mean([F1_SCORE_FINAL_TRAIN, F1_SCORE_FINAL_TEST])
        FINAL_STD_F1_SCORE = np.std([F1_SCORE_FINAL_TRAIN, F1_SCORE_FINAL_TEST])
        file4.write("""ACCURACY_FINAL_TRAIN = {}\n ACCURACY_FINAL_TEST = {}\n FINAL_MEAN_ACC = {}\n FINAL_STD_ACC = {}\n ROC_AUC_FINAL_TRAIN = {}\n ROC_AUC_FINAL_TEST = {}\n FINAL_MEAN_ROC_AUC = {}\n FINAL_STD_ROC_AUC = {}\n PRECISION_FINAL_TRAIN = {}\n PRECISION_FINA_TEST = {}\n FINAL_MEAN_PRECISION = {}\n FINAL_STD_PRECISION = {}\n RECALL_FINAL_TRAIN = {}\n RECALL_FINAL_TEST = {}\n FINAL_MEAN_RECALL = {}\n FINAL_STD_RECALL = {}\n F1_SCORE_FINAL_TRAIN = {}\n F1_SCORE_FINAL_TEST = {}\n FINAL_MEAN_F1_SCORE = {}\n FINAL_STD_F1_SCORE = {}\n""".format(
                    ACCURACY_FINAL_TRAIN, 
                    ACCURACY_FINAL_TEST,
                    FINAL_MEAN_ACC, 
                    FINAL_STD_ACC,
                    ROC_AUC_FINAL_TRAIN,
                    ROC_AUC_FINAL_TEST,
                    FINAL_MEAN_ROC_AUC,
                    FINAL_STD_ROC_AUC,
                    PRECISION_FINAL_TRAIN, 
                    PRECISION_FINAL_TEST, 
                    FINAL_MEAN_PRECISION,
                    FINAL_STD_PRECISION,
                    RECALL_FINAL_TRAIN,
                    RECALL_FINAL_TEST,
                    FINAL_MEAN_RECALL,
                    FINAL_STD_RECALL,
                    F1_SCORE_FINAL_TRAIN,
                    F1_SCORE_FINAL_TEST,
                    FINAL_MEAN_F1_SCORE,
                    FINAL_STD_F1_SCORE))
        print("""ACCURACY_FINAL_TRAIN = {}
                    ACCURACY_FINAL_TEST = {}
                    FINAL_MEAN_ACC = {}
                    FINAL_STD_ACC = {}
                    ROC_AUC_FINAL_TEST = {}
                    FINAL_MEAN_ROC_AUC = {}
                    FINAL_STD_ROC_AUC = {}
                    PRECISION_FINAL_TRAIN = {}
                    PRECISION_FINA_TEST = {}
                    FINAL_MEAN_PRECISION = {}
                    FINAL_STD_PRECISION = {}
                    RECALL_FINAL_TRAIN = {}
                    RECALL_FINAL_TEST = {}
                    FINAL_MEAN_RECALL = {}
                    FINAL_STD_RECALL = {}
                    F1_SCORE_FINAL_TRAIN = {}
                    F1_SCORE_FINAL_TEST = {}
                    FINAL_MEAN_F1_SCORE = {}
                    FINAL_STD_F1_SCORE = {}""".format(
                    ACCURACY_FINAL_TRAIN, 
                    ACCURACY_FINAL_TEST,
                    FINAL_MEAN_ACC, 
                    FINAL_STD_ACC,
                    ROC_AUC_FINAL_TRAIN,
                    ROC_AUC_FINAL_TEST,
                    FINAL_MEAN_ROC_AUC,
                    FINAL_STD_ROC_AUC,
                    PRECISION_FINAL_TRAIN, 
                    PRECISION_FINAL_TEST, 
                    FINAL_MEAN_PRECISION,
                    FINAL_STD_PRECISION,
                    RECALL_FINAL_TRAIN,
                    RECALL_FINAL_TEST,
                    FINAL_MEAN_RECALL,
                    FINAL_STD_RECALL,
                    F1_SCORE_FINAL_TRAIN,
                    F1_SCORE_FINAL_TEST,
                    FINAL_MEAN_F1_SCORE,
                    FINAL_STD_F1_SCORE))        
        file0.flush(); file1.flush(); file2.flush(); file3.flush(); file4.flush()
        return FINAL_MEAN_ROC_AUC
    else:
        file0.flush(); file1.flush(); file2.flush(); file3.flush(); file4.flush()
        return MEAN_ROCAUC
def processingFormulas(formulaList): 
    for i in range(len(formulaList)):
        if "add" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("add", "np.add")
        if "sub" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("sub", "np.subtract")
        if "mul" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("mul", "np.multiply")
        if "neg" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("neg", "np.negative")
        if "abs" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("abs", "np.abs")
        if "sin" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("sin", "np.sin")
        if "cos" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("cos", "np.cos")
        if "tan" in formulaList[i]:
            formulaList[i] = formulaList[i].replace("tan", "np.tan")
    procFormulaList = formulaList
    return procFormulaList
def log(x1):
      with np.errstate(divide = "ignore", invalid = "ignore"):
          return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)),0.)
def sqrt(x1):
    return np.sqrt(np.abs(x1))
def div(x1,x2):
    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.where(np.abs(x2) > 0.001, np.divide(x1,x2), 1.)    

###############################################################################
# DATA PREPARATION AND PROGRAM EXECUTION 
###############################################################################
df = pd.read_csv("AllKNN.csv").sample(frac=1)
y = df.pop('target')

X_train, X_test, y_train, y_test = train_test_split(df,y, test_size=0.3, random_state = 102)
name="test"
file0 = open("{}_GP_History_log.data".format(name),"w")
file1 = open("{}_GP_Parameters.data".format(name),"w")
file2 = open("{}_GP_Raw_Formulas.data".format(name),"w")
file3 = open("{}_GP_Clean_Formulas.data".format(name),"w")
file4 = open("{}_GP_Scores.data".format(name),"w")
k = 0 
while True:
    file0.write("k = {}\n".format(k))
    file1.write("k = {}\n".format(k))
    file2.write("k = {}\n".format(k))
    file3.write("k = {}\n".format(k))
    file4.write("k = {}\n".format(k))
    Parameters = generateGPParameters()
    ROCAUCSCORE = GeneticProgramming(Parameters,X_train,X_test, y_train, y_test)
    if ROCAUCSCORE > 0.99:
        print("Solution is Found!")
        file0.write("Solution is Found!!\n")
        file1.write("Solution is Found!!\n")
        file2.write("Solution is Found!!\n")
        file3.write("Solution is Found!!\n")
        file4.write("Solution is Found!!\n")
        break
    else:
        k += 1
        pass
file0.close()
file1.close()
file2.close()
file3.close()
file4.close()
