"""
==============================
ssp Naive Bayes
# License: BSD 3-clause
==============================
"""

import numpy as np
from SSP_one_slack_ssvm import OneSlackSSVM
from SSP import (SSP_Utils, SSP_InputInstance)
from basic_Utils import (Utils)
from SSP_USCO import (Model)
import multiprocessing
import argparse
import os
import sys
from datetime import datetime

class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataname',  default='col', 
                        choices=['kro', 'col', 'ny'])
    parser.add_argument(
        '--vNum', type=int, default=512, choices=[1024,768,512],
                        help='kro 1024, ny 768, col 512')
    
    
    parser.add_argument(
        '--featureNum', type=int, default=160,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniform', \
            choices=['uniform','true'], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=160, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=100, help='number of testing data')   
    
    parser.add_argument(
        '--testBatch', type=int, default=5, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=4, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default=None,  help='if store a pre_train model')
    
    
    args = parser.parse_args()
    #utils= Utils()
    
    problem ="ssp"
    
    dataname=args.dataname
    vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    testBatch =args.testBatch
    #pairMax=2500
    
    thread = args.thread
    verbose=6
    
    #parameter used in SVM
    C = 0.0001
    tol=0.001
    max_iter =1
    
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    

    
    maxFeatureNum = 10000
    if dataname == "kro":
        pairMax = 570900
        vNum=1024
        
    if dataname == "col":
        pairMax = 137196
        vNum = 512
        
    if dataname == "ny":
        pairMax = 395265   
        vNum = 768
    
    

    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=path+"/data"
    pair_path = "{}/{}/{}/{}_{}_trainAllShuffle".format(data_path,problem,dataname,problem,dataname)
    stoGraphPath = "{}/{}/{}/{}_{}".format(data_path,problem,dataname,problem,dataname)
    featurePath = "{}/{}/{}/features/{}_{}".format(data_path,problem,dataname,featureGenMethod,maxFeatureNum)
    #if args.log_paht is not None:
    logpath=path+"/log"

    
    X_train, Y_train, Y_train_length, X_test, Y_test, Y_test_length = SSP_Utils.getDataTrainTestRandom(pair_path,trainNum,testNum*testBatch, pairMax)
    print("data fetched")
    Utils.writeToFile(logpath, "data fetched")
 
    instance = SSP_InputInstance(stoGraphPath, featurePath, featureNum, vNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    
    #sys.exit("stop")
    #**************************OneSlackSSVM
    model = Model()
    model.initialize(X_train, Y_train, instance)
    
    one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             max_iter = max_iter, log = logpath)
    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Testing NB Started", toconsole = True,preTrainPathResult = preTrainPathResult)
    Y_pred = SSP_Utils.NB_prediction(instance.stoGraph.unitGraph, X_train, Y_train, X_test, n_jobs = thread)
    instance.test_batch(X_test, Y_test_length,  Y_pred, testBatch, testNum, logpath, preTrainPathResult = preTrainPathResult)



    Utils.writeToFile(logpath, dataname, toconsole = True,preTrainPathResult = preTrainPathResult)
    #print(dataname)
    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True,preTrainPathResult = preTrainPathResult)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()