"""
==============================
BM USCO-Solver
# License: BSD 3-clause
==============================
"""

import numpy as np
from BM_one_slack_ssvm import OneSlackSSVM
from BM import (BM_Utils, BM_InputInstance)
from basic_Utils import Utils
from basic_USCO import Model
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
        '--dataname',  default='norm_30_128', 
                        choices=['orm_30_128'])

    
    
    parser.add_argument(
        '--featureNum', type=int, default=640,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniformFloat1000', \
            choices=['uniform','true', ], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=640, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=160, help='number of testing data')   
    
    
     
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default=None,  help='if store a pre_train model')
    
    
    args = parser.parse_args()

    
    problem ="bm"
    
    dataname=args.dataname

    
    
    
    trainNum =args.trainNum
    testNum =args.testNum

    
    thread = args.thread
    verbose=6
    
    #parameter used in SVM
    C = 0.0001
    tol=0.001
    max_iter = 5
    
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    
    
    
    if dataname == "norm_30":
        maxFeatureNum = 10000
        pairMax = 10000
        scale = 20
        #vNum=1024
        
    if dataname == "norm_30_128":
        if featureGenMethod == "uniform":
            maxFeatureNum = 10000
        if featureGenMethod == "uniform001_ver0":
            maxFeatureNum = 4000
        if featureGenMethod == "uniform001_ver1":
            maxFeatureNum = 4000
            
        if featureGenMethod == "uniformFloat100":
            maxFeatureNum = 1000
        if featureGenMethod == "uniformFloat1000":
            maxFeatureNum = 1000
        pairMax = 1000
        scale = 20
        #vNum=1024
        
    
    
    
    

    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=path+"/data"
    pair_path = "{}/{}/{}/{}_{}_train_{}_{}".format(data_path,problem,dataname,problem,dataname,pairMax,  scale)
    stoBMGraphPath = "{}/{}/{}/{}_{}".format(data_path,problem,dataname,problem,dataname)
    featurePath = "{}/{}/{}/features/{}_{}".format(data_path,problem,dataname,featureGenMethod,maxFeatureNum)
    #if args.log_paht is not None:
    logpath=path+"/log/bm_"+dataname

    

    X_train, Y_train, X_test, Y_test = BM_Utils.getDataTrainTestRandom(pair_path,trainNum,testNum, pairMax)

    print("data fetched")
    
    Utils.writeToFile(logpath, "data fetched")
    

    instance = BM_InputInstance(stoBMGraphPath, featurePath, featureNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    

    #**************************OneSlackSSVM
    model = Model()
    model.initialize(X_train, Y_train, instance)
    
    one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             max_iter = max_iter, log = logpath)
    
    
    one_slack_svm.fit(X_train, Y_train, initialize = False)
    
    
    if pre_train is True:
        now = datetime.now()
        preTrainPath=path+"/pre_train/"+problem+"/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        Utils.save_pretrain(preTrainPath, one_slack_svm.w, instance.featureIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole = True)   
        preTrainPathResult = preTrainPath+"/result"
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Testing USCO Started", toconsole = True,preTrainPathResult = preTrainPathResult)
    Y_pred = one_slack_svm.predict(X_test, featureNum)

    instance.test(X_test, Y_test,  Y_pred, logpath, preTrainPathResult = preTrainPathResult)
    
    

    Utils.writeToFile(logpath, dataname, toconsole = True,preTrainPathResult = preTrainPathResult)

    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True,preTrainPathResult = preTrainPathResult)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()