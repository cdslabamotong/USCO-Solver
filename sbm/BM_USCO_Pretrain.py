"""
==============================
BM USCO-Solver preTrain models
# License: BSD 3-clause
==============================
"""

import numpy as np
from BM_one_slack_ssvm import OneSlackSSVM
from BM import (BM_Utils, BM_InputInstance)
from basic_Utils import Utils
from basic_USCO import Model
import argparse
import os


class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataname',  default='norm_30_128', 
                        choices=['norm_30_128'])

    
    
    
    parser.add_argument(
        '--featureGenMethod', default='uniform', \
            choices=['uniform','true', 'uniformFloat1000'], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
    parser.add_argument(
        '--preTrainPath', default="/pre_train/final/norm30_128_uniform_6400_160_640/", 
                        help='number of features (random subgraphs) used in StratLearn ')
    
    parser.add_argument(
        '--trainNum', type=int, default=640, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=640, help='number of testing data')   
    
    parser.add_argument(
        '--testBatch', type=int, default=1, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
    
    
    
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
    
    featureGenMethod = args.featureGenMethod
    preTrainPath = args.preTrainPath
    
    

        
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
        if featureGenMethod == "true":
            maxFeatureNum = 10000
        pairMax = 10000
        scale = 20

        
    
    
    


    
    #get data
    path = os.getcwd() 
    data_path=path+"/data"
    pair_path = "{}/{}/{}/{}_{}_train_{}_{}".format(data_path,problem,dataname,problem,dataname,pairMax,  scale)
    stoBMGraphPath = "{}/{}/{}/{}_{}".format(data_path,problem,dataname,problem,dataname)
    featurePath = "{}/{}/{}/features/{}_{}".format(data_path,problem,dataname,featureGenMethod,maxFeatureNum)
    logpath=path+"/log/bm_"+dataname
    
    
    X_train, Y_train, X_test, Y_test = BM_Utils.getDataTrainTestRandom(pair_path,trainNum,testNum, pairMax)

    print("data fetched")
    
    Utils.writeToFile(logpath, "data fetched")
    

    featureIndexes = []
    w=[]
    infile = open(path+preTrainPath+"/featureIndex", 'r') 
    while True:   
        line = infile.readline()
        if not line:
          infile.close()
          break;
        items = line.split()
        index = int(items[0])
        weight = float(items[1])
        featureIndexes.append(index)
        w.append(weight)

    featureNum=len(featureIndexes)

    instance = BM_InputInstance(stoBMGraphPath, featurePath, featureNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    

    #**************************OneSlackSSVM
    model = Model()
    model.initialize(X_train, Y_train, instance)
    
    one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             max_iter = max_iter, log = logpath)
    
    
    one_slack_svm.w=np.array(w)   
    
    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True)
    
    Utils.writeToFile(logpath, "Testing USCO Started", toconsole = True)
    Y_pred = one_slack_svm.predict(X_test, featureNum)

    instance.test(X_test, Y_test,  Y_pred, logpath)
    

    Utils.writeToFile(logpath, dataname, toconsole = True)

    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True)
    
if __name__ == "__main__":
    main()