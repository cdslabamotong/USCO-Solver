"""
==============================
ssc random method
# License: BSD 3-clause
==============================
"""

import numpy as np
from DR_one_slack_ssvm import OneSlackSSVM
from DR import (DR_Utils, DR_InputInstance)
from basic_Utils import (Utils)
from basic_USCO import (Model)
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
        '--dataname',  default='yahoo', 
                        choices=['cora', 'yahoo', 'ny'])
    #parser.add_argument(
    #    '--vNum', type=int, default=768, choices=[1024,768,512],
    #                    help='kro 1024, power768 768, ER512 512')
    
    
    parser.add_argument(
        '--featureNum', type=int, default=16,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniform', \
            choices=['uniform','true', ], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=8, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=640, help='number of testing data')   
    
    #parser.add_argument(
    #    '--testBatch', type=int, default=1, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default=None,  help='if store a pre_train model')
    
    
    args = parser.parse_args()
    #utils= Utils()
    
    problem ="dr"
    
    dataname=args.dataname
    #vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    #testBatch =args.testBatch
    #pairMax=2500
    
    thread = args.thread
    verbose=6
    
    #parameter used in SVM
    C = 0.0001
    tol=0.001
    max_iter = 2
    
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    
    
    
    if dataname == "cora":
        maxFeatureNum = 10000
        pairMax = 10000
        fraction = 0.1
        scale = 200
        #vNum=1024
        
    if dataname == "yahoo":
        maxFeatureNum = 10000
        pairMax = 10000
        fraction = 0.1
        scale = 200
        
    if dataname == "ny":
        pairMax = 395265   
        vNum = 768
    
    
    

    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=path+"/data"
    pair_path = "{}/{}/{}/{}_{}_train_{}_{}_{}".format(data_path,problem,dataname,problem,dataname,pairMax, fraction, scale)
    #unitpair_path = "{}/{}/{}/{}_{}_unitAll".format(data_path,problem,dataname,problem,dataname)
    stoCoverGraphPath = "{}/{}/{}/{}_{}".format(data_path,problem,dataname,problem,dataname)
    featurePath = "{}/{}/{}/features/{}_{}".format(data_path,problem,dataname,featureGenMethod,maxFeatureNum)
    #if args.log_paht is not None:
    logpath=path+"/log/dr_random_"+dataname
    #print(data_path)
    #print(pair_path)
    #print(stoGraphPath)
    #print(featurePath)
    
    #sys.exit("stop")
    
    X_train, Y_train, X_test, Y_test = DR_Utils.getDataTrainTestRandom(pair_path,trainNum,testNum, pairMax)
    #print(X_train)
    print("data fetched")
    #sys.exit()
    Utils.writeToFile(logpath, "data fetched")
    '''
    for x, y in zip (X_train, Y_train):
        print(x)
        print(y)
        print()
    '''  
    instance = DR_InputInstance(stoCoverGraphPath, featurePath, featureNum, fraction,
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    
    #sys.exit("stop")
    #**************************OneSlackSSVM
    #model = Model()
    #model.initialize(X_train, Y_train, instance)
    
    #one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             #max_iter = max_iter, log = logpath)
    
    
    #one_slack_svm.fit(X_train, Y_train, initialize = False)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Testing Random Started", toconsole = True,preTrainPathResult = preTrainPathResult)
    Y_pred = DR_Utils.random_prediction(instance, X_test, featureNum)
    #sys.exit("stop")
    instance.test(X_test, Y_test,  Y_pred, logpath)
    
    
    


    #print("Prediction Started")
    #Utils.writeToFile(logpath, "Prediction Started", toconsole = True)
    #Y_pred = one_slack_svm.predict(X_test, featureNum)
    #sys.exit("stop")
    #print("Testing Started")
    #Utils.writeToFile(logpath, "Prediction Started", toconsole = True)
    #instance.test(X_test, Y_test_length, Y_pred,logpath)
    
    
    
    #print("All One Started")
    #instance.testUnitAllPair(pair_path,unitpair_path)
    '''
    Y_allOne_pred = [];
    for x in X_test:
        Y_allOne_pred.append(instance.inferenceBasic(x))
    print("AllOne Testing Started")
    instance.test(X_test, Y_test_length, Y_allOne_pred)
    '''
    Utils.writeToFile(logpath, dataname, toconsole = True,preTrainPathResult = preTrainPathResult)
    #print(dataname)
    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    #print("featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C))
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True,preTrainPathResult = preTrainPathResult)
    #print("trainNum:{}, testNum:{} ".format(trainNum, testNum))
    #print("loss_type:{}, LAI_method:{}, ".format(loss_type.name, LAI_method))
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()