"""
==============================
ssp USCO-Solver pretrain
# License: BSD 3-clause
==============================
"""

import numpy as np
from SSP_one_slack_ssvm import OneSlackSSVM
from SSP import (SSP_Utils, SSP_InputInstance)
from basic_Utils import (Utils)
from basic_USCO import (Model)
import argparse
import os


class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataname',  default='col', 
                        choices=['kro', 'ny', 'col'])
    parser.add_argument(
        '--vNum', type=int, default=512, choices=[1024,768,512],
                        help='kro 1024, ny 768, col 512')
    
    
   
    parser.add_argument(
        '--featureGenMethod', default='true', \
            choices=['uniform','true',], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
    parser.add_argument(
        '--preTrainPath', default="/pre_train/col/col_true_6400_160_6400/", 
                        help='number of features (random subgraphs) used in StratLearn ')
    
    parser.add_argument(
        '--trainNum', type=int, default=0, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=10, help='number of testing data')   
    
    parser.add_argument(
        '--testBatch', type=int, default=5, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
   
    
    
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
    
    #featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    

    if dataname == "col":
        maxFeatureNum = 10000
        pairMax = 137196
    
    

    
    preTrainPath = args.preTrainPath
        

    
    #get data
    path = os.getcwd() 
    data_path=path+"/data"
    pair_path = "{}/{}/{}/{}_{}_trainAllShuffle".format(data_path,problem,dataname,problem,dataname)
    stoGraphPath = "{}/{}/{}/{}_{}".format(data_path,problem,dataname,problem,dataname)
    featurePath = "{}/{}/{}/features/{}_{}".format(data_path,problem,dataname,featureGenMethod,maxFeatureNum)
    logpath=path+"/log"

    
    X_train, Y_train, Y_train_length, X_test, Y_test, Y_test_length = SSP_Utils.getDataTrainTestRandom(pair_path,trainNum,testNum*testBatch, pairMax)
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
    
    instance = SSP_InputInstance(stoGraphPath, featurePath, featureNum, vNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread,indexes=featureIndexes)
    
    
    #sys.exit("stop")
    #**************************OneSlackSSVM
    model = Model()
    model.initialize(X_train, Y_train, instance)
    
    one_slack_svm = OneSlackSSVM(model, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             max_iter = max_iter, log = logpath)
    

    
    
    one_slack_svm.w=np.array(w)   
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True)
    
    Utils.writeToFile(logpath, "Testing USCO Started", toconsole = True)
    Y_pred = one_slack_svm.predict(X_test, featureNum)
    instance.test_batch(X_test, Y_test_length,  Y_pred, testBatch, testNum, logpath)



    Utils.writeToFile(logpath, dataname, toconsole = True)
    #print(dataname)
    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True)
    
if __name__ == "__main__":
    main()