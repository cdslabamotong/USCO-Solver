"""
==============================
BM random method
# License: BSD 3-clause
==============================
"""

from BM import (BM_Utils, BM_InputInstance)
from basic_Utils import Utils
import argparse
import os


class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataname',  default='norm_30_128', 
                        choices=['cora', 'yahoo', 'ny'])

    
    
    parser.add_argument(
        '--featureNum', type=int, default=0,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniform', \
            choices=['uniform','true', ], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=8, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=640, help='number of testing data')   
     
     
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

    
    #parameter used in SVM
    C = 0.0001

    
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    
    

        
    if dataname == "norm_30_128":
        maxFeatureNum = 1000
        pairMax = 1000
        scale = 20

        
    
    
    

    preTrainPathResult = None

    
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
    

    instance = BM_InputInstance(stoBMGraphPath, featurePath, featureNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    

    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Testing Random Started", toconsole = True,preTrainPathResult = preTrainPathResult)

    Y_pred = BM_Utils.random_prediction(X_test)

    instance.test(X_test, Y_test,  Y_pred, logpath, preTrainPathResult = preTrainPathResult)
    
    
    

    Utils.writeToFile(logpath, dataname, toconsole = True,preTrainPathResult = preTrainPathResult)

    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True,preTrainPathResult = preTrainPathResult)

    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()