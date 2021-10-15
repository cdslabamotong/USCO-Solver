"""
==============================
ssp random
# License: BSD 3-clause
==============================
"""


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
                        choices=['kro', 'col', 'ny'])
    
    
    
    parser.add_argument(
        '--featureNum', type=int, default=160,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniform', \
            choices=['uniform','true', 'weibull55','WC_Weibull_structure'], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=160, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=1000, help='number of testing data')   
    
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
    #vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    testBatch =args.testBatch
    #pairMax=2500
    
    thread = args.thread

    
    #parameter used in SVM
    C = 0.0001
    
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
    
    

    
    preTrainPathResult = None

    
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
    '''
    for x, y in zip (X_train, Y_train):
        print(x)
        print(y)
        print()
    '''  
    instance = SSP_InputInstance(stoGraphPath, featurePath, featureNum, vNum, 
                             featureRandom = True, maxFeatureNum = maxFeatureNum,
                             thread = thread)
    
    
    #sys.exit("stop")
    #**************************OneSlackSSVM
    model = Model()
    model.initialize(X_train, Y_train, instance)
    

    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Testing random Started", toconsole = True,preTrainPathResult = preTrainPathResult)
    Y_pred = SSP_Utils.random_prediction(instance.stoGraph.unitGraph.nodes, X_train, Y_train, X_test, n_jobs = thread)
    instance.test_batch(X_test, Y_test_length,  Y_pred, testBatch, testNum, logpath, preTrainPathResult = preTrainPathResult)


   
    Utils.writeToFile(logpath, dataname, toconsole = True,preTrainPathResult = preTrainPathResult)
    #print(dataname)
    
    Utils.writeToFile(logpath, "featureNum:{}, featureGenMethod: {}, c:{} ".format(featureNum, featureGenMethod, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum:{}, testNum:{} ".format(trainNum, testNum), toconsole = True,preTrainPathResult = preTrainPathResult)

    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()