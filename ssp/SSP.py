# -*- coding: utf-8 -*-
"""
ssp utilities
# License: BSD 3-clause
"""
import numpy as np
import sys
import math
import random
import time
import copy
import multiprocessing
from basic_Utils import Utils
#from base import StructuredModel

class SSP_Utils(object):
    @staticmethod
    def getWeibull(alpha, beta):
            time = alpha*math.pow(-math.log(1-random.uniform(0, 1)), beta);
            if time >= 0:
                return math.ceil(time)+1
            else:
                sys.exit("time <0") 
                return None
    
            
    @staticmethod        
    def getDataTrainTestRandom(path, trainNum, testNum, Max):
        lineNums=(np.random.permutation(Max))[0:(trainNum+testNum)] 
        trainLineNums=lineNums[0:trainNum]
        #print(trainLineNums)
        testLineNums=lineNums[trainNum:(trainNum+testNum)]
        #print(testLineNums)
        
        #lineNums.sort()
        file1 = open(path, 'r') 
        lineNum = 0
        X_train, Y_train, Y_train_length = ([] for i in range(3))
        X_test, Y_test, Y_test_length= ([] for i in range(3))
        
        while True:
            line = file1.readline() 
            if not line: 
                break 
            strings=line.split()
            if lineNum in trainLineNums:
                #print(str(lineNum))
                #print(trainLineNums)
                x = [strings[0],strings[1]]
                #print(x)
                X_train.append(x)
                y=strings[3:]
                Y_train.append(y);
                Y_train_length.append(float(strings[2]))
            #print("===============================")    
            if lineNum in testLineNums:
                x = [strings[0],strings[1]]
                X_test.append(x)
                y=strings[3:]
                Y_test.append(y);
                Y_test_length.append(float(strings[2]))
            
            lineNum += 1   
        
                
        '''
        while len(lineNums)>0:
            line = file1.readline() 
            if not line: 
                break 
            strings=line.split()
            if lineNum != lineNums[0] or strings[0]==strings[1]:
                lineNum += 1 
            else:
                if(len(Y_train)<trainNum):
                    
                    x = [strings[0],strings[1]]
                    print(x)
                    X_train.append(x)
                    y=strings[3:]
                    Y_train.append(y);
                    Y_train_length.append(float(strings[2]))
                    lineNum += 1   
                    lineNums=np.delete(lineNums, 0)
                    #print(Y_train)
                    #print("train++", len(Y_train),len(lineNums))
                else:
                    x = [strings[0],strings[1]]
                    X_test.append(x)
                    y=strings[3:]
                    Y_test.append(y);
                    Y_test_length.append(float(strings[2]))
                    lineNum += 1   
                    lineNums=np.delete(lineNums, 0)
                    #print("test++ {}"+format(len(lineNums)))
        '''
        file1.close()    
        if (len(X_train) != trainNum) or (len(Y_test) != testNum):
            sys.exit("getDataRandom: data fetch failed with sizes: X_train {} Y_test {}".format(
                len(X_train),len(Y_test))) 
        return X_train, Y_train, Y_train_length, X_test, Y_test, Y_test_length
    
    def removeuu(oldpath, newpath):
        with open(newpath, 'w') as outfile:
            infile = open(oldpath, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                node1 = ints[0]
                node2 = ints[1]
                if node1 != node2:
                    outfile.write(line)     
        outfile.close()    
    
    def NB_prediction(unitGraph, X_train, Y_train,  X_test, n_jobs = 1):
        freMatrix={}
        for node1 in unitGraph.nodes:
            freMatrix[node1]={}
            for node2 in unitGraph.nodes:
                freMatrix[node1][node2]=0
        for x, y in zip(X_train, Y_train):
            for node in y:
                freMatrix[x[0]][node] = freMatrix[x[0]][node]+1
                freMatrix[x[1]][node] = freMatrix[x[1]][node]+1
        
               
        Y_pred = []
        p = multiprocessing.Pool(n_jobs)
        Y_pred = p.starmap(SSP_Utils.NB_prediction_pair, ((unitGraph, freMatrix, x) for x  in X_test))
        p.close()
        p.join()
        
         
        return Y_pred      
    
                
        #return Y_pred
            
    def NB_prediction_pair(unitGraph, freMatrix, x):
        #print("NB_prediction_pair")
        nodeScore = {}
        for node in unitGraph.nodes:
            #print(node)
            #print(x[0])
            nodeScore[node] = freMatrix[x[0]][node]+freMatrix[x[1]][node]
        sortedNodeScore = sorted(nodeScore.items(), key=lambda kv: kv[1],reverse=True)
        sortedNodes = []
        for item in sortedNodeScore:
            sortedNodes.append(item[0])
            
        return SSP_Utils.prediction_from_sortedNodes_ver1(unitGraph, x, sortedNodes)
        
    
        '''
        sortedNodeScore = sorted(nodeScore.items(), key=lambda kv: kv[1],reverse=True)
        sortedNodes = []
        for item in sortedNodeScore:
            sortedNodes.append(item[0])
        
        #y_pred= []
        c_nodes=copy.deepcopy(nodes)
        for node in nodes:
            c_nodes[node].neighbor={}
            
        c_nodes[x[0]].neighbor=nodes[x[0]].neighbor
        c_nodes[x[1]].neighbor=nodes[x[1]].neighbor     
        count = 0
        for node in sortedNodes:
            c_nodes[node].neighbor=nodes[node].neighbor
            count +=1
            if count>len(sortedNodes)*0.75:
                break;
            #c_graph.print_()
        c_graph = Graph(len(c_nodes), c_nodes)
        if c_graph.isConnected(x[0], x[1]):
            result = Dijkstra.dijkstra(c_graph, x[0])
            if x[1] not in result:
                sys.exit("connected but no path")
            print(x)
            print(result[x[1]][3:])
            return result[x[1]][3:]
        sys.exit("not path found")
        '''
    
    #def prediction_weightedList(nodes, x, sortedList):
        
        
    def   random_prediction(nodes, X_train, Y_train,  X_test, n_jobs = 1):    
        
       # min_weight=0;
        #nodes  =  copy.deepcopy(self.stoGraph.nodes) 
        for node in nodes:
            for tonode in nodes[node].neighbor:
                nodes[node].neighbor[tonode]=random.random()
          
        comGraph=Graph(len(nodes), nodes)         
        
        if n_jobs == 1:
            result =[]
            for x in X_test:
                result.append(Dijkstra.dijkstra_1(comGraph, x[0])[1][x[1]][3:])
            print("inference block DONE")
            return result
        else:
            #print("111")
            results={}
            p = multiprocessing.Pool(n_jobs)
            #print("222")
            resultsPair=p.starmap(Dijkstra.dijkstra_1, ((comGraph, node) for node in nodes))
            #print("333")
            p.close()
            p.join()
            for pair in resultsPair:
                results[pair[0]]=pair[1];
            result =[]
            for x in X_test:
                result.append(results[x[0]][x[1]][3:])
            print("inference block DONE")
            return result        
    
    def prediction_from_sortedDic_batch(nodes, X_test, sortedDic, n_jobs = 1):
        for node in nodes:
           for tonode in nodes[node].neighbor:
               nodes[node].neighbor[tonode]=math.exp(-sortedDic[node])
               
        comGraph=Graph(len(nodes), nodes)         
                
        if n_jobs == 1:
            Y_pred =[]
            for x in X_test:
                Y_pred.append(Dijkstra.dijkstra_1(comGraph, x[0])[1][x[1]][3:])
            print("inference block DONE")
            return Y_pred
        else:
            #print("111")
            results={}
            p = multiprocessing.Pool(n_jobs)
            #print("222")
            resultsPair=p.starmap(Dijkstra.dijkstra_1, ((comGraph, node) for node in nodes))
            #print("333")
            p.close()
            p.join()
            for pair in resultsPair:
                results[pair[0]]=pair[1];
            Y_pred =[]
            for x in X_test:
                Y_pred.append(results[x[0]][x[1]][3:])
            print("inference block DONE")
            return Y_pred   
        
    def prediction_from_sortedDic(nodes, x, sortedDic, n_jobs = 1):
        for node in nodes:
           for tonode in nodes[node].neighbor:
               print(sortedDic[node])
               nodes[node].neighbor[tonode]=math.exp(-sortedDic[node])
               
        comGraph=Graph(len(nodes), nodes)         
        results = Dijkstra.dijkstra_1(comGraph, x)
        return results[x[0]][x[1]][3:]
    
    
    def prediction_from_sortedNodes(unitGraph, x, sortedNodes, n_jobs = 1):
        path = []
        path.append(x[0])
        for node in sortedNodes:
            path.append(node)
        path.append(x[1]) 
        
        pred = []
        pred.append(path[0])
        for i in range(len(path)-1):
            print(path[i])
            if path[i+1] in unitGraph.nodes[path[i]].neighbor:
                pred.append(path[i+1])
            else:
                result = Dijkstra.dijkstra(unitGraph, path[i])
                if path[i+1] in result:
                    for node in result[path[i+1]][4:]:
                        pred.append(node)
                else:
                    return None
        
        
        return pred
    
    def prediction_from_sortedNodes_ver1(unitGraph, x, sortedNodes, n_jobs = 1):
        nodes = unitGraph.nodes
        c_nodes=copy.deepcopy(nodes)
        for node in nodes:
            c_nodes[node].neighbor={}
            
        c_nodes[x[0]].neighbor=nodes[x[0]].neighbor
        c_nodes[x[1]].neighbor=nodes[x[1]].neighbor     
        count = 0
        for node in sortedNodes:
            c_nodes[node].neighbor=nodes[node].neighbor
            count +=1
            if count>len(sortedNodes)*0.75:
                break;
            #c_graph.print_()
            
        c_graph = Graph(len(c_nodes), c_nodes)
        if c_graph.isConnected(x[0], x[1]):
            result = Dijkstra.dijkstra(c_graph, x[0])
            if x[1] not in result:
                sys.exit("connected but no path")
            #print(x)
            #print(result[x[1]][3:])
            return result[x[1]][3:]
        #print("prediction_from_sortedNodes_ver1 not path found")
        
        return None
    
    
    def prediction_from_sortedNodes_ver2(unitGraph, x, sortedNodes, n_jobs = 1):
        nodes = unitGraph.nodes
        c_nodes=copy.deepcopy(nodes)
        for node in nodes:
            c_nodes[node].neighbor={}
            
        c_nodes[x[0]].neighbor=nodes[x[0]].neighbor
        c_nodes[x[1]].neighbor=nodes[x[1]].neighbor     
        count = 0
        for node in sortedNodes:
            c_nodes[node].neighbor=nodes[node].neighbor
            count +=1
            if count>len(sortedNodes)*0.25:
                c_graph = Graph(len(c_nodes), c_nodes)
                if c_graph.isConnected(x[0], x[1]):
                    result = Dijkstra.dijkstra(c_graph, x[0])
                    if x[1] not in result:
                        sys.exit("connected but no path")
                    return result[x[1]][3:]
                        
            if count>len(sortedNodes)*0.50 and count <= len(sortedNodes)*0.25:
                c_graph = Graph(len(c_nodes), c_nodes)
                if c_graph.isConnected(x[0], x[1]):
                    result = Dijkstra.dijkstra(c_graph, x[0])
                    if x[1] not in result:
                        sys.exit("connected but no path")  
                    return result[x[1]][3:]
            
            if count>len(sortedNodes)*0.80 and count <= len(sortedNodes)*0.50:
               c_graph = Graph(len(c_nodes), c_nodes)
               if c_graph.isConnected(x[0], x[1]):
                   result = Dijkstra.dijkstra(c_graph, x[0])
                   if x[1] not in result:
                       sys.exit("connected but no path")  
                   return result[x[1]][3:]
            
            if count>len(sortedNodes)*0.90 and count <= len(sortedNodes)*0.80:
               c_graph = Graph(len(c_nodes), c_nodes)
               if c_graph.isConnected(x[0], x[1]):
                   result = Dijkstra.dijkstra(c_graph, x[0])
                   if x[1] not in result:
                       sys.exit("connected but no path")  
                   return result[x[1]][3:]
            
            if count>len(sortedNodes)*0.99 and count <= len(sortedNodes)*0.90:
                c_graph = Graph(len(c_nodes), c_nodes)
                if c_graph.isConnected(x[0], x[1]):
                    result = Dijkstra.dijkstra(c_graph, x[0])
                    if x[1] not in result:
                        sys.exit("connected but no path")  
                    return result[x[1]][3:]
                else:
                    return None
            #print(x)
            #print(result[x[1]][3:])
            
            #c_graph.print_()
            
        
        #print("prediction_from_sortedNodes_ver1 not path found")
        
        '''
        sortedNodeScore = sorted(nodeScore.items(), key=lambda kv: kv[1],reverse=True)
        sortedNodes = []
        for item in sortedNodeScore:
            sortedNodes.append(item[0])
        
        #y_pred= []
        c_nodes=copy.deepcopy(nodes)
        for node in nodes:
            c_nodes[node].neighbor={}
            
        c_nodes[x[0]].neighbor=nodes[x[0]].neighbor
        c_nodes[x[1]].neighbor=nodes[x[1]].neighbor     
        count = 0
        for node in sortedNodes:
            c_nodes[node].neighbor=nodes[node].neighbor
            count +=1
            if count>len(sortedNodes)*0.75:
                break;
            #c_graph.print_()
        c_graph = Graph(len(c_nodes), c_nodes)
        if c_graph.isConnected(x[0], x[1]):
            result = Dijkstra.dijkstra(c_graph, x[0])
            if x[1] not in result:
                sys.exit("connected but no path")
            print(x)
            print(result[x[1]][3:])
            return result[x[1]][3:]
        sys.exit("not path found")
        '''
        return None
    
    def read_NY(inpath, outpath, vnum, times):
        with open(outpath, 'w') as outfile:
            infile = open(inpath, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                if ints[0] == "a" and int(ints[1]) < vnum and int(ints[2]) < vnum:
                    node1 = ints[1]
                    node2 = ints[2]
                    alpha = math.ceil(10*random.uniform(0, 1))
                    beta = math.ceil(10*random.uniform(0, 1))
                    mean = 0
                    for i in range(times):
                        mean +=  SSP_Utils.getWeibull(float(alpha), float(beta))
                    mean = mean / times
                   
                    outfile.write(node1+" ") 
                    outfile.write(node2+" ")
                    outfile.write(str(alpha)+" ") 
                    outfile.write(str(beta)+" ") 
                    outfile.write(str(mean) + "\n") 
        outfile.close()    
        
       
        
       
    '''
    def random_prediction_pair(nodes, x):
        print("NB_prediction_pair")
        sortedNodes = []
        for node in nodes:
            sortedNodes.append(node)
            
        random.shuffle(sortedNodes)
        
        c_nodes=copy.deepcopy(nodes)
        for node in nodes:
            c_nodes[node].neighbor={}
            
        c_nodes[x[0]].neighbor=nodes[x[0]].neighbor
        c_nodes[x[1]].neighbor=nodes[x[1]].neighbor     
        
        for node in sortedNodes:
            c_nodes[node].neighbor=nodes[node].neighbor
            c_graph = Graph(len(c_nodes), c_nodes)
            if c_graph.isConnected(x[0], x[1]):
                result = Dijkstra.dijkstra(c_graph, x[0])
                if x[1] not in result:
                    sys.exit("connected but no path")
                print(x)
                print(result[x[1]][3:])
                return result[x[1]][3:]
            #c_graph.print_()
            
        sys.exit("not path found")
        
    
    def random_prediction_pair_1(nodes, x):
        #print("NB_prediction_pair")
        sortedNodes = []
        for node in nodes:
            sortedNodes.append(node)
            
        random.shuffle(sortedNodes)
        
        c_nodes=copy.deepcopy(nodes)
        c_graph = Graph(len(c_nodes), c_nodes)
        
        path = []
        path.append(x[0])
        for node in sortedNodes:
            #new_path = path.copy()
            path.append(node)
            path.append(x[1])
            length = c_graph.pathLength(x, path)
            if length > 0:
                return path.copy()
            path.pop()

        sys.exit("no path found")
    
    
    def random_path(nodes, x):
        c_node = x[0]
        path = []
        path.append(x[0])
        while c_node != x[1]:
            available_keys = []
            for tonode in nodes[c_node].neighbor:
                if tonode not in path:
                    available_keys.append(tonode)
            if len(available_keys) == 0:
                return []
            tonode =random.choice(available_keys)
            path.append(tonode)
            c_node=tonode
        path.append(c_node)
        return path
    '''
                
'''
def computeScore(self, x, y, w)
def computeFeature(self, x, y)
def computeScoreOneFeature(self, x, y, featureIndex)
def inference(self, x, w)
def inference_block(self, X, w)
def inferenceBasic(self,x)
def loss(self, y, y_hat)
def loss_augmented_inference(self, x, y ,w, comGraph=None, results = None)
def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs=1)
def test(self, X_test, Y_length, Y_pred, logpath= None)
'''
    
class SSP_InputInstance(object):
    
    def __init__(self, stoGraphPath, featurePath, featureNum, vNum, loss_type = "hamming", 
                 featureRandom = False, maxFeatureNum = 500, thread = 1, indexes=None):
        #self.graphs=[];
        self.featureNum = featureNum
        self.vNum = vNum
        self.stoGraph = StoGraph(stoGraphPath, vNum);
        self.edgeVectors= {};
        #self.socialGraph.print()
        #print(self.socialGraph.nodes)
        self.loss_type=loss_type
        self.n_jobs=thread
        self.featureIndicator=[]
        self.featureIndicator_reset(1)
        #if loss_type != None:
        #    self.loss_type=loss_type.name
         #   if loss_type.name == "hamming": 
         #       self.hammingWeight=loss_type.weight
        #self.hammingWeight = None
        self.featureRandom = featureRandom
        self.maxFeatureNum = maxFeatureNum
        
        # self.readFeatures(path, featureNum)
        # read social graph
       
        #read features
        self.graphs = [];
        if indexes is not None:
            self.featureIndexes=indexes
        else:  
            if self.featureRandom:
                lineNums=(np.random.permutation(maxFeatureNum))[0:featureNum]
                self.featureIndexes=lineNums
                #print("lineNums: {}".format(lineNums))
            else:
                for i in range(featureNum):
                    self.featureIndexes.append(i)
              
        self.edgeVectors  =  copy.deepcopy(self.stoGraph.nodes) 
        for node in self.edgeVectors:
            for tonode in self.edgeVectors[node].neighbor:
                self.edgeVectors[node].neighbor[tonode]=[]
                    
        for i in self.featureIndexes:      
            path_graph="{}/{}".format(featurePath, i)
            infile = open(path_graph, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                #print(ints)
                #print(path_graph)
                node1 = ints[0]
                node2 = ints[1]
                weight = float(ints[2])
                self.edgeVectors[node1].neighbor[node2].append(weight);
            #graph = Graph(vNum,temp_nodes)
            #self.graphs.append(graph)

    def featureIndicator_reset(self, x):
        self.featureIndicator=[]
        for i in range(self.featureNum):
            self.featureIndicator.append(x)
        
    def computeScore(self, x, y, w):
        feature = self.computeFeature(x, y)
        return w.dot(feature)
        
    def computeFeature(self, x, y):
        feature = [];
        #print(self.featureNum)
        for i in range(self.featureNum):
            if True:
                feature.append(self.computeScoreOneFeature(x, y, i))
        #print(feature)
        return feature   
    
    def computeScoreOneFeature(self, x, y, featureIndex):    
        '''compute f^g(M,P)'''
        
        length = 0
        if y[0]!= x[0] or y[-1]!=x[1]:
            #print(x)
            #print(y)
            sys.exit("path y not for x") 
            return -1;
        else:
            for j in range(len(y)-1):
                if y[j] in self.edgeVectors and y[j+1] in self.edgeVectors[y[j]].neighbor:
                    #length += math.exp(-self.edgeVectors[y[j]].neighbor[y[j+1]][featureIndex])
                    length += self.edgeVectors[y[j]].neighbor[y[j+1]][featureIndex]
                else:
                    sys.exit("edge not existing") 
                    return -1;
            return length
        
        
        #return 1/graph.pathLength(x, y)
        
    
    
    def inference(self, x, w):
        #print("inference") 
        nodes  =  copy.deepcopy(self.stoGraph.nodes) 
        for node in nodes:
            for tonode in nodes[node].neighbor:
                temp = 0
                for i in range(self.featureNum):
                    temp += self.featureIndicator[i]*w[i]*self.edgeVectors[node].neighbor[tonode][i]
                nodes[node].neighbor[tonode]=temp
        graph=Graph(self.vNum,nodes) 
        result = Dijkstra.dijkstra(graph, x[0])
        #print(x)
        #print(result[x[1]])
        return result[x[1]][3:]
    
    def batch_inference(self, X, w, n_jobs=1, offset = None):
        #print("inference") 
        print("inference block RUNNING")
        min_weight=0;
        nodes  =  copy.deepcopy(self.stoGraph.nodes) 
        for node in nodes:
            for tonode in nodes[node].neighbor:
                temp = 0
                for i in range(self.featureNum):
                    if w[i]>0:			
                        temp += self.featureIndicator[i]*w[i]*self.edgeVectors[node].neighbor[tonode][i]
                nodes[node].neighbor[tonode]=temp
                if temp < min_weight:
                    min_weight=temp
        if min_weight<0:
            print(min_weight)
            for node in nodes:
                for tonode in nodes[node].neighbor:
                    nodes[node].neighbor[tonode] += min_weight
          
        comGraph=Graph(self.vNum, nodes)         
        
        if n_jobs == 1:
            result =[]
            for x in X:
                result.append(Dijkstra.dijkstra_1(comGraph, x[0])[1][x[1]][3:])
            print("inference block DONE")
            return result
        else:
            #print("111")
            results={}
            p = multiprocessing.Pool(n_jobs)
            #print("222")
            resultsPair=p.starmap(Dijkstra.dijkstra_1, ((comGraph, node) for node in nodes))
            #print("333")
            p.close()
            p.join()
            for pair in resultsPair:
                results[pair[0]]=pair[1];
            result =[]
            for x in X:
                result.append(results[x[0]][x[1]][3:])
            print("inference block DONE")
            return result        
    
    def batch_predict(self, X, w, n_jobs=1, offset = None):
        return self.batch_inference(X, w, n_jobs=n_jobs, offset = offset)
    
    # This is a baseline treating all weight as 1
    def inferenceBasic(self,x):
        nodes  =  copy.deepcopy(self.stoGraph.nodes) 
        for node in nodes:
            for tonode in nodes[node].neighbor:
                nodes[node].neighbor[tonode]=1
                
        graph=Graph(self.vNum,nodes) 
        result = Dijkstra.dijkstra(graph, x[0])
        return result[x[1]][3:]
    
    
    def loss(self, y, y_hat):
        if self.loss_type == None:
            sys.exit("loss method not speficied.") 
        if self.loss_type == "area":
            return self.similarity(y, y)-self.similarity(y, y_hat)
        if self.loss_type == "hamming":
            if y == y_hat:
                return 0
            else:
                return 1


    
    
    def loss_augmented_inference(self, x, y ,w, comGraph=None, results = None):
        if results != None:
            if x[0] in results:
                return float(results[x[0]][x[1]][2])
            else:
                if comGraph== None:
                    sys.exit("no graph given, but result not None") 
                else:
                    results[x[0]]=Dijkstra.dijkstra(comGraph, x[0])
                    return float(results[x[0]][x[1]][2])
        if results == None:
            return self.inference(x, w)

            #sys.exit("loss_augmented_inference method not speficied.") 
    
    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs=1):
        print("loss_augmented_inference block RUNNING")
        result=self.batch_inference(X, w, n_jobs)

        print("loss_augmented_inference block Done")
        return result
      
    def test_batch(self, X_test, Y_test_length, Y_pred, testBatch, testNum, logpath = None, preTrainPathResult = None):  
        avg_ratios = []
        avg_fail_ratios = []
        #print(testNum)
        #print(testBatch)
        for i in range(testBatch):
            #print(i*testNum)
            #print((i+1)*testNum)
            ratio, fail_ratio = self.test(X_test[i*testNum:(i+1)*testNum], Y_test_length[i*testNum:(i+1)*testNum], Y_pred[i*testNum:(i+1)*testNum], logpath= logpath,preTrainPathResult = preTrainPathResult)
            avg_ratios.append(ratio)
            avg_fail_ratios.append(fail_ratio)
            
        mean_avg_ratios=np.mean(np.array(avg_ratios))
        std_avg_ratios=np.std(np.array(avg_ratios))
        
        mean_avg_fail_ratios=np.mean(np.array(avg_fail_ratios))
        std_avg_fail_ratios=np.std(np.array(avg_fail_ratios))
        
        
        output = "mean_avg_ratios: "+ Utils.formatFloat(mean_avg_ratios)
        Utils.writeToFile(logpath, output, toconsole = True, preTrainPathResult = preTrainPathResult)
        
        output = " std_avg_ratios: "+ Utils.formatFloat(std_avg_ratios)
        Utils.writeToFile(logpath, output, toconsole = True, preTrainPathResult = preTrainPathResult)
        
        output = "mean_avg_fail_ratios: "+ Utils.formatFloat(mean_avg_fail_ratios)
        Utils.writeToFile(logpath, output, toconsole = True, preTrainPathResult = preTrainPathResult)
        
        output = " std_avg_fail_ratios: "+ Utils.formatFloat(std_avg_fail_ratios)
        Utils.writeToFile(logpath, output, toconsole = True, preTrainPathResult = preTrainPathResult)

    def test(self, X_test, Y_length, Y_pred, logpath= None, preTrainPathResult = None):
        Y_hat_length = []
        #print(X_test)
        for x , y in zip(X_test,  Y_pred):
            Y_hat_length.append(self.stoGraph.EGraph.pathLength(x,y))
        Y_true_total=0
        Y_pred_total=0
        count_success = 0
        count_fail = 0
        for y,y_hat in zip(Y_length,Y_hat_length):
            if y_hat is None:
                count_fail += 1
            else:
                count_success += 1
                Y_true_total += y
                Y_pred_total += y_hat
            #print("{} {}".format(y, y_hat))
        #Utils.writeToFile(logpath, "{} {}".format(Y_true_total, Y_pred_total), toconsole = True)
        #print("{} {}".format(Y_true_total, Y_pred_total))
        #Utils.writeToFile(logpath, "{} {}".format(Y_true_total/len(X_test), Y_pred_total/(len(X_test))), toconsole = True)
        #print("{} {}".format(Y_true_total/len(X_test), Y_pred_total/(len(X_test))))
        #Utils.writeToFile(logpath, str(Y_pred_total/Y_true_total), toconsole = True)
        
        output = "Y_pred_total/Y_true_total: "+ Utils.formatFloat(Y_pred_total/Y_true_total)
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        output = "fail ratio: "+ Utils.formatFloat(count_fail/(count_fail+count_success))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        #print(Y_pred_total/Y_true_total);
        return Y_pred_total/Y_true_total, count_fail/(count_fail+count_success)
        
    def testUnitAllPair(self, truepath, unitpath):
        true = open(truepath, 'r')
        unit = open(unitpath, 'r')
        num = 0
        Y_true_total=0
        Y_pred_total=0

        while True:
            line_true = true.readline() 
            line_unit = unit.readline() 
            if not line_true: 
                break 
            true_strings=line_true.split()
            unit_strings=line_unit.split()
            if true_strings[0]!=unit_strings[0] or true_strings[1]!=unit_strings[1]:
                sys.exit("testAllUnitPair wrong")
            num += 1
            Y_true_total += float(true_strings[2])
            x = [unit_strings[0],unit_strings[1]]
            y_hat = unit_strings[3:]
            #print(x)
            #print(y_hat)
            Y_pred_total += self.stoGraph.EGraph.pathLength(x,y_hat)
        
                    #print("test++ {}"+format(len(lineNums)))
        true.close()   
        unit.close()
        print("{} {}".format(Y_true_total, Y_pred_total))
        print("{} {}".format(Y_true_total/num, Y_pred_total/num))
        print(Y_pred_total/Y_true_total);
        
           
class StoGraph(object):  
    class Node(object):
        def __init__(self,index):
            self.index = index
            self.neighbor = {}
            self.in_degree = 0
            self.out_degree = 0
        def print(self):
            print(self.index)
            for node in self.neighbor:
                print("{} {} {} {}".format(str(self.index), str(node) , str(self.neighbor[node][0]), str(self.neighbor[node][1])))       
    def __init__(self, path, vNum):
        self.nodes={}
        self.vNum = vNum
        self.EGraph = None
        
        for v in range(self.vNum):
             node = self.Node(str(v))
             node.neighbor={}
             self.nodes[str(v)]=node
             
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            alpha = ints[2]
            beta = ints[3]
            mean = float(ints[4]) # mean of Chi-square Gaussian
            # para_2 = float(ints[3])
            
            if node1 in self.nodes:
                self.nodes[node1].neighbor[node2]=[alpha,beta, mean]
                self.nodes[node1].out_degree += 1
                self.nodes[node2].in_degree += 1
            else:
                sys.exit("non existing node") 
                
            if node2 not in self.nodes:
                sys.exit("non existing node") 
        
        #create mean graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=self.nodes[node].neighbor[tonode][2];
        
        self.EGraph = Graph(self.vNum, temp_nodes)
        
        #create unit graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=1;
        
        self.unitGraph = Graph(self.vNum, temp_nodes)
        
        #create random graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=random.random()
        self.randomGraph = Graph(self.vNum, temp_nodes)

    def EgraphShortest(self, source, destination = None):
        return Dijkstra.dijkstra(self.EGraph,source)
        
    def genMultiRealization(self, num, outfolder, rType, startIndex=0):
        for cout in range(num):
            print(cout)
            path = "{}{}".format(outfolder, cout+startIndex)
            if rType=="true":
                self.genOneRealizationTrue(path)
            if rType=="100000uniform":
                self.genOneRealizationUniform(path)
            if rType=="uniform_1":
                self.genOneRealizationUniform_1(path)
            if rType=="weibull55":
                self.genOneRealization_weibull55(path)
            if rType=="gau":
                self.genOneRealizationGau(path)
                
            
    def genOneRealizationTrue(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for tonode in self.nodes[node].neighbor:
                    alpha = self.nodes[node].neighbor[tonode][0]
                    beta = self.nodes[node].neighbor[tonode][1]
                    weight = SSP_Utils.getWeibull(float(alpha), float(beta))
                    outfile.write(node+" ") 
                    outfile.write(tonode+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close()
        
    
    def genOneRealizationUniform(self, outpath):
        temp_nodes  =  copy.deepcopy(self.nodes) 
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for tonode in self.nodes[node].neighbor:
                    weight = math.ceil(100000*random.uniform(0, 1));
                    temp_nodes[node].neighbor[tonode]=weight
                    outfile.write(node+" ") 
                    outfile.write(tonode+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close() 
    
    def genOneRealizationUniform_1(self, outpath):
        choices = [1,1,1,1,1,1,1,1,5,5,5,5,10,10,10,10,100,100,1000,1000,100000]
        temp_nodes  =  copy.deepcopy(self.nodes) 
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for tonode in self.nodes[node].neighbor:
                    weight = choices[math.floor(len(choices)*random.uniform(0, 1))]
                    temp_nodes[node].neighbor[tonode]=weight
                    outfile.write(node+" ") 
                    outfile.write(tonode+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close() 
        
    def genOneRealization_weibull55(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for tonode in self.nodes[node].neighbor:
                    #alpha = self.nodes[node].neighbor[tonode][0]
                    #beta = self.nodes[node].neighbor[tonode][1]
                    weight = SSP_Utils.getWeibull(5, 5)
                    outfile.write(node+" ") 
                    outfile.write(tonode+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close()
    
    def genOneRealizationGau(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for tonode in self.nodes[node].neighbor:
                    #alpha = self.nodes[node].neighbor[tonode][0]
                    #beta = self.nodes[node].neighbor[tonode][1]
                    weight = random.normalvariate(0, 1)
                    outfile.write(node+" ") 
                    outfile.write(tonode+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close()
        
   
        
    def genAllTrainPairs(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                result = Dijkstra.dijkstra(self.EGraph, node);
                for tonode in self.nodes:
                    if tonode in result:
                        string = "";
                        for i in result[tonode]:
                            #print(" "+i)
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0]!=result[tonode][1]):
                            outfile.write(string) 
                print(node)
        outfile.close()
        
    def genAllTrainPairsShuffle(self, outpath):
        strings = []        
        for node in self.nodes:
                result = Dijkstra.dijkstra(self.EGraph, node);
                for tonode in self.nodes:
                    if tonode in result:
                        string = "";
                        for i in result[tonode]:
                            #print(" "+i)
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0]!=result[tonode][1] and len(result)>5):
                            strings.append(string)
                            
                print(node)
        random.shuffle(strings)
        with open(outpath, 'w') as outfile:
            for string in strings:
                outfile.write(string) 
        outfile.close()
        
    
    def genAllOnePairs(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                result = Dijkstra.dijkstra(self.unitGraph, node);
                for tonode in self.nodes:
                    if tonode in result:
                        string = "";
                        for i in result[tonode]:
                            
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0]!=result[tonode][1] ):
                            print(string)
                            outfile.write(string) 
                print(node);       
        outfile.close()
        
    @staticmethod
    def GenStoGraph(inpath = None, outpath = "temp", EdgeType = "Chi", times=10000):
      with open(outpath, 'w') as outfile:
        infile = open(inpath, 'r') 
        while True:   
          line = infile.readline()
          if not line:
            infile.close()
            break;
          ints = line.split()
          node1 = ints[0]
          node2 = ints[1]
          alpha = ints[2]
          beta = ints[3]
          mean = 0;
          for i in range(times):
              mean +=  SSP_Utils.getWeibull(float(alpha), float(beta))
          mean = mean / times
         
          outfile.write(node1+" ") 
          outfile.write(node2+" ")
          outfile.write(alpha+" ") 
          outfile.write(beta+" ") 
          outfile.write(str(mean) + "\n") 
      outfile.close()    
        #g.dijkstra(0)
        
                
class Graph(): 
 
    def __init__(self, vNum, nodes, path = None):
        self.vNum = vNum
        self.nodes = nodes
    
    def isConnected(self, node1, node2):
        checkedList=[]
        c_nodes=[]
        
        c_nodes.append(node1)
        #checkedList.append(node2)
        
        while len(c_nodes)>0:
            temp_node = []
            for node in c_nodes:
                for tonode in self.nodes[node].neighbor: 
                    if tonode == node2:
                        return True
                    if tonode not in checkedList:
                        temp_node.append(tonode) 
                checkedList.append(node) 
            c_nodes=copy.copy(temp_node)
        
        return False
    
    def print_(self):
        for node in self.nodes:
            for tonode in self.nodes[node].neighbor:
                print(node+" "+tonode+" "+str(self.nodes[node].neighbor[tonode]))
                
    def pathLength(self, x, y):
        #print(x)
        #print(y)
        if y is None:
            return None
        length = 0
        if y[0]!= x[0] or y[-1]!=x[1]:
            print(x)
            print(y)
            sys.exit("path y not for x") 
            return None;
        else:
            for i in range(len(y)-1):
                if y[i] in self.nodes and y[i+1] in self.nodes[y[i]].neighbor:
                    length += float(self.nodes[y[i]].neighbor[y[i+1]])
                else:
                    print(y[i]+" "+y[i+1])
                    print(y)
                    sys.exit("edge not existing") 
                    return None;
            return length
        #self.adjmatrix = {};
   
    
            
class Dijkstra(object): 
    
    @staticmethod
    def minDistance(dist, queue): 
        # Initialize min value and min_index as -1 
        minimum = float("Inf") 
        min_index = -1
          
        # from the dist array,pick one which 
        # has min value and is till in queue 
        for node in dist: 
            if dist[node] < minimum and node in queue: 
                minimum = dist[node] 
                min_index = node 
        return min_index 
    
    @staticmethod
    def printPath(graph, parent, j, path): 
          
        #Base Case : If j is source 
        if parent[j] == -1 :  
            #print(j," ",end='') 
            path.append(j);
            return
        Dijkstra.printPath(graph, parent , parent[j], path) 
        #print(j," ", end='') 
        path.append(j);
    
    @staticmethod
    def printSolution(graph, src, dist, parent): 
        #src = 0
        #print("Vertex \t\tDistance from Source\tPath") 
        result={};
        for node in dist: 
            #print("{} {} {} ".format(src, node, dist[node])), 
            if dist[node] != float("Inf"):
                path=[];
                path.append(src);
                path.append(node);
                path.append(str(dist[node]));
                Dijkstra.printPath(graph, parent, node, path) 
                #print()
                #print(path);
                result[node] = path;
        #print(result)
        return result;
    
    @staticmethod      
    def dijkstra(graph, src, results = None): 

        dist={}
        for node in graph.nodes:
            dist[node]=float("Inf")

        parent={}
        for node in graph.nodes:
           parent[node]=-1

        dist[src] = 0     
        queue = [] 
        for node in graph.nodes: 
            queue.append(node) 
              
        while queue: 

            u = Dijkstra.minDistance(dist,queue)  
            if u == -1:
                break     
            queue.remove(u)  
            for node in graph.nodes: 
                if node in graph.nodes[u].neighbor and node in queue: 
                    #print(graph.nodes[u].neighbor[node])
                    if dist[u] + graph.nodes[u].neighbor[node] < dist[node]: 
                        
                        dist[node] = dist[u] + graph.nodes[u].neighbor[node] 
                        parent[node] = u   
        
        # print the constructed distance array 
        result = Dijkstra.printSolution(graph, src, dist, parent) 
        if results != None:
            results[src]=result
            print(src)
            sys.stdout.flush()
            
        return result
    
    @staticmethod      
    def dijkstra_1(graph, src): 

        dist={}
        for node in graph.nodes:
            dist[node]=float("Inf")

        parent={}
        for node in graph.nodes:
           parent[node]=-1

        dist[src] = 0     
        queue = [] 
        for node in graph.nodes: 
            queue.append(node) 
              
        while queue: 

            u = Dijkstra.minDistance(dist,queue)  
            if u == -1:
                break     
            queue.remove(u)  
            for node in graph.nodes: 
                if node in graph.nodes[u].neighbor and node in queue: 
                    if dist[u] + graph.nodes[u].neighbor[node] < dist[node]: 
                        dist[node] = dist[u] + graph.nodes[u].neighbor[node] 
                        parent[node] = u   
        
        # print the constructed distance array 
        result = Dijkstra.printSolution(graph, src, dist, parent) 
       
        #print(src)
        #sys.stdout.flush()      
        return src, result
        # print the constructed distance array 


# Driver program
#g = Graph(9)
if __name__ == "__main__":
    pass
    #SSP_Utils.read_NY("data/ssp/USA-road-d.COL.gr", "data/ssp/col/ssp_col", 512, 1000)
    #StoGraph.GenStoGraph(inpath="data/ssp/col/ssp_col",outpath = "ssp_col")
    #stoGraph=StoGraph("data/ssp/kro/ssp_kro", 1024)
    stoGraph=StoGraph("data/ssp/ny/ssp_ny", 768)
    #stoGraph.genAllTrainPairsShuffle("data/ssp/col/ssp_col_trainAllShuffle")
    stoGraph.genMultiRealization(10000, "data/ssp/ny/features/uniform100000_10000/", "100000uniform",startIndex=0)
#   stoGraph.genMultiRealization(1000, "data/ssp/kro/features/weibull55/", "weibull55",startIndex=0)
#stoGraph.genAllOnePairs("data/ssp/ssp_kro_OneAll")
#result = stoGraph.EgraphShortest("0")
#for node in result:
#    print(result[node])
#g.dijkstra(0)                
