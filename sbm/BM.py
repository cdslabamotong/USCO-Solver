# -*- coding: utf-8 -*-
"""
BM instance
# License: BSD 3-clause
"""
import numpy as np
import sys
import math
import random
#import time
import copy
import multiprocessing
import heapq
import scipy
from basic_Utils import Utils
from scipy.stats import powerlaw
#from base import StructuredModel

class BM_Utils(object):
    
    def test():
        x=[]
        array=["a","b","c"]
        x.append(array)
        array=["d","e","f"]
        x.append(array)
        
        matrix = []
        i=1
        for left_node in x[0]:
            array=[]
            for right_node in x[1]:
                array.append(i)
                i=2*i+1
            matrix.append(array)          
        row_ind, col_ind=scipy.optimize.linear_sum_assignment(matrix)
        print(matrix)
        print(row_ind)
        print(col_ind)
        print(np.array(matrix)[row_ind, col_ind].sum())
        
    @staticmethod
    def genExp(outpath, vNum, times):
        with open(outpath, 'w') as outfile:
            for i in range(vNum):
                for j in range(vNum):
                    beta = math.ceil(10*random.uniform(0, 1))
                    mean = 0
                    for _ in range(times):
                        mean = mean + random.exponential(beta)
                    mean = mean / times
                    outfile.write(str(i)+" ") 
                    outfile.write(str(j)+" ")
                    outfile.write(str(beta)+" ")  
                    outfile.write(str(mean) + "\n") 
        outfile.close()
        
    @staticmethod
    def genNorm(outpath, vNum, sigma_ratio, times):
        with open(outpath, 'w') as outfile:
            for i in range(vNum):
                for j in range(vNum):
                    mu = math.ceil(10*random.uniform(0, 1))
                    sigma = mu*sigma_ratio
                    mean = 0
                    for _ in range(times):
                        mean = mean + random.normalvariate(mu, sigma)
                    mean = mean / times
                    outfile.write(str(i)+" ") 
                    outfile.write(str(j)+" ")
                    outfile.write(str(mu)+" ")  
                    outfile.write(str(sigma)+" ")  
                    outfile.write(str(mean) + "\n") 
        outfile.close()
        
                
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
        X_train, Y_train = ([] for i in range(2))
        X_test, Y_test= ([] for i in range(2))
        
        while True:
            line = file1.readline() 
            if not line: 
                break 
            strings=line.split("|")
            if lineNum in trainLineNums:
                x=[]
                x.append(strings[0].split())
                x.append(strings[1].split())
                temp = strings[2].split(":")
                #print(temp)
                y={}
                for item in temp:                  
                    if len(item.split())>1:
                        y[item.split()[0]]=item.split()[1]
               
                X_train.append(x)
                Y_train.append(y)
                
            #print("===============================")    
            if lineNum in testLineNums:
                x=[]
                x.append(strings[0].split())
                x.append(strings[1].split())
                temp = strings[2].split(":")
                y={}
                
                for item in temp:                  
                    if len(item.split())>1:
                        y[item.split()[0]]=item.split()[1]
                        
                X_test.append(x)
                Y_test.append(y)           
            lineNum += 1
            
        file1.close()    
        if (len(X_train) != trainNum) or (len(Y_test) != testNum):
            sys.exit("getDataRandom: data fetch failed with sizes: X_train {} Y_test {}".format(
                len(X_train),len(Y_test))) 
        return X_train, Y_train, X_test, Y_test
    
    
    
    def NB_prediction(instance, X_train, Y_train, X_test, n_jobs = 1):
        matrix={}
        for left_node in instance.stoBMGraph.left_nodes:
            temp = {}
            for right_node in instance.stoBMGraph.right_nodes:
                temp[right_node]=0
            matrix[left_node]=temp
        for x, y in zip(X_train, Y_train):
            for item in y:
                matrix[item][y[item]] += 1
                
                
        Y_pred = []
        for x in X_test:
            Y_pred.append(BM_Utils.NB_prediction_pair(x, matrix))
        return Y_pred
    
    def NB_prediction_pair(x, matrix):
        y={}
        c_selected_right_nodes=[]
        for _ in range(len(x[0])):
            max_left = None
            max_right = None
            max_weight = -sys.maxsize
            for left_node in x[0]:
                for right_node in x[1]:
                    if left_node not in y and right_node not in c_selected_right_nodes:
                        if matrix[left_node][right_node] > max_weight:
                            max_weight = matrix[left_node][right_node]
                            max_right = right_node
                            max_left = left_node
            y[max_left]=max_right
            c_selected_right_nodes.append(max_right)
        #print(x)
        #print(y)
        return y
                        
                
      
    
        
        
    def random_prediction(X_test, n_jobs = 1):  
        Y_pred = []
        for x in X_test:
            y={}
            temp = copy.deepcopy(x[1])
            random.shuffle(temp)
            for left_node, right_node in zip(x[0],temp):
                y[left_node]=right_node
            Y_pred.append(y)
            
        return Y_pred
      

    def read_raw(inpath, outpath):
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
                alpha = math.ceil(10*random.uniform(0, 1))
                beta = math.ceil(10*random.uniform(0, 1))
                mean = alpha/(alpha+beta)
                
               
                outfile.write(node1+" ") 
                outfile.write(node2+" ")
                outfile.write(str(alpha)+" ") 
                outfile.write(str(beta)+" ") 
                outfile.write(str(mean) + "\n") 
        outfile.close()
    
    def normalize_index(inpath, outpath):
        node_map = {}
        topic_map = {}
        strings =[]
        with open(outpath, 'w') as outfile:
            infile = open(inpath, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                
                node = ints[0]
                if node not in node_map:
                    node_map[node]=str(len(node_map))
                topic = ints[1]
                if topic not in topic_map:
                    topic_map[topic]=str(len(topic_map))
                    
                alpha = math.ceil(10*random.uniform(0, 1))
                beta = math.ceil(10*random.uniform(0, 1))
                mean = alpha/(alpha+beta)
                string = node_map[node]+" "+topic_map[topic]+" "+str(alpha)+" "+str(beta)+" "+str(mean) + "\n"
                strings.append(string)
                outfile.write(string)
        outfile.close()
        
    def normalize_index_1(inpath, outpath):
        vertex_map = {}
        #topic_map = {}
        strings =[]
        with open(outpath, 'w') as outfile:
            infile = open(inpath, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                
                node = ints[0]
                if node not in vertex_map:
                    vertex_map[node]=str(len(vertex_map))
                topic = ints[1]
                if topic not in vertex_map:
                    vertex_map[topic]=str(len(vertex_map))
                    
                alpha = math.ceil(10*random.uniform(0, 1))
                beta = math.ceil(10*random.uniform(0, 1))
                mean = alpha/(alpha+beta)
                string = vertex_map[node]+" "+vertex_map[topic]+" "+str(alpha)+" "+str(beta)+" "+str(mean) + "\n"
                strings.append(string)
                outfile.write(string)
        outfile.close()
                
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
    
class BM_InputInstance(object):
    
    def __init__(self, stoBMGraphPath, featurePath, featureNum,  loss_type = "hamming", 
                 featureRandom = False, maxFeatureNum = 500, thread = 1, indexes=None):
        #self.graphs=[];
        self.featureNum = featureNum

        self.stoBMGraph = StoBMGraph(stoBMGraphPath)
        self.edgeVectors= {}
        self.loss_type=loss_type
        self.n_jobs=thread
        #self.featureIndicator=[]
        #self.featureIndicator_reset(1)
        self.featureRandom = featureRandom
        self.maxFeatureNum = maxFeatureNum
        #self.fraction=fraction
        self.featureIndicator=[]
        self.featureIndicator_reset(1)
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
              
        self.edgeVectors  =  copy.deepcopy(self.stoBMGraph.left_nodes) 
        for node in self.edgeVectors:
            for tonode in self.edgeVectors[node].neighbor:
                self.edgeVectors[node].neighbor[tonode]=np.zeros(self.maxFeatureNum)*100
                    
        for index, i in zip(self.featureIndexes, range(self.maxFeatureNum)):      
            path_graph="{}/{}".format(featurePath, index)
            infile = open(path_graph, 'r') 
            while True:   
                line = infile.readline()
                if not line:
                  infile.close()
                  break;
                ints = line.split()
                #print(ints)
                #print(path_graph)
                node = ints[0]
                topic = ints[1]
                
                #weight = int(ints[2]) # this is zero and one
                self.edgeVectors[node].neighbor[topic][i]=float(ints[2])
            #graph = Graph(temp_nodes)
            #self.graphs.append(graph)
        '''
        for node in self.edgeVectors:
            for tonode in self.edgeVectors[node].neighbor:
                print(self.edgeVectors[node].neighbor[tonode])
        '''
        
    def featureIndicator_reset(self, x):
        self.featureIndicator=[]
        for i in range(self.featureNum):
            self.featureIndicator.append(x)
        
    def computeScore(self, x, y, w): # x: topic, y: left nodes
        feature = self.computeFeature(x, y)
        return w.dot(feature)
    

   
    def computeFeature(self, x, y):
        feature = []
        #print(self.featureNum)
        for i in range(self.featureNum):
            if True:
                feature.append(self.computeScoreOneFeature(x, y, i))
        #print(feature)
        return feature   
    
    def computeScoreOneFeature(self, x, y, featureIndex, getCover = False):    # x is topic, y is node
        '''compute f^g(M,P)'''
        cost = 0
        for item in y:
            cost += self.edgeVectors[item].neighbor[y[item]][featureIndex]
            
        return cost

    
        
        #return 1/graph.pathLength(x, y)
    '''    
    def computeScoreGain(self, x , y, w, c_covers):
        new_c_covers = copy.deepcopy(c_covers)
        total_gain = 0;
        for i, cover, ww in zip(range(self.featureNum), new_c_covers, w):
            gain = 0
            coveredTopics, _ = self.computeScoreOneFeature(x, y, i, getCover=True)
            for topic in coveredTopics:
                if cover[topic] == 0:
                    gain += 1
                    cover[topic] = 1
            total_gain += ww*gain
        return total_gain, new_c_covers
    '''        
            
    def inference(self, x, w):
        #print("inference") 
        matrix = []
        for left_node in x[0]:
            array=[]
            for right_node in x[1]:
                weight = 0
                for i in range(self.featureNum): 
                    weight += w[i]*self.edgeVectors[left_node].neighbor[right_node][i]
                array.append(weight)
            matrix.append(array)
            
        row_ind, col_ind=scipy.optimize.linear_sum_assignment(matrix)
        y={}
        for row, col in zip(row_ind, col_ind):
            y[x[0][row]]=x[1][col]
        
        #print(np.array(matrix)[row_ind, col_ind].sum())     

        return y
    
    
    

    '''
    def batch_inference(self, X, w, n_jobs=1, offset = None):
        #results={}
        p = multiprocessing.Pool(n_jobs)
        results=p.starmap(self.inference, ((x, w) for x in X))
        p.close()
        p.join()
        print("inference block DONE")
        return results 
    '''
    def batch_inference(self, X, w, n_jobs=1, offset = None):
        print("inference block ver 1 Running {}".format(n_jobs))
        results=[]
        p = multiprocessing.Pool(n_jobs)
        block_size =int (len(X)/n_jobs)
        resultss=p.starmap(self.inference_block, ((X[i*block_size:min([len(X),(i+1)*block_size])], w) for i in range(n_jobs) ))
        p.close()
        p.join()
        for item in resultss:
            results.extend(item)
        print("inference block ver 1 DONE")
        return results   

    def inference_block(self, X, w):
        Y_pred = []
        for x in X:
            Y_pred.append(self.inference(x, w))
            
        return Y_pred

    
    def batch_predict(self, X, w, n_jobs=1, offset = None):
        return self.batch_inference(X, w, n_jobs=n_jobs, offset = offset)
    
    # This is a baseline treating all weight as 1
    def inferenceBasic(self, x):
        pass
    
    
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
        return self.inference(x, w)

            #sys.exit("loss_augmented_inference method not speficied.") 
    
    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs=1):
        print("loss_augmented_inference block RUNNING")
        result=self.batch_inference(X, w, n_jobs)

        print("loss_augmented_inference block Done")
        return result
    
    def test(self, X_test, Y_test, Y_pred, logpath= None, preTrainPathResult = None):

        true_cost_total=0
        pred_cost_total=0
        #count_success = 0
        #count_fail = 0
        for x, y, y_hat in zip(X_test, Y_test, Y_pred):
            #print(y)
            #print(y_hat)
            #print()
            true_cost = self.stoBMGraph.stoBMGraphCost(x, y)
            
            true_cost_total += true_cost  
            
            pred_cost = self.stoBMGraph.stoBMGraphCost(x, y_hat)
            pred_cost_total += pred_cost
            
            #print("{} {}".format(true_coverage, pred_coverage))

        
        output = "pred_coverage_total/true_coverage_total: "+ Utils.formatFloat(pred_cost_total/true_cost_total)
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        return Utils.formatFloat(pred_cost_total/true_cost_total)
        
    def testUnitAllPair(self, truepath, unitpath):
        pass
        
           
class StoBMGraph(object):  
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
    def __init__(self, path):
        self.left_nodes={}
        self.right_nodes = []
        self.vertices=[]

        self.EGraph = None
        
 
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            items = line.split()
            left_node = items[0]
            right_node = items[1]
            mu = float(items[2])
            sigma = float(items[3])
            mean = float(items[4]) # mean of Chi-square Gaussian
            # para_2 = float(ints[3])
            
            if left_node in self.left_nodes:
                self.left_nodes[left_node].neighbor[right_node]=[mu, sigma, mean]
            else:
                new_node = self.Node(left_node)
                new_node.neighbor={}
                self.left_nodes[left_node]= new_node
                self.left_nodes[left_node].neighbor[right_node]=[mu, sigma, mean]
                
            if right_node not in self.right_nodes:
                self.right_nodes.append(right_node) 
                
            if left_node not in self.vertices:
                self.vertices.append(left_node) 
                
            
                
            if right_node not in self.vertices:
               self.vertices.append(right_node)   
                

        
    #def EgraphShortest(self, source, destination = None):
    #    return Dijkstra.dijkstra(self.EGraph,source)
        
    def genMultiRealization(self, num, outfolder, rType, startIndex=0):
        for cout in range(num):
            print(cout)
            path = "{}{}".format(outfolder, cout+startIndex)
            if rType=="true":
                self.genOneRealizationTrue(path)
            if rType=="uniform":
                self.genOneRealizationUniform(path)
            if rType=="uniformFloat30":
                self.genOneRealizationUniformFloat30(path)
            if rType=="uniformFloat100":
                self.genOneRealizationUniformFloat100(path)
            if rType=="uniformFloat500":
                self.genOneRealizationUniformFloat500(path)
            if rType=="uniformGaussian500":
                self.genOneRealizationUniformFloat500(path)
            if rType=="uniformFloat1000":
                self.genOneRealizationUniformFloat1000(path)
            if rType=="uniform001_ver0":
                self.genOneRealizationUniform001_ver0(path)
            if rType=="uniform001_ver1":
                self.genOneRealizationUniform001_ver1(path)
            if rType=="uniform01_ver1":
                self.genOneRealizationUniform01_ver1(path)
            #sys.exit("wrong feature type")
                
            
    def genOneRealizationTrue(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    mu = self.left_nodes[left_node].neighbor[right_node][0]
                    sigma = self.left_nodes[left_node].neighbor[right_node][1]                   
                    weight = random.normalvariate(mu, sigma)
                    outfile.write(left_node +" "+right_node+" "+str(weight)+"\n") 
        outfile.close()
        
    def genOneRealizationUniformFloat30(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    mu = self.left_nodes[left_node].neighbor[right_node][0]
                    mu_1= mu*0.7+random.uniform(0,1)*mu*0.6
                    sigma = self.left_nodes[left_node].neighbor[right_node][1]                   
                    weight = random.normalvariate(mu_1, sigma)
                    outfile.write(left_node +" "+right_node+" "+Utils.formatFloat2bits(weight)+"\n") 
        outfile.close()
    
    def genOneRealizationUniformFloat100(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    mu = self.left_nodes[left_node].neighbor[right_node][0]
                    mu_1= random.uniform(0,1)*mu*2
                    sigma = self.left_nodes[left_node].neighbor[right_node][1]                   
                    weight = random.normalvariate(mu_1, sigma)
                    outfile.write(left_node +" "+right_node+" "+Utils.formatFloat2bits(weight)+"\n") 
        outfile.close()
        
    def genOneRealizationUniformFloat500(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    mu = self.left_nodes[left_node].neighbor[right_node][0]
                    mu_1= mu*(1-5)+random.uniform(0,1)*mu*(2*5)
                    sigma = self.left_nodes[left_node].neighbor[right_node][1]                   
                    weight = random.normalvariate(mu_1, sigma)
                    outfile.write(left_node +" "+right_node+" "+Utils.formatFloat2bits(weight)+"\n") 
        outfile.close()
        
    def genOneRealizationUniformFloat1000(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    mu = self.left_nodes[left_node].neighbor[right_node][0]
                    mu_1= mu*(1-10)+random.uniform(0,1)*mu*(2*10)
                    sigma = self.left_nodes[left_node].neighbor[right_node][1]                   
                    weight = random.normalvariate(mu_1, sigma)
                    outfile.write(left_node +" "+right_node+" "+Utils.formatFloat2bits(weight)+"\n") 
        outfile.close()
        
    
    def genOneRealizationUniform(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    weight = random.uniform(0,1)*10
                    outfile.write(left_node + " " + right_node +" " + str(weight)+"\n") 
        outfile.close() 
        
        
        
    def genOneRealizationUniform001_ver0(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:                   
                    if random.uniform(0,1)<0.01:
                        weight = random.uniform(0,1)*10
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
        outfile.close() 
        
    def genOneRealizationUniform01_ver0(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:                   
                    if random.uniform(0,1)<0.1:
                        weight = random.uniform(0,1)*10
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
        outfile.close() 
        
    def genOneRealizationUniform001_ver1(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    
                    if random.uniform(0,1)<0.01:
                        weight = random.uniform(0,1)*10
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
                    else:
                        weight = 100
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
        outfile.close() 
        
    def genOneRealizationUniform01_ver1(self, outpath):
        with open(outpath, 'w') as outfile:
            for left_node in self.left_nodes:
                for right_node in self.left_nodes[left_node].neighbor:
                    
                    if random.uniform(0,1)<0.1:
                        weight = random.uniform(0,1)*10
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
                    else:
                        weight = 100
                        outfile.write(left_node + " " + right_node +" " + str(weight)+"\n")
        outfile.close() 
    
    
   
        
    def genTrainPairs(self, outpath, num, scale):
        r = powerlaw.rvs(2.5, scale = scale, size=num)
        r =[int(x)+1 for x in r]
        #print(r)
        count = 0
        with open(outpath, 'w') as outfile:
            for size in r:
                
                new_left_nodes =  copy.deepcopy(list(self.left_nodes.keys()))
                new_right_nodes =  copy.deepcopy(self.right_nodes)
                random.shuffle(new_left_nodes)
                random.shuffle(new_right_nodes)
                
                #print(new_topics)
                x=[]
                x.append(new_left_nodes[0:size])
                x.append(new_right_nodes[0:size])
                string=""
                y=self.inferenceTrue(x)
                
                for left_node in x[0]:
                    string = string+left_node+" "
                string += "|"
                for right_node in x[1]:
                    string = string+right_node+" "
                string += "|"
                for item in y:
                    string = string+item+" "+y[item]+":"
                string += "\n"
                outfile.write(string) 
                print(string)
                count += 1
                print(count)
        outfile.close()
    
    def inferenceTrue(self, x):
        matrix = []
        for left_node in x[0]:
            array=[]
            for right_node in x[1]:
                array.append(self.left_nodes[left_node].neighbor[right_node][2])
            matrix.append(array)
            
        row_ind, col_ind=scipy.optimize.linear_sum_assignment(matrix)
        #print(row_ind) 
        #print(col_ind)
        #print(x[0]) 
        #print(x[1])
        y={}
        for row, col in zip(row_ind, col_ind):
            #print(x[0][row]+" "+x[1][col])
            y[x[0][row]]=x[1][col]
        
        print(np.array(matrix)[row_ind, col_ind].sum())  
        #print(y)
        return y
    
    def stoBMGraphCost(self, x, y):
        set_x0=set(x[0])
        set_x1=set(x[1])
        set_y0=set(list(y.keys()))
        set_y1=set(list(y.values()))
        
        if set_x0 != set_y0:
            sys.exit("set_x0 != set_y0")
        if set_x1 != set_y1:
            sys.exit("set_x1 != set_y1")
            
        cost = 0
        
        for item in y:
            cost += self.left_nodes[item].neighbor[y[item]][2]
            
        return cost
        
    
    
        




# Driver program
#g = Graph(9)
if __name__ == "__main__":
    pass
    #BM_Utils.test()
    #BM_Utils.genNorm("data/bm/norm_1_128/bm_norm_1_128", 128, 0.01, 1000)
    #DR_Utils.normalize_index_1("data/dr/yahoo/yahoo_ad_raw", "data/dr/cora/dr_yahoo_1")

    stoBMCoverGraph=StoBMGraph("data/bm/norm_30_128/bm_norm_30_128") 
    #stoBMCoverGraph.genTrainPairs("data/bm/norm_1_128/bm_norm_1_128_train_10000_20", 10000, 20)
    stoBMCoverGraph.genMultiRealization(1000, "data/bm/norm_30_128/features/uniformFloat30_1000/", "uniformFloat30", startIndex=0)
    #stoBMCoverGraph.genMultiRealization(10000, "data/bm/norm_1_128/features/true_10000/", "true", startIndex=0)
 