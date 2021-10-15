# -*- coding: utf-8 -*-
"""
ssc utilities
# License: BSD 3-clause
"""
import numpy as np
import sys
import math
import random
import time
import copy
import multiprocessing
import heapq
from basic_Utils import Utils
from scipy.stats import powerlaw
#from base import StructuredModel

class DR_Utils(object):
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
                topics = strings[0].split()
                nodes = strings[1].split()
                X_train.append(topics)
                Y_train.append(nodes)
                
            #print("===============================")    
            if lineNum in testLineNums:
               topics = strings[0].split()
               nodes = strings[1].split()
               X_test.append(topics)
               Y_test.append(nodes)           
            lineNum += 1
            
        file1.close()    
        if (len(X_train) != trainNum) or (len(Y_test) != testNum):
            sys.exit("getDataRandom: data fetch failed with sizes: X_train {} Y_test {}".format(
                len(X_train),len(Y_test))) 
        return X_train, Y_train, X_test, Y_test
    
    
    
    def NB_prediction(instance, X_train, Y_train, X_test, n_jobs = 1):
        matrix={}
        for topic in instance.stoCoverGraph.topics:
            matrix[topic]={}
            for node in instance.stoCoverGraph.nodes:
                matrix[topic][node]=0
                
        for x, y in zip(X_train, Y_train):
            for topic in x:
                for node in y:
                    matrix[topic][node] += 1
        
        
        Y_pred = []
        for x in X_test:
            rank={}
            for node in instance.stoCoverGraph.nodes:
                score = 0
                for topic in x:
                    score += matrix[topic][node]
                rank[node]=score
            k=int(instance.fraction*len(x))
            sortedNodeScore = sorted(rank.items(), key=lambda kv: kv[1],reverse=True)
            y = []
            for item in sortedNodeScore:
                y.append(item[0])
                if len(y) == k:
                    break
            Y_pred.append(y)
                
        return Y_pred
                
      
    
        
        
    def random_prediction(instance, X_test, n_jobs = 1):  
        Y_pred = []
        for topic in X_test:
            size = int(instance.fraction*len(topic))
            nodes = copy.deepcopy(list(instance.stoCoverGraph.nodes.keys()))
            random.shuffle(nodes)
            Y_pred.append(nodes[0:size])
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
    
class DR_InputInstance(object):
    
    def __init__(self, stoCoverGraphPath, featurePath, featureNum, fraction, loss_type = "hamming", 
                 featureRandom = False, maxFeatureNum = 500, thread = 1, indexes=None):
        #self.graphs=[];
        self.featureNum = featureNum

        self.stoCoverGraph = StoCoverGraph(stoCoverGraphPath)
        self.edgeVectors= {}
        self.loss_type=loss_type
        self.n_jobs=thread
        #self.featureIndicator=[]
        #self.featureIndicator_reset(1)
        self.featureRandom = featureRandom
        self.maxFeatureNum = maxFeatureNum
        self.fraction=fraction
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
              
        self.edgeVectors  =  copy.deepcopy(self.stoCoverGraph.nodes) 
        for node in self.edgeVectors:
            for tonode in self.edgeVectors[node].neighbor:
                self.edgeVectors[node].neighbor[tonode]=np.zeros(self.maxFeatureNum)
                    
        for index, i in zip(self.featureIndexes,range(self.maxFeatureNum)):      
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
                self.edgeVectors[node].neighbor[topic][i]=1
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
        coveredTopics = []
        for node in y:
            for topic in self.edgeVectors[node].neighbor:
                if self.edgeVectors[node].neighbor[topic][featureIndex] == 1 and topic in x and topic not in coveredTopics:
                    coveredTopics.append(topic)
        if getCover:
            return coveredTopics, len(coveredTopics)
        else:
            return len(coveredTopics)

        
        
        #return 1/graph.pathLength(x, y)
        
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
            
            
    def inference(self, x, w):
        #print("inference") 
        solution = []
        gains = []
        c_covers = []
        
            
        for i in range(self.featureNum):
            cover={}
            for topic in x:
                cover[topic]=0
            c_covers.append(cover)
            #print(c_coverOneGraph)
            
        for node in self.edgeVectors:
            gain, node_cover = self.computeScoreGain(x, [node], w, c_covers) 
            heapq.heappush(gains, (-gain, node, node_cover))
        # 
        score_gain, node, node_cover = heapq.heappop(gains)
        solution.append(node)
        c_score = -score_gain
        #print("{} {}".format(node, -score_gain)) 

        c_cover = node_cover
        
        for _ in range(int(self.fraction*len(x)) - 1):
            matched = False
            while not matched:
                _, current_node, _ = heapq.heappop(gains)
                score_gain, new_cover = self.computeScoreGain(x, [node], w, c_covers)
                heapq.heappush(gains, (-score_gain, current_node, new_cover))
                matched = gains[0][1] == current_node

            score_gain, node, c_cover = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.append(node)

        return solution
    
    
    def computeScoreGain_ver1(self, x , y, w, c_covers):
        #new_c_covers = copy.deepcopy(c_covers)
        total_gain = 0;
        for i, cover, ww in zip(range(self.featureNum), c_covers, w):
            gain = 0
            coveredTopics, _ = self.computeScoreOneFeature(x, y, i, getCover=True)
            for topic in coveredTopics:
                if cover[topic] == 0:
                    gain += 1
                    #cover[topic] = 1
            total_gain += ww*gain
        return total_gain
            
    def getCover(self, y, c_covers):
        for i in range(self.featureNum): 
            for node in y:
                for topic in self.edgeVectors[node].neighbor:
                    if self.edgeVectors[node].neighbor[topic][i] == 1 and topic in c_covers:
                        c_covers[topic]=1
        return c_covers
            
        
        
    def inference_ver1(self, x, w):
        #print("inference") 
        solution = []
        gains = []
        c_covers = []
        
            
        for i in range(self.featureNum):
            cover={}
            for topic in x:
                cover[topic]=0
            c_covers.append(cover)
            #print(c_coverOneGraph)
            
        for node in self.edgeVectors:
            gain = self.computeScoreGain_ver1(x, [node], w, c_covers) 
            heapq.heappush(gains, (-gain, node))
        # 
        score_gain, node = heapq.heappop(gains)
        solution.append(node)
        c_covers=self.getCover(solution, c_covers)
        c_score = -score_gain
        #print("{} {}".format(node, -score_gain)) 

        #c_cover = node_cover
        
        for _ in range(int(self.fraction*len(x)) - 1):
            matched = False
            while not matched:
                _, current_node = heapq.heappop(gains)
                score_gain = self.computeScoreGain_ver1(x, [node], w, c_covers)
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node

            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.append(node)
            c_covers=self.getCover([node],c_covers)

        return solution

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
        print("inference block ver 1 Running")
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
            Y_pred.append(self.inference_ver1(x, w))
            
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
    '''
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
    '''
    def test(self, X_test, Y_test, Y_pred, logpath= None, preTrainPathResult = None):

        true_coverage_total=0
        pred_coverage_total=0
        #count_success = 0
        #count_fail = 0
        for topic, y, y_hat in zip(X_test, Y_test, Y_pred):
            #print(y)
            #print(y_hat)
            #print()
            true_coverage = self.stoCoverGraph.stoGraphCoverage(topic, y)
            
            true_coverage_total += true_coverage
            pred_coverage = self.stoCoverGraph.stoGraphCoverage(topic, y_hat)
            pred_coverage_total += pred_coverage
            
            #print("{} {}".format(true_coverage, pred_coverage))

        
        output = "true_coverage_total/pred_coverage_total: "+ Utils.formatFloat(true_coverage_total/pred_coverage_total)
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        return Utils.formatFloat(true_coverage_total/pred_coverage_total)
        
    def testUnitAllPair(self, truepath, unitpath):
        pass
        
           
class StoCoverGraph(object):  
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
        self.nodes={}
        self.topics = {}
        self.vertices=[]

        self.EGraph = None
        
 
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            items = line.split()
            node = items[0]
            topic = items[1]
            alpha = items[2]
            beta = items[3]
            mean = float(items[4]) # mean of Chi-square Gaussian
            # para_2 = float(ints[3])
            
            if node in self.nodes:
                self.nodes[node].neighbor[topic]=[alpha,beta, mean]
            else:
                new_node = self.Node(node)
                new_node.neighbor={}
                self.nodes[node]= new_node
                self.nodes[node].neighbor[topic]=[alpha,beta, mean]
                
            if topic in self.topics:
                self.topics[topic].neighbor[node]=[alpha,beta, mean]
            else:
                new_node = self.Node(topic)
                new_node.neighbor={}
                self.topics[topic] = new_node
                self.topics[topic].neighbor[node]=[alpha,beta, mean]
            
            if node not in self.vertices:
                self.vertices.append(node)
            if topic not in self.vertices:
                self.vertices.append(topic)    
                
        self.nodeNum = len(self.nodes)
        self.topicNum=len(self.topics)
        
        #create mean graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for topic in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[topic]=self.nodes[node].neighbor[topic][2];
        
        self.ECoverGraph = CoverGraph(temp_nodes)
        
        #create unit graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for topic in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[topic]=1;
        
        self.unitCoverGraph = CoverGraph(temp_nodes)
        
        #create random graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for topic in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[topic]=random.random()
        self.randomCoverGraph = CoverGraph(temp_nodes)

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
            if rType=="uniform_1":
                self.genOneRealizationUniform_1(path)
            if rType=="weibull55":
                self.genOneRealization_weibull55(path)
                
            
    def genOneRealizationTrue(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for topic in self.nodes[node].neighbor:
                    if random.uniform(0, 1)<self.nodes[node].neighbor[topic][2]:
                        outfile.write(node+" "+topic+"\n") 
        outfile.close()
        
    
    def genOneRealizationUniform(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for topic in self.nodes[node].neighbor:
                    if random.uniform(0, 1)<0.1:
                        outfile.write(node+" "+topic+"\n") 
        outfile.close() 
    
    '''
    def genOneRealizationUniform_1(self, outpath):
        choices = [1,1,1,1,1,1,1,1,5,5,5,5,10,10,10,10,100,100,1000,1000,100000]
        temp_nodes  =  copy.deepcopy(self.nodes) 
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for topic in self.nodes[node].neighbor:
                    weight = choices[math.floor(len(choices)*random.uniform(0, 1))]
                    temp_nodes[node].neighbor[topic]=weight
                    outfile.write(node+" ") 
                    outfile.write(topic+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close() 
       
    def genOneRealization_weibull55(self, outpath):
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for topic in self.nodes[node].neighbor:
                    #alpha = self.nodes[node].neighbor[tonode][0]
                    #beta = self.nodes[node].neighbor[tonode][1]
                    weight = DR_Utils.getWeibull(5, 5)
                    outfile.write(node+" ") 
                    outfile.write(topic+" ")
                    outfile.write(str(weight)+"\n") 
        outfile.close()
    '''     
   
        
    def genTrainPairs(self, outpath, num, fraction, scale):
        r = powerlaw.rvs(2.5, scale = scale, size=num)
        r =[int(x) for x in r]
        #print(r)
        count = 0
        with open(outpath, 'w') as outfile:
            for size in r:
                
                new_topics =  copy.deepcopy(self.topics)
                random.shuffle(new_topics)
                #print(new_topics)
                x=new_topics[0:size]
                string=""
                y=self.inferenceTrue(x, fraction)
                for topic in x:
                    string = string+topic+" "
                string += "|"
                for node in y:
                    string = string+node+" "
                string += "\n"
                outfile.write(string) 
                #print(string)
                count += 1
                print(count)
        outfile.close()
    
    def inferenceTrue(self, x, fraction):
        k=int(fraction*len(x))
        y=[]
        c_covers={}
        gains = []
        
        for topic in x:
            c_covers[topic]=0
        for node in self.nodes:
            gain, node_cover = self.inferenceTrueGain([node], c_covers) 
            heapq.heappush(gains, (-gain, node, node_cover))
        # 
        score_gain, node, node_cover = heapq.heappop(gains)
        y.append(node)
        c_score = -score_gain
        #print("{} {}".format(node, -score_gain)) 

        c_cover = node_cover
        
        for _ in range(k-1):
            matched = False
            while not matched:
                _, current_node, _ = heapq.heappop(gains)
                score_gain, new_cover = self.inferenceTrueGain([node], c_covers)
                heapq.heappush(gains, (-score_gain, current_node, new_cover))
                matched = gains[0][1] == current_node

            score_gain, node, c_cover = heapq.heappop(gains)
            c_score = c_score -  score_gain
            y.append(node)

        return y
    
    def inferenceTrueGain(self, y, c_covers):
        new_c_covers=copy.deepcopy(c_covers)
        total_gain = 0
        for topic in c_covers:
            prod = 1-c_covers[topic]
            for node in y:
                if node in self.ECoverGraph.nodes and topic in self.ECoverGraph.nodes[node].neighbor:
                    prod = prod*(1-self.ECoverGraph.nodes[node].neighbor[topic])          
            new_c_covers[topic]=1-prod
            topic_gain=(1-prod)-c_covers[topic]
            total_gain = total_gain + topic_gain
        return total_gain, new_c_covers
    
    def stoGraphCoverage(self, topics, nodes):
        c_covers={}
        
        for topic in topics:
            c_covers[topic]=0
        coverage, _ =  self.inferenceTrueGain(nodes, c_covers) 
        return coverage
    
    def genGCNNodeGraph(self, outpath):
        with open(outpath, 'w') as outfile:
            for node_1 in self.nodes:
                for node_2 in self.nodes:
                    set1=set(list(self.nodes[node_1].neighbor.keys()))
                    set2=set(list(self.nodes[node_2].neighbor.keys()))                 
                    weight = len(set1.intersection(set2))
                    outfile.write(node_1+" "+node_2+" "+str(weight)+"\n")
        outfile.close()      
        
    def genGCNTopicGraphSparse(self, outpath, subsampling = None):
       with open(outpath, 'w') as outfile:
           for topic_1 in self.topics:
               for topic_2 in self.topics:
                   set1=set(list(self.topics[topic_1].neighbor.keys()))
                   set2=set(list(self.topics[topic_2].neighbor.keys()))                 
                   weight = len(set1.intersection(set2))
                   if weight>0:
                       if subsampling is None:
                           outfile.write(topic_1+" "+topic_2+" "+str(weight)+"\n")
                       else:
                           if random.uniform(0,1)<subsampling:
                               outfile.write(topic_1+" "+topic_2+" "+str(weight)+"\n")
       outfile.close()      
                
        
        
    @staticmethod
    def GenStoCoverGraph(inpath = None, outpath = "temp", EdgeType = "Chi", times=10000):
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
          mean = 0;
          for i in range(times):
              mean +=  DR_Utils.getWeibull(float(alpha), float(beta))
          mean = mean / times
         
          outfile.write(node1+" ") 
          outfile.write(node2+" ")
          outfile.write(alpha+" ") 
          outfile.write(beta+" ") 
          outfile.write(str(mean) + "\n") 
      outfile.close()    
        #g.dijkstra(0)
        
        
    
        
                
class CoverGraph(): 
 
    def __init__(self, nodes, path = None):

        self.nodes = nodes
    
    def coverage(self, nodes, topics):
        coveredTopics = []
        for node in nodes:
            for topic in self.nodes[node].neighbor:
                if topic in topics and topic not in coveredTopics:
                    coveredTopics.append(topic)
                                   
        return len(coveredTopics)
    
    def print_(self):
        for node in self.nodes:
            for tonode in self.nodes[node].neighbor:
                print(node+" "+tonode+" "+str(self.nodes[node].neighbor[tonode]))
                
   
            



# Driver program
#g = Graph(9)
if __name__ == "__main__":
    pass
    #DR_Utils.read_raw("data/dr/yahoo/yahoo_ad_raw", "data/dr/yahoo/dr_yahoo")
    #DR_Utils.normalize_index("data/dr/yahoo/yahoo_ad_raw", "data/dr/yahoo/dr_yahoo_0")

    stoCoverGraph=StoCoverGraph("data/dr/yahoo/dr_yahoo_0") 
    stoCoverGraph.genGCNTopicGraphSparse("data/dr/yahoo/dr_yahoo_gcnTopicSparse001",0.01)
    #stoCoverGraph.genTrainPairs("data/dr/yahoo/dr_yahoo_0_train_10000_0.1_200", 10000, 0.1, 200)
    #stoCoverGraph.genMultiRealization(10000, "data/dr/cora/features/uniform_10000_1/", "uniform", startIndex=0)
