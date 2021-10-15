# -*- coding: utf-8 -*-
"""
USCO functions
"""
import numpy as np
import sys
import math
import random
import time
import copy
import multiprocessing
#from base import StructuredModel

class Utils(object):
    def writeToFile(path, string, toconsole = False, preTrainPathResult = None):
         logfile = open(path, 'a')
         logfile.write(string+"\n") 
         logfile.close() 
         if toconsole is True:
             print(string)
             
         if preTrainPathResult is not None:
             logfile = open(preTrainPathResult, 'a')
             logfile.write(string+"\n") 
             logfile.close() 
    
    def formatFloat(x):
        return "{:.4f}".format(x)
             
    def formatFloat2bits(x):
        return "{:.2f}".format(x)

    
    def save_pretrain(path, weights, featureIndex, featurePath):
         with open(path+"/featureIndex", 'w') as outfile:
            for index, w in zip(featureIndex, list(weights)):
                outfile.write(str(index)+" "+str(w)+" "+"\n") 
         outfile.close()
         
         
        

# Driver program
#g = Graph(9)
if __name__ == "__main__":
    pass
             
