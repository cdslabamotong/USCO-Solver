# -*- coding: utf-8 -*-
######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs
######################
import numpy as np
import sys
import math
import random
import copy
import multiprocessing
from basic_base import StructuredModel

class Model(StructuredModel):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionality of the joint
        features for an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, X, Y, instance):
        # set any data-specific parameters in the model
        #self.featureNum = instance.featureNum
        self.size_joint_feature= instance.featureNum
        self.instance = instance
        self.inference_calls = 0
        #if self.n_features is None:
        #    self.n_features = n_features
        #elif self.n_features != n_features:
        #    raise ValueError("Expected %d features, got %d"
        #                     % (self.n_features, n_features))

        #n_labels = Y.shape[1]
        #if self.n_labels is None:
        #    self.n_labels = n_labels
        #elif self.n_labels != n_labels:
        #    raise ValueError("Expected %d labels, got %d"
          #                   % (self.n_labels, n_labels))

        #self._set_size_joint_feature()
        #self._set_class_weight()
        pass
    """
    def joint_feature(self, x, y):
        raise NotImplementedError()
    """ 
    def joint_feature(self, x, y, n_jobs =1):
        if n_jobs == 1:
            return self.instance.computeFeature(x,y)
        else:
            return self.instance.computeFeature(x, y, n_jobs=n_jobs)
        '''
        feature = np.zeros(self.featureNum)
        index = 0
        for graph in self.instance.graphs:
            distance_matrix = np.zeros( (2, 3) )
            for v in range(self.instance.nNUm):
                x_min=sys.maxsize
                for u in x:
                    if distance_matrix[v][u]<x_min:
                        x_min=distance_matrix[v][u]
                y_min=sys.maxsize
                for u in y:
                    if distance_matrix[v][u]<y_min:
                        y_min=distance_matrix[v][u]
                if y_min<x_min:
                    feature[index] += 1
            index += 1
        return feature
        '''
    
    def joint_feature_block(self, X, Y):
        #joint_feature_ = np.zeros(self.size_joint_feature)
        joint_feature_ = []
        for x, y in zip(X, Y):
                joint_feature_.append(self.joint_feature(x, y))
        return joint_feature_
    
    
    def batch_joint_feature(self, X, Y, Y_true=None, n_jobs=1):
        print("batch_joint_feature running")
        #print(Y)
        joint_feature_ = np.zeros(self.size_joint_feature)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                joint_feature_ += self.joint_feature(x, y, y_true)
        else:
            print("here " + str(n_jobs))
            
            p = multiprocessing.Pool(n_jobs)
            block_size =int (len(X)/n_jobs)
            Ys=p.starmap(self.joint_feature_block, ((X[i*block_size:min([len(X),(i+1)*block_size])], Y[i*block_size:min([len(X),(i+1)*block_size])]) for i in range(n_jobs) ))
            p.close()
            p.join()
            for y_temp in Ys:
                for feature in y_temp:
                    joint_feature_ += np.array(feature)
                    
            
            #for x, y in zip(X,Y):
            #    joint_feature_ += np.array(self.joint_feature(x,y,n_jobs=n_jobs))
            
            
            #p = multiprocessing.Pool(n_jobs)
            #results=p.starmap(self.joint_feature, ((x, y) for x, y in zip(X,Y)))
            #p.close()
            #p.join()
            #for feature_block in results:
            #    for feature in feature_block:
            #        joint_feature_ += np.array(feature)
            
        #print(joint_feature_)
        print("batch_joint_feature done")
        return joint_feature_

    def _loss_augmented_djoint_feature(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.joint_feature(x_loss_augmented, y)
                - self.joint_feature(x_loss_augmented, y_hat))
    
    def batch_predict(self, X, w,relaxed=None, constraints=None, n_jobs=1):
        return self.instance.batch_predict(X,w,n_jobs=n_jobs)
    
    def batch_inference(self, X, w,relaxed=None, constraints=None, n_jobs=1):
        return self.instance.batch_inference(X,w,n_jobs=n_jobs)
        

    def inference(self, x, w, relaxed=None, constraints=None):
        self.inference_calls += 1
        solution = self.instance.inference(x,w)
        return solution
        #raise NotImplementedError()
    
    '''
    def batch_inference(self, X, w, relaxed=None, constraints=None):
        # default implementation of batch inference
        if constraints:
            return [self.inference(x, w, relaxed=relaxed, constraints=c)
                    for x, c in zip(X, constraints)]
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]
    '''
    def loss(self, y, y_hat):
        '''
        # hamming loss:
        if isinstance(y_hat, tuple):
            return self.continuous_loss(y, y_hat[0])
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)
        '''
        return self.instance.loss(y,y_hat)
        

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def max_loss(self, y):
        # maximum possible los on y for macro averages
        sys.exit("max_loss not implemented") 
        if hasattr(self, 'class_weight'):            return np.sum(self.class_weight[y])
        return y.size

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        sys.exit("continuous_loss not implemented") 
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        
        #print("FALLBACK no loss augmented inference found")
        #return self.inference(x, w)
        #print("loss_augmented_inference RUNNING")
        self.inference_calls += 1
        y_pre = self.instance.loss_augmented_inference(x,y,w)
        #print("loss_augmented_inference DONE")
        return y_pre
    
    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs =1):
        
        #print("FALLBACK no loss augmented inference found")
        #return self.inference(x, w)
        #print("loss_augmented_inference RUNNING")
        '''
        self.inference_calls += len(X)
        result =[]
        for x, y in zip(X,Y):
            result.append(self.instance.loss_augmented_inference(x,y,w))
        '''
        return self.instance.batch_loss_augmented_inference(X,Y,w,n_jobs=n_jobs)
    

        
    '''
    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None):
        sys.exit("batch_loss_augmented_inference not implemented") 
        # default implementation of batch loss augmented inference
        return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
                for x, y in zip(X, Y)]
    '''
    def _set_class_weight(self):
        sys.exit("_set_class_weight not implemented") 
        if not hasattr(self, 'size_joint_feature'):
            # we are not initialized yet
            return

        if hasattr(self, 'n_labels'):
            n_things = self.n_labels
        else:
            n_things = self.n_states

        if self.class_weight is not None:

            if len(self.class_weight) != n_things:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(n_things)
            self.uniform_class_weight = True
        
