######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3-clause
#
# Implements structured SVM as described in Joachims et. al.
# Cutting-Plane Training of Structural SVMs
import numpy as np
#from sklearn.externals.joblib import Parallel, delayed
#from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from basic_inference import objective_primal

import multiprocessing
import sys

class BaseSSVM(BaseEstimator):
    """ABC that implements common functionality."""
    def __init__(self, model, max_iter=100, C=1.0, verbose=0,
                 n_jobs=1, show_loss_every=0, logger=None):
        self.model = model
        self.max_iter = max_iter
        self.C = C
        self.verbose = verbose
        self.show_loss_every = show_loss_every
        self.n_jobs = n_jobs
        self.logger = logger

    def predict(self, X, inferenceFeaNum, constraints=None):
        """Predict output on examples in X.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.

        constraints : None or a list of hard logic constraints

        Returns
        -------
        Y_pred : list
            List of inference results for X using the learned parameters.

        """
        #if inferenceFeaNum != None:
        dic = {}
        for i in range(len(self.w)):
            dic[i]=self.w[i]
            
        dic1={k: v for k, v in sorted(dic.items(), key=lambda item: item[1],reverse=True)}
        #print(dic1) 
        
        w_new=np.zeros(len(self.w))
        #print(self.w) 
        #print(w_new) 
        
        count = 0;
        for x in dic1:
            if count < inferenceFeaNum:
                 w_new[x]=self.w[x]
                 count += 1
            else:
                 break;
        #print(w_new)   
        #print(self.w)
        #input("Press Enter to continue...")
        
        #for i in range(inferenceFeaNum):
           
            
        verbose = max(0, self.verbose - 3)
        #if self.n_jobs != 1:
        if constraints:
            sys.exit("no constraints")
            #p = multiprocessing.Pool(self.n_jobs)
            #prediction = sum(p.starmap(self.model.inference, ((x, self.w, constraints=c)for x, c in zip(X, constraints) )))
            #p.close()
            #p.join()

            #prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            #    delayed(inference)(self.model, x, self.w, constraints=c)
            #    for x, c in zip(X, constraints))
        else:
            Y= self.model.batch_predict(X,w_new,n_jobs=self.n_jobs)
            '''
            p = multiprocessing.Pool(self.n_jobs)
            block_size =int (len(X)/self.n_jobs)
            Y = []
            Ys = p.starmap(self.model.inference_block, ((X[i*block_size:(i+1)*block_size], w_new) for i in range(self.n_jobs) ))
            for y_temp in Ys:
                Y.extend(y_temp)
            p.close()
            p.join()
            '''
            #prediction = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            #    delayed(inference)(self.model, x, self.w) for x in X)
        return Y
        '''
        else:
            sys.exit("parallel prediction preferred")
            if hasattr(self.model, 'batch_inference'):
                if constraints:
                    return self.model.batch_inference(X, w_new,
                                                      constraints=constraints)
                else:
                    return self.model.batch_inference(X, w_new)
            if constraints:
                return [self.model.inference(x, w_new, constraints=c)
                        for x, c in zip(X, constraints)]
            return [self.model.inference(x, w_new) for x in X]
        '''
    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of model.loss)
        over X and Y.

        Parameters
        ----------
        X : iterable
            Evaluation data.

        Y : iterable
            True labels.

        Returns
        -------
        score : float
            Average of 1 - loss over training examples.
        """
        if hasattr(self.model, 'batch_loss'):
            losses = self.model.batch_loss(Y, self.predict(X))
        else:
            losses = [self.model.loss(y, y_pred)
                      for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

    def _compute_training_loss(self, X, Y, iteration):
        # optionally compute training loss for output / training curve
        if (self.show_loss_every != 0
                and not iteration % self.show_loss_every):
            if not hasattr(self, 'loss_curve_'):
                self.loss_curve_ = []
            display_loss = 1 - self.score(X, Y)
            if self.verbose > 0:
                print("current loss: %f" % (display_loss))
            self.loss_curve_.append(display_loss)

    def _objective(self, X, Y):
        if type(self).__name__ == 'OneSlackSSVM':
            variant = 'one_slack'
        else:
            variant = 'n_slack'
        return objective_primal(self.model, self.w, X, Y, self.C,
                                variant=variant, n_jobs=self.n_jobs)
