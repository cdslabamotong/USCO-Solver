# -*- coding: utf-8 -*-

import os
import math
import random
import json

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
#import h5py
import numpy as np
import scipy.optimize
import torch
import multiprocessing
from datetime import datetime
from SSP import SSP_Utils


def get_loader(dataset, batch_size, num_workers=1, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,pair_path, size, batch_size, lineNums , vNum, train=True, full=False, max_size=None):
        self.train = train
        self.full = full
        self.size=size
        self.max_size=max_size
        self.data = self.cache(pair_path, lineNums,vNum)
        #self.max = 20
        self.batch_size=batch_size
        

    def cache(self, pair_path,lineNums, vNum):
        print("Processing dataset...")
        sample_data, max_size = extract_training_sets(pair_path,self.size,lineNums)
        if self.max_size == None:
            self.max_size=max_size 
        data = []
        #print(len(sample_data))
        for datapoint in sample_data:
            set_train, label = datapoint
            set_train = [int(i) for i in set_train]
            label = [int(i) for i in label]
            
            
            #if not self.train:
                #set_train.sort()
                #print("set_train {}".format(set_train))
            
            set_train = one_hot_encoder(np.array(set_train), vNum ,max_size).transpose()
            #print("set_train.shape {}".format(set_train.shape))

            #set_train = [np.array(set_train).transpose()]
            #print(set_train.shape)
            label = one_hot_encoder(np.array(label), vNum,max_size).transpose()
            
            #if not self.train:
                #set_train.sort()
                #print("set_train.shape {}".format(set_train.shape))
                #label.sort()
                #print(label)
                #a=utils.one_hot_to_number(torch.tensor(set_train))
                #a.sort()
                #print(a)
                #print("-------")
            #print("label shape {}".format(label.shape))
            #label = np.array(label).transpose()
            data.append((torch.Tensor(set_train), torch.Tensor(label), label.shape[0]))
            #torch.save(data, cache_path)
        print("Done!")
        return data
    

    def __getitem__(self, item):
        sys.exit("__getitem__ not implemented")


    def __len__(self):
        return self.size

def extract_training_sets(filename, size,lineNums):
    content, max_size = load_file(filename, size, lineNums)
    
    #content = list(filter(None, load_file(filename)))
    #print(content)
    X=[]
    Y=[]
    for i, item in enumerate(content):
        #print(item)
        strings=item.split()
        x = [strings[0],strings[1]]
                #print(x)
        X.append(x)
        y=strings[3:]
        Y.append(y)
        #print(y)
    
    '''
    X = [x for i,x in enumerate(content) if i%2==0]
    y = [x for i,x in enumerate(content) if i%2==1]
    
    # Transforming data format
    X = [i.split() for i in X]
    y = [i.split() for i in y]
    '''
    #print(len(X))
    return list(zip(X,Y)), max_size


def load_file(filename, size, lineNums):
    max_size=0
    #print(lineNums)
    with open(filename) as f:
        content=[]
        lineNum = 0
        while True:
            line = f.readline() 
            if not line: 
                break 
            if lineNum in lineNums:
                if len(line.split())>max_size:
                    max_size=len(line.split())
                content.append(line) 
                #print(line)
                if (len(content) == size):
                    return content, max_size-3
            lineNum += 1 
    print("no enough data")            
    


def one_hot_encoder(data, max_value, max_size):
    shape = (max_size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    #print(one_hot.shape)
    #print(one_hot)
    #input("wait")
    
    return one_hot




class DSPN(nn.Module):

    def __init__(self, encoder,  set_channels,max_set_size,  iters, lr, batch_size):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.
        channels: Number of channels of the set to predict.
        max_set_size: Maximum size of the set.
        iter: Number of iterations to run the DSPN algorithm for.
        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr
        self.batch_size = batch_size
        self.set_channels=set_channels

        self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
        
        #self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))
        #self.linear = nn.Linear(set_channels*max_set_size, max_set_size)

    def forward(self, target_repr, max_set_size):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.
        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        # copy same initial set over batch
        
        current_set = self.starting_set.expand(
            target_repr.size(0), *self.starting_set.size()[1:]
        ).detach().cpu()
        #current_set = self.starting_set
        #print(current_set.shape)
        #current_mask = self.starting_mask.expand(
        #    target_repr.size(0), self.starting_mask.size()[1]
        #)
        
        #current_set = self.starting_set 
        # make sure mask is valid
        #current_mask = current_mask.clamp(min=0, max=1)
        
        # info used for loss computation
        intermediate_sets = [current_set]
        #intermediate_masks = [current_mask]
        # info used for debugging
        repr_losses = []
        grad_norms = []
        
                    #self.starting_set.requires_grad = True
        for i in range(self.iters):
            # regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
            with torch.enable_grad():
                if not  self.training or True:
                    current_set.requires_grad = True

                predicted_repr = self.encoder(current_set)
                repr_loss = F.smooth_l1_loss(
                    predicted_repr, target_repr, reduction="mean"
                )
                
                # change to make to set and masks to improve the representation
                set_grad = torch.autograd.grad(
                    inputs=[current_set],
                    outputs=repr_loss,
                    only_inputs=True,
                    create_graph=True,
                )[0]


            
            current_set = current_set - self.lr * set_grad
            
            
            current_set = current_set.detach().cpu()

            repr_loss = repr_loss.detach().cpu()
            set_grad = set_grad.detach().cpu()

                
            # keep track of intermediates
            #print(current_set.shape)
            #print(current_set.sum(2).shape)
            intermediate_sets.append(current_set.sum(2))
            #intermediate_masks.append(current_mask)
            repr_losses.append(repr_loss)
            grad_norms.append(set_grad.norm())
        
        '''
        for i in range(len(intermediate_sets)):
            intermediate_sets[i] = self.linear(intermediate_sets[i].view(intermediate_sets[i].shape[0], -1))
            #intermediate_sets[i] = intermediate_sets[i].div_(torch.norm(intermediate_sets[i],2))
            intermediate_sets[i] = F.normalize(intermediate_sets[i], dim=1)

        '''
        return intermediate_sets, None, repr_losses, grad_norms
      

def build_net(args):
    set_channels = args.vNum
    set_size = args.set_size

    output_channels = 256

    set_encoder_dim=1024
    input_encoder_dim = 256

    inner_lr = args.inner_lr
    iters = args.iters

    
    set_encoder = MLPEncoderInput(set_channels, output_channels, set_size, set_encoder_dim)

    set_decoder = DSPN(set_encoder, set_channels, set_size, iters, inner_lr,args.batch_size)

    
    input_encoder = MLPEncoderInput(set_channels, output_channels, set_size, input_encoder_dim)
    
    net = Net(
        input_encoder=input_encoder, set_encoder=set_encoder, set_decoder=set_decoder
    )
    return net        
    
class Net(nn.Module):
    def __init__(self, set_encoder, set_decoder, input_encoder=None):
        """
        In the auto-encoder setting, don't pass an input_encoder because the target set and mask is
        assumed to be the input.
        In the general prediction setting, must pass all three.
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.input_encoder = input_encoder
        self.set_decoder = set_decoder
        
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, max_set_size):
        if self.input_encoder is None:
            # auto-encoder, ignore input and use target set and mask as input instead
            print("HERE 1")
            #latent_repr = self.set_encoder(input, target_mask)
            #print("HERE 2")

            #target_repr = self.set_encoder(target_set, target_mask)
        else:
            #print("HERE 3")
            # set prediction, use proper input_encoder
            latent_repr = self.input_encoder(input)
            # note that target repr is only used for loss computation in training
            # during inference, knowledge about the target is not needed
            target_repr = self.set_encoder(target_set)
        #print("target_repr.shape {}".format(target_repr.shape))
        predicted_set = self.set_decoder(latent_repr, max_set_size)
        return predicted_set, (latent_repr, target_repr)


############
# Encoders #
############

    
class MLPEncoderInput(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim), 
            nn.ReLU(),

            nn.Linear(dim, dim),
            nn.ReLU(),
            
            nn.Linear(dim, 256),
            nn.ReLU(),

            nn.Linear(256, output_channels),
        )

    def forward(self, x, mask=None):
        x1=x.sum(2)
        x = self.model(x1)
        return x

class MLPEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim), 
            nn.ReLU(),
            #nn.Linear(dim, dim),
            #nn.ReLU(),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x, mask=None):
        x1=x.sum(2)
        x = self.model(x1)
        return x




############
# Decoders #
############



def hungarian_loss(predictions, targets, thread_pool):
    # predictions and targets shape :: (n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(1)

    squared_error_np = squared_error.detach().cpu().numpy()
    indices = thread_pool.map(hungarian_loss_per_sample, squared_error_np)
    losses = [
        sample[row_idx, col_idx].mean()
        for sample, (row_idx, col_idx) in zip(squared_error, indices)
    ]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss


def hungarian_loss_per_sample(sample_np):
    return scipy.optimize.linear_sum_assignment(sample_np)

'''

def one_hot_encoder(data, max_value):
    shape = (data.size, max_value+1)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    return one_hot
'''

def chamfer_loss(predictions, targets):
    # predictions and targets shape :: (k, n, c, s)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (k, n, s, s)
    squared_error = F.smooth_l1_loss(predictions, targets.expand_as(predictions), reduction="none").mean(2)
    loss = squared_error.min(2)[0] + squared_error.min(3)[0]
    return loss.view(loss.size(0), -1).mean(1)
 

def tensor_to_set(my_tensor):
    my_tensor = my_tensor.numpy()
    set_list = []
    for i in my_tensor:
        temp_list = []
        for j in i:
            temp_list.append(np.argmax(j))
        set_list.append(set(temp_list))
    
    return set_list


def one_hot_to_number(matrix):
    numbers = []
    #print("-----------")
    #print(matrix.shape)
    for i in torch.transpose(matrix,0,1):
        number = torch.argmax(i).item()
        if number not in numbers:
            numbers.append(number)
    #print(numbers)
    
    return numbers

def one_hot_to_str(matrix):
    numbers = []
    #print("-----------")
    #print(matrix.shape)
    #sys.exit(torch.transpose(matrix,0,1))
    #print("!!!-----------")
    for i in torch.transpose(matrix,0,1):
        if torch.max(i).numpy() > 0:
            number = torch.argmax(i).item()
            #print(number)
            #if str(number) not in numbers:
            numbers.append(str(number))
    #print("!!!-----------")
    #sys.exit()
    #print(numbers)
    return numbers

def one_hot_to_sortedDic(matrix):
    sys.exit("one_hot_to_sortedDic")
    #snumber = []
    #print("-----------")
    #print(matrix.shape)
    result = {}
    x = matrix.tolist()
    for i in range(len(x)):
        result[str(i)]=x[i]
        
    #for i in torch.transpose(matrix,0,1):
     #   result[torch.argmax(i).item()]=x[i]
    
    
    return result


def matrix_to_one_hot(matrix, target_number):
    #print("matrix_to_one_hot")
    #print(matrix.shape)
    indices = torch.argmax(matrix, dim=0)
    #print(indices)
    #print(indices)
    indices_tensor = torch.zeros(target_number)
    #print("matrix_to_one_hot")
    for i in indices:
        indices_tensor[i] = 1
    
    return indices_tensor
    

def scatter_masked(tensor, mask, binned=False, threshold=None):
    s = tensor[0].detach().cpu()
    mask = mask[0].detach().clamp(min=0, max=1).cpu()
    if binned:
        s = s * 128
        s = s.view(-1, s.size(-1))
        mask = mask.view(-1)
    if threshold is not None:
        keep = mask.view(-1) > threshold
        s = s[:, keep]
        mask = mask[keep]
    return s, mask


def outer(a, b=None):
    """ Compute outer product between a and b (or a and a if b is not specified). """
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b

def testing(X_test,Y_test,Y_pred,args, instance,infTimes=1080):
    #print("Testing Started")

    thread = args.thread;
    block_size =int (infTimes/thread);
    p = multiprocessing.Pool(thread)
   
    influence_Xs = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes) for i in range(thread)),1)
    p.close()
    p.join()
   
    p = multiprocessing.Pool(thread)
    influence_Ys = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_test[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
   
    p = multiprocessing.Pool(thread)
    influence_Y_preds = p.starmap(instance.testInfluence_0_block, ((X_test[i*block_size:(i+1)*block_size], infTimes, Y_pred[i*block_size:(i+1)*block_size]) for i in range(thread)),1)
    p.close()
    p.join()
   
   
    influence_X=[]
    influence_Y=[]
    influence_Y_pred=[]
    for i in range(thread):
        influence_X.extend(influence_Xs[i])
        influence_Y.extend(influence_Ys[i])
        influence_Y_pred.extend(influence_Y_preds[i])
   
   
    reduce_percent_opt=[]
    reduce_percent_pre = []
    com_to_opt = []
    error_abs = []
    error_ratio = []
    for influence_x, influence_y, influence_y_pred in zip(influence_X, influence_Y, influence_Y_pred):
        #print("{} {} {} ".format(influence_x,influence_y,influence_y_pred))
        reduce_percent_opt.append((influence_x-influence_y)/influence_x)
        reduce_percent_pre.append( (influence_x-influence_y_pred)/influence_x)
        com_to_opt.append((influence_x-influence_y_pred)/(influence_x-influence_y+0.01))
        error_abs.append((influence_y_pred-influence_y))
        error_ratio.append((influence_y_pred-influence_y)/influence_y)
        #print()
    print(args.dataname)
    print("error_abs: {} +- {}".format(np.mean(np.array(error_abs)), np.std(np.array(error_abs))))
    print("error_ratio: {} +- {}".format(np.mean(np.array(error_ratio)), np.std(np.array(error_ratio))))
    print("reduce_percent_opt: {} +- {}".format(np.mean(np.array(reduce_percent_opt)), np.std(np.array(reduce_percent_opt))))
    print("reduce_percent_pre: {} +- {}".format(np.mean(np.array(reduce_percent_pre)), np.std(np.array(reduce_percent_pre))))
    print("com_to_opt: {} +- {}".format(np.mean(np.array(com_to_opt)), np.std(np.array(com_to_opt))))
    
    print("trainNum:{}, testNum:{}, infTimes:{} ".format(args.trainNum, args.testNum,  infTimes))
    
    if args.output:
        now = datetime.now()
        with open(now.strftime("%d-%m-%Y %H:%M:%S"), 'a') as the_file:
            for x_test, y_test, y_pred in zip(X_test,Y_test,Y_pred):
                for target in [x_test, y_test, y_pred]:
                    line='';
                    for a in target:
                        line += a
                        line += ' '
                    line += '\n'
                    the_file.write(line)
                the_file.write('\n')


    print("===============================================================")
    
def runTesting(X_test, Y_test, Y_pred_sortedNodes, args, instance):
    #print("runTesting")
    #X_test_set=[]
    #Y_test_set=[]
    Y_pred = []
    '''
    for x, y, y_hat in zip (X_test, Y_test, Y_pred):
        X_test_set.append(vec_to_set(x))
       # Y_test_set.append(vec_to_set(y))
        Y_pred_set.append(vec_to_set(y_hat))
    '''
    
    p = multiprocessing.Pool(args.thread)
            #print("222")
    Y_pred=p.starmap(SSP_Utils.prediction_from_sortedNodes_ver2, ((instance.stoGraph.unitGraph, x, y) for x, y in zip(X_test, Y_pred_sortedNodes)))
    #print("333")
    p.close()
    p.join()
    '''       
    for x, y in zip(X_test, Y_pred_sortedNodes):
        #print(y)
        Y_pred.append(SSP_Utils.prediction_from_sortedNodes_ver1(instance.stoGraph.unitGraph, x, y))
      
    print("++++++++++++")
    for x, y, y_hat in zip (X_test, Y_test, Y_pred):
        print(x)
        print(y)
        print(y_hat)
        print()
    print("++++++++++")
    '''
    #test_batch(self, X_test, Y_test_length, Y_pred, testBatch, testNum, logpath = None, preTrainPathResult = None):
    Y_test_length = []
    for x, y in zip(X_test,Y_test):
        Y_test_length.append(instance.stoGraph.EGraph.pathLength(x,y))
        
    instance.test_batch(X_test, Y_test_length, Y_pred, args.testBatch, args.testNum, logpath = args.logpath)


    #testing(X_test_set,Y_test_set,Y_pred_set,args, instance)
    
        
def vec_to_set(X):
    y=set()
    for x in X:
        y.add(str(x))
    return y


if __name__ == "__main__":
    pass