import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import multiprocessing 
from datetime import datetime
from DR import DR_Utils
import sys


def load_adj(filename, vNum, no_add_features=True):
    # Reading graphs
    with open(filename) as f:
        content = f.readlines()
        
    content = [x.strip() for x in content]
    if not no_add_features:
        content = [i.split(' ')[:3] for i in content]
    else:
        content = [i.split(' ')[:2] for i in content]
        
    for i, x in enumerate(content):
        content[i] = [int(j) for j in x]
        if no_add_features:
            content[i].append(1)
    #print(content)
    arr = np.array(content)
    
    #shape = tuple(arr.max(axis=0)[:2]+2)
    #if shape[0] != shape[1]:
    shape = (vNum, vNum)
    
    adj = sp.coo_matrix((arr[:, 2], (arr[:, 0], arr[:, 1])), shape=shape,
                        dtype=arr.dtype)
    
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print("Done, finished processing adj matrix...")
    return adj

'''
def load_file(filename):
    print("def load_file...")
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
  
    return content
'''
def load_file(filename,  lineNums):

    #print(lineNums)
    with open(filename) as f:
        content=[]
        lineNum = 0
        while True:
            line = f.readline() 
            if not line: 
                break 
            if lineNum in lineNums:
                content.append(line) 
                #print(line)
                #if (len(content) == size):
                #    return content
            lineNum += 1 
    f.close()
    return content
    #print("no enough data")            
    

def one_hot_encoder(data, max_value):
    shape = (data.size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    
    return one_hot

def one_hot_to_str(x):
    numbers = []
    #print("-----------")
    #print(matrix.shape)
    #index = 0
    for i in range(len(x)):
        if x[i] == 1:
            numbers.append(str(i))
    #print(numbers)
    
    return numbers


'''
def extract_training_sets(filename, dsize=True):
    content = list(filter(None, load_file(filename)))
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
    #print(len(X))
    
    return list(zip(X,Y))
'''

def extract_training_sets(filename, lineNums):
    content = load_file(filename,  lineNums)
    
    #content = list(filter(None, load_file(filename)))
    #print(content)
    X=[]
    Y=[]
    for i, item in enumerate(content):
        #print(item)
        strings=item.split("|")
        topics = strings[0].split()
        nodes = strings[1].split()
        X.append(topics)
        Y.append(nodes)
        #print(y)
    
    '''
    X = [x for i,x in enumerate(content) if i%2==0]
    y = [x for i,x in enumerate(content) if i%2==1]
    
    # Transforming data format
    X = [i.split() for i in X]
    y = [i.split() for i in y]
    '''
    #print(len(X))
    return list(zip(X,Y))

def get_loader(dataset, batch_size, num_workers=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, pairPath, adjPath, nodeNum, topicNum,  lineNums, train=True, dsize=True, full=False):
        self.path = "data/kro/"
        self.dsize = dsize
        self.lineNums = lineNums
        #self.total_size=total_size
        self.data = self.cache(pairPath, adjPath, nodeNum, topicNum,)
        
        
        
    def cache(self, pairPath, adjPath,  nodeNum, topicNum):
        print("Processing dataset...")
        adj = load_adj(adjPath, topicNum)
        sample_data = extract_training_sets(pairPath, self.lineNums)
        data = []
        for datapoint in sample_data:
            # Extract input and target for training and testing
            x_train, y_train = datapoint
            x_train = [int(i) for i in x_train]
            y_train = [int(i) for i in y_train]
            #Transform the input to identity matrix
            # Getting cardnality of the sample
            if self.dsize:
                temp_card = len(x_train)
            #temp_tensor = torch.zeros(topicNum, topicNum)
            #for i in x_train:
            #    temp_tensor[i][i] = 1
           # x_train = temp_tensor
           
            y_train = one_hot_encoder(np.array(y_train), nodeNum)
            y_train = torch.sum(torch.tensor(y_train), dim=0)#/len(y_train)
            y_train = y_train.unsqueeze(-1)
            
            x_train = one_hot_encoder(np.array(x_train), topicNum)
            x_train = torch.sum(torch.tensor(x_train), dim=0)#/len(y_train)
            x_train = x_train.unsqueeze(-1)
            
            #print(x_train.shape)
            #sys.exit()
            if self.dsize:
                data.append((x_train, y_train, adj, temp_card))
            else:
                data.append((x_train, y_train, adj))

        print("Done!")
        return data
    
    def __getitem__(self, item):
        #print(item)
        #print(len(self.data))
        x, y, adj, cardinality = self.data[item]
        return x, y, adj, cardinality
    
    def __len__(self):
        return len(self.lineNums)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def list_to_set(X_list):
    X_set=set()
    for x in X_list:
        X_set.add(str(x))
    return X_set




class GCN(nn.Module):
    def __init__(self, node_size, topic_size, dropout):
        super(GCN, self).__init__()
        
        
        self.gc1 = GraphConvolution(1, 1)
        self.gc2 = GraphConvolution(1, 1)
        self.dropout = dropout
        self.MLP = nn.Sequential(
            nn.Linear(topic_size,  512), 
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
  
            nn.Linear(256, node_size),
        )

        
    def forward(self, x, adj):
        x=x.sum(1)
        x=x.unsqueeze(1)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x=x.squeeze()
        
        x=self.MLP(x)
        return x
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input.float(), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
