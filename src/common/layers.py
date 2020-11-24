#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import sys
sys.path.append('../src')
from common.functions import *


# In[2]:


class Affine:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None
    
    def __call__(self, X):
        self.X = X
        next_X = np.dot(self.X, self.W) + self.b
        return next_X

    def backward(self, delta):
        next_delta = np.dot(delta, self.W.T)
        self.dW = np.dot(self.X.T, delta)
        self.db = np.sum(delta, axis=0)
        return next_delta
    


# In[3]:


class Sigmoid:
    
    def __init__(self):
        self.X = None
        
    def __call__(self, X):
        next_X = sigmoid(X)
        self.X = next_X
        return next_X
    
    def backward(self, delta):
        next_delta = delta * sigmoid_deriv(self.X)
        return next_delta
    


# In[4]:


class SoftmaxWithLoss:
    
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def __call__(self, X, t):
        self.y = softmax(X)
        self.t = t
        loss = cross_entropy_error(self.y, self.t)
        self.loss = loss
        return loss
    
    def backward(self, delta=1):
        batch_size = self.t.shape[0]
        # クラスラベルt（0以上の整数）がone-hotエンコーディング済の場合
        if self.t.size == self.y.size: 
            next_delta = (self.y - self.t) / batch_size
        else:
            next_delta = self.y.copy()
            idx_y = np.arange(batch_size)
            idx_t = self.t
            next_delta[idx_y, idx_t] -= 1
            next_delta /= batch_size
        return next_delta
    

