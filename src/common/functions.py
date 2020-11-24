#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


# In[3]:


def sigmoid_deriv(x):
    y = x * (1 - x)
    return y


# In[4]:


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    y = exp_x / sum_exp_x
    return y


# In[5]:


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
        t = np.expand_dims(t, axis=0)
        
    batch_size = y.shape[0]
    idx_y = np.arange(batch_size)   
    
    # クラスラベルt（0以上の整数）がone-hotエンコーディング済の場合    
    if t.size == y.size:
        idx_t = t.argmax(axis=1)  # (0, 0, 0, 1, 0) -> 3
    else:
        idx_t = t[0]
             
    eps = 1e-7
    delta = -np.sum(np.log(y[idx_y, idx_t] + eps))
    delta /= batch_size
    return delta

