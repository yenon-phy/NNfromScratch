#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import sys
sys.path.append('../src')
from common.layers import Affine
from common.layers import Sigmoid
from common.layers import SoftmaxWithLoss


# In[2]:


class NeuralNetwork():
    
    def __init__(self, n_features, n_output, n_hidden=30, l2=0.0, l1=0.0,                  epochs=50, eta=0.001, decrease_const=0.0, shuffle=True,                  n_minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.l2 = l2
        self.l1 = l1
        self.epochs = epochs
        self.eta = eta
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.n_minibatches = n_minibatches
        
        self.params = {}
        self._init_weights()
        
        self.layers = {}
        self.layers['Affine_1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine_2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()
        
        self._loss = []
        self._iter_t = 0
        
    def _init_weights(self):
        ls_nodes = [self.n_features, self.n_hidden, self.n_output]
        scale_1 = np.sqrt(1.0 / ls_nodes[0])
        scale_2 = np.sqrt(1.0 / ls_nodes[1])
  
        self.params['W1'] = scale_1 * np.random.randn(ls_nodes[0], ls_nodes[1])
        self.params['b1'] = np.zeros(ls_nodes[1])   
        self.params['W2'] = scale_2 * np.random.randn(ls_nodes[1], ls_nodes[2])
        self.params['b2'] = np.zeros(ls_nodes[2])  
        
    def predict(self, X):
        for layer in self.layers.values():
            X = layer(X)
        y_hat = X
        return y_hat

    def _calc_loss(self, X, t):
        y_hat = self.predict(X)

        W1, W2 = self.params['W1'], self.params['W2']
        l2_term, l1_term = 0.0, 0.0
        l2_term += 0.5 * self.l2 * np.sum(W1 ** 2)
        l2_term += 0.5 * self.l2 * np.sum(W2 ** 2)
        l1_term += 0.5 * self.l1 * np.abs(W1).sum()
        l1_term += 0.5 * self.l1 * np.abs(W2).sum()
        
        loss = self.last_layer(y_hat, t) + l2_term + l1_term
        return loss

    def accuracy(self, X, t):
        y_hat = self.predict(X)
        y = np.argmax(y_hat, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(X.shape[0])
        return accuracy
    
    def _encode_labels(self, y, n_labels):
        onehot = np.zeros((y.shape[0], n_labels))
        for idx, val in enumerate(y):
            onehot[idx, val] = 1.0
        return onehot
    
    def fit(self, X, y, print_progress=False):
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)
        
        self._loss = []
        
        for i in range(self.epochs):
            self.eta /= (1 + self.decrease_const*i)

            if print_progress:
                sys.stderr.write('\rEpoch: {}/{}'.format(i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[idx]

            batches = np.array_split(range(y_data.shape[0]), self.n_minibatches)
            self._iter_t = 0
            for batch in batches:
                # forward
                X_batch, y_batch = X_data[batch], y_enc[batch]
                loss = self._calc_loss(X_batch, y_batch)
                self._loss.append(loss)
                
                # backward
                delta = 1
                delta = self.last_layer.backward(delta)

                layers = list(self.layers.values())
                layers.reverse()
                for layer in layers:
                    delta = layer.backward(delta)

                # gradients
                grads = {}
                W1 = self.layers['Affine_1'].W
                W2 = self.layers['Affine_2'].W
                
                grads['W1'] = self.layers['Affine_1'].dW                 + self.l2 * W1                 + self.l1 * np.sign(W1)
                grads['b1'] = self.layers['Affine_1'].db

                grads['W2'] = self.layers['Affine_2'].dW                 + self.l2 * W2                 + self.l1 * np.sign(W2)
                grads['b2'] = self.layers['Affine_2'].db
                
                self._update_grads(self.params, grads)
                
        return self
    
    """
    # SGD
    def _update_grads(self, params, grads):
        for key in params.keys():
            params[key] -= self.eta * grads[key] 
            
    """    
    
    # Adam
    def _update_grads(self, params, grads):
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m, v = {}, {}
        for key, val in params.items():
            m[key] = np.zeros_like(val)
            v[key] = np.zeros_like(val)
        
        self._iter_t += 1
        for key in params.keys():
            t = self._iter_t
            # (1 - beta)で因数分解されたAdamの更新式
            m[key] += (1 - beta1) * (grads[key] + m[key])
            m[key] /= 1 - beta1 ** t
            v[key] += (1 - beta2) * (grads[key]**2 + v[key])
            v[key] /= 1 - beta2 ** t
            
            params[key] -= self.eta * m[key] / (np.sqrt(v[key]) + eps) 
    
    @property
    def loss_(self):
        return self._loss
    

