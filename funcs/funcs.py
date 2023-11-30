from dataset.mnist import load_mnist
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple,Callable
import matplotlib.pyplot as plt

def step(res):
    return np.where(res>0,1,0)

def sigmoid(res:np.ndarray):
    return 1./(1.+np.exp(res*-1))

def relu(res:np.ndarray):
    return np.where(res>0,res,0)

def idenitity(res:np.ndarray):
    return res

def softmax(res:np.ndarray):
    return np.exp(res) / np.sum(np.exp(res))

def list_slider_inputs(label,row_nums=2,col_nums=2):
    res = np.zeros(shape=(row_nums,col_nums))

    cols = st.columns(col_nums)
    for row_ind in range(row_nums):
        for col_ind,_ in enumerate(cols):
            with cols[col_ind]:
                res[row_ind,col_ind]=st.slider(f'{label},row_{row_ind},column_{col_ind}',-1.,1.,0.)
    return res

_global_count=0
def perceptron(np_input,activation=step,col_dim_to=1):
    global _global_count
    _global_count+=1
    input_w_theta = list_slider_inputs(f'{_global_count}w and theta',np_input.shape[1]+1,col_dim_to)
    
    np_W = input_w_theta[:-1,:]
    np_W
    np_theta = np.expand_dims(input_w_theta[-1,:],axis=0)
    res=np.dot(np_input,np_W)
    res=res+np_theta
    res_activation=activation(res)
    res_activation
    return np_W, res_activation

def show_fx(func:Callable):
    '''
    show function 
    ## Parameters:
    func : function of 1-dim
    like f(x)=x+1.
    ## Examples:
    import matplotlib.pyplot as plt
    fig,ax= plt.subplots(1,1)
    show_fx(lambda x : x+1)
    plt.show(fig)
    '''
    x_range=np.arange(-5,5,0.01)
    fx_range=func(x_range)
    plt.plot(x_range,fx_range)