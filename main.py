import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple,Callable

rng = np.random.default_rng()
#np_X = np.array([[n%2,n//2] for n in range(4)])
np_X = np.array([[1.0,0.5],[0,1]])

np_X

layers=2

def step(res):
    return np.where(res>0,1,0)

def sigmoid(res:np.ndarray):
    return 1/(1+np.exp(res*-1))

def relu(res:np.ndarray):
    return np.where(res>0,res,0)


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
    
    np_W = np.array(input_w_theta)[:-1,:]
    np_theta = np.array(input_w_theta)[-1,:]

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


all_activation=relu

np_W1,res_activation1 = perceptron(np_X,activation=all_activation,col_dim_to=3)
np_W2,res_activation2 = perceptron(res_activation1,activation=all_activation,col_dim_to=2)
np_W3,res_activation3 = perceptron(res_activation2,activation=all_activation,col_dim_to=2)

fig,ax= plt.subplots(1,1)

temp_c=c=np.arange(np_X.shape[0])
#plt.scatter(np_X[:,0],np_X[:,1],c=temp_c,cmap='hsv')
show_fx(relu)
#plt.plot(np_X[:,0],np_X[:,1])
#plt.scatter(np_X[:,0]*np_W[0],np_X[:,1]*np_W[1],c=temp_c,cmap='hsv')
#plt.plot(np_X[:,0]*np_W[0],np_X[:,1]*np_W[1])

st.pyplot(fig)