import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple,Callable

rng = np.random.default_rng()
#np_X = np.array([[n%2,n//2] for n in range(4)])
np_X = np.array([[1.0,0.5]])

np_X

layers=2

def step(res):
    return np.where(res>0,1,0)

def sigmoid(res:np.ndarray):
    return 1/(1+np.exp(res*-1))

def relu(res:np.ndarray):
    return np.where(res>0,res,0)


def list_slider_inputs(label,row_nums=2,col_nums=2):
    cols = st.columns(col_nums)
    res_2nd=[]
    for col_ind,_ in enumerate(cols):
        with cols[col_ind]:
            res_1st=[]
            for row_ind in range(row_nums):
                v=st.slider(f'{label},row_{row_ind},column_{col_ind}',-1.,1.,0.)
                res_1st.append(v)
            res_2nd.append(res_1st)
    return res_2nd


tdtest=np.array([[[1,2,3],[4,5,6],[1,2,3],[4,5,6]],[[26,33,24],[41,52,63],[1,2,3],[4,5,6]]])
tdtest.shape
tdtest
tdtest_T=tdtest.T
tdtest_T.shape
tdtest_T
_perceptron_temp_num=0
def perceptron(np_input,activation=step,col_dim_to=1):
    global _perceptron_temp_num
    _perceptron_temp_num=_perceptron_temp_num+1

    input_w_theta = list_slider_inputs('w and theta',3,4)

    np_W = np.array(input_w_theta)
    np_W
    return
    res=np.dot(np_input,np_W)
    res=res+theta_2nddim
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
#np_W2,res_activation2 = perceptron(np_X,activation=all_activation)
#res_activation1
#res_activation2
#res_1f=np.vstack([res_activation1,res_activation2]).T
#st.write(res_1f)

#st.write(res_1f.shape)
#np_Wf,res_activationf = perceptron(res_1f,activation=all_activation)



fig,ax= plt.subplots(1,1)

temp_c=c=np.arange(np_X.shape[0])
#plt.scatter(np_X[:,0],np_X[:,1],c=temp_c,cmap='hsv')
show_fx(relu)
#plt.plot(np_X[:,0],np_X[:,1])
#plt.scatter(np_X[:,0]*np_W[0],np_X[:,1]*np_W[1],c=temp_c,cmap='hsv')
#plt.plot(np_X[:,0]*np_W[0],np_X[:,1]*np_W[1])

st.pyplot(fig)