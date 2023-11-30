import streamlit as st
import numpy as np
import pandas as pd
from funcs.funcs import softmax
import matplotlib.pyplot as plt
from pyplus.builtin.mathtool import numerical_derivative
from pyplus.matplotlib.plus import show_fx

def sum_squared_error(y:pd.Series,t:pd.Series)->pd.Series:
    return sum((y-t).apply(lambda x:x**2))/2
input_a1=st.data_editor(pd.DataFrame({'y':[],'t':[]}),num_rows='dynamic')

np_y=input_a1.y.to_numpy()
input_a1['y_softmaxed']=softmax(np_y)
input_a1['logged']=np.log(input_a1['y_softmaxed'])
input_a1['cross_entropy_error']=input_a1['logged']*input_a1['t']*-1
input_a1
error=sum_squared_error(input_a1.y,input_a1.t)
error

    

aa=np.random.choice(input_a1['y'],size=2)
aa
fig,ax=plt.subplots(1,1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
#show_fx(lambda x:np.sin(x))





inp_slid=st.slider('derivative',-5.,5.,0.,0.01)
show_fx(np.sin)
aa=numerical_derivative(np.sin,inp_slid)
show_fx(lambda x:aa*(x-inp_slid)+np.sin(inp_slid))
st.pyplot(fig)