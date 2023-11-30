import streamlit as st
import numpy as np
import pandas as pd
from funcs.funcs import show_fx,softmax
import matplotlib.pyplot as plt

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
#show_fx(lambda x:np.sin(x))

def deriv(func,current_x:float):
    def numerical_diff(func,current_x,delta=1):
        return (func(current_x+delta)-func(current_x-delta))/(2*delta)
    def mean_weight(nums,weights):
        np_nums=np.array(nums)
        np_weights=np.array(weights)
        return sum(np_nums*np_weights)/sum(np_weights)
    
    dic_res={'slope':[],'weight':[]}
    delt=4
    while True:
        delt = delt/2
        slope=numerical_diff(func,current_x,delt)
        
        #ax.plot([current_x,current_x+delt],[func(current_x),func(current_x+delt)])

        dic_res['slope'].append(slope)
        dic_res['weight'].append(1./delt)
        

        yield mean_weight(dic_res['slope'],dic_res['weight'])


inp_slid=st.slider('derivative',-5.,5.,0.,0.01)
sss=deriv(np.sin,inp_slid)

for ind,val in zip(range(10),sss):
    #ls=np.linspace(-5.,5.,0.01)
    #plt.scatter(ls,np.sin(ls))
    plt.scatter(ind,val)



st.pyplot(fig)