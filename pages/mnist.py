from dataset.mnist import load_mnist
import streamlit as st
import pandas as pd
import numpy as np
from funcs.funcs import perceptron,relu

(x_train,t_train),(x_test,t_test)=load_mnist()
x_train

x_train.shape
datas = pd.DataFrame({'t':t_train})

#np_W , datas = perceptron(x_train,relu,1)

#datas['x']=[np.array(img).reshape(28,28) for img in x_train]
#st.dataframe(data=datas,column_config={'x':st.column_config.ImageColumn('preview')})
#datas
#st.image(x_train[0].reshape(28,28))
#t_train[0]