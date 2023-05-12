#EDA
import pandas as pd
import numpy as np                                
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,OneHotEncoder
#                                                 0 - 1

#Statístics
from scipy import stats
from scipy.stats import norm, skew


#Plot
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Machine Learning 
from sklearn.neural_network import MLPClassifier

# Redução de Dimensionalidade 
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
from sklearn.decomposition import PCA

# Balaceamento De classes
from imblearn.over_sampling import SMOTE

#Clusters
from sklearn.cluster import KMeans


#Split / cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

#Search best params
from sklearn.model_selection import GridSearchCV

#Avaliation metrics
from sklearn.metrics import accuracy_score, classification_report


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pickle
from PIL import Image
import json
from streamlit_lottie import st_lottie

#-----------------------------------------------------------------
# Carregando a animação
def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

computer = load_lottiefile("lottiefile/computer.json")

#--------------------------------------------------------------------
# Interface 

def main():
    st.set_page_config(layout = "wide", page_title='Homepage',page_icon="chart_with_upwards_trend")

    st.sidebar.empty()

    col1,col2 = st.columns([5,5])
    col1.markdown("<h1 style='text-align: center; color: white;padding-top: 40%;'>Risco de Crédito</h1>", unsafe_allow_html=True)
    with col2:
        col2.markdown("<h1 style='padding-left:20%;'>\n</h1>",unsafe_allow_html=True)
        st_lottie(computer,speed=1,reverse=False,loop=True,quality="high",
                    height=400,
                    width=400,
                    key=None,)
    st.markdown("---")

    st.header("Projeto de Portfólio Risco de crédito")
    st.write("")
    st.write("Fonte : https://www.kaggle.com/datasets/laotse/credit-risk-dataset")
    
    hide_st_style="""
            <style>
            footer{visibility:hidden;}
            </style>
              """
    st.markdown(hide_st_style,unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def carregar_df():
    with open('./Databases/df_traduzido.pkl','rb') as f:
        df_traduzido=pickle.load(f)
    return df_traduzido

def display_df(df_traduzido):
    use_dtf = st.checkbox('Mostrar DataFrame')
    if use_dtf:
        st.dataframe(df_traduzido)
        st.write(f"Linhas - {df_traduzido.shape[0]}")
        st.write(f"Colunas - {df_traduzido.shape[1]}")



#-----------------------------------------------------------------


if __name__ == "__main__":
    main()
    display_df(carregar_df())