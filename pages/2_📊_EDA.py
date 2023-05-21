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
import plotly.figure_factory as ff

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

# Principais imports
import streamlit as st
import pickle
from PIL import Image
import json
from streamlit_lottie import st_lottie
import time
import Homepage
from streamlit_option_menu import option_menu
#-----------------------------------------------------------------
# Carregando a animação
def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

analise = load_lottiefile("lottiefile/analise.json")

# --------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # Titulo e Animação
    st.sidebar.empty()
    
    #col1,col2 = st.columns([4,6])
    #with col1:
    #    col1.markdown("<h1 style='text-align: center; color: white;padding-top: 40%;'>Análise Exploratória</h1>", unsafe_allow_html=True)
    #with col2:
    #    col2.markdown("<h1 style='padding-left:20%;'>\n</h1>",unsafe_allow_html=True)
    #    st_lottie(analise,speed=1,reverse=False,loop=True,quality="high",
    #                height=400,
    #                width=400,
    #                key=None,)


    # Carregar df
    df = Homepage.carregar_df()
    
    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns
    all_columns = df.columns
    
    # Plots e sidebar itens
    # Visualization
    plot_types = ["Countplot", "Histplot","Describe","Scatter","Boxplot"]
    visuals = st.sidebar.selectbox("Selecione o tipo de visualização",plot_types)

    if "Countplot" in visuals:
        if len(cat_columns) == 0:
            st.sidebar.write("Sem Features categóricas")
        else:
            selected_columns_cat = st.sidebar.selectbox("Selecione a coluna do dataframe",cat_columns)
            st.empty()
            colcount_1,colcount_2,colcount_3 = st.columns([0.3,2.5,0.3])
            with colcount_2:
                sns.set_style("darkgrid") # Config do plot, cores e etc.
                sns.set(rc={"figure.facecolor":"0E1117",
                            "axes.labelcolor":"white",
                            "xtick.color":"white",
                            "ytick.color":"white",
                            "axes.titlecolor":"white",
                            'legend.labelcolor': 'black',
                            })
                if selected_columns_cat == "Intencao_de_emprestimo":
                    sns.countplot(x=df[selected_columns_cat],hue=df["Target_Label"])
                    plt.xticks(fontsize=6)
                    plt.title(f"{selected_columns_cat}")
                    st.pyplot(plt.gcf())
                else:
                    sns.countplot(x=df[selected_columns_cat],hue=df["Target_Label"])
                    plt.title(f"{selected_columns_cat}")
                    st.pyplot(plt.gcf())
            with st.sidebar:
                 with st.spinner("Loading..."):
                    time.sleep(1)
                    st.success("Done!")
         
    if "Histplot" in visuals:
        if len(num_columns) == 0:
            st.sidebar.write("Sem Features numéricos")
        else:
            selected_columns_num = st.sidebar.selectbox("Selecione as colunas do dataframe",num_columns)     
            st.empty()
            colhist_1,colhist_2,colhist_3 = st.columns([0.3,2.5,0.3]) # Centralizar o plot 
            with colhist_2:
                sns.set_style("darkgrid") # Config do plot, cores e etc.
                sns.set(rc={"figure.facecolor":"0E1117",
                            "axes.labelcolor":"white",
                            "xtick.color":"white",
                            "ytick.color":"white",
                            "axes.titlecolor":"white",
                            'legend.labelcolor': 'black',
                            })
                # "figure.figsize":(10,10)
                if selected_columns_num == "Idade" : 
                    plt.xlim(left=20,right=50) # Mudar a escala do eixo x
                    sns.histplot(x=df[selected_columns_num],kde=True,hue=df["Target_Label"])
                    plt.title(f"{selected_columns_num}")
                    st.pyplot(plt.gcf())
                else:
                    sns.histplot(x=df[selected_columns_num],kde=True,hue=df["Target_Label"])
                    plt.title(f"{selected_columns_num}")
                    st.pyplot(plt.gcf())
                with st.sidebar:
                    with st.spinner("Loading..."):
                        time.sleep(1)
                        st.success("Done!") 

    if "Scatter" in visuals:
        selected_scatter_columns = st.sidebar.selectbox("Selecione a primeira feature",all_columns)
        selected_scatter_columns2 = st.sidebar.selectbox("Selecione a segunda feature",all_columns)
        st.empty()
        colscatter1,colscatter2,colscatter3 = st.columns([0.3,2.5,0.3])
        with colscatter2:
            sns.set_style("darkgrid") # Config do plot, cores e etc.
            sns.set(rc={"figure.facecolor":"0E1117",
                            "axes.labelcolor":"white",
                            "xtick.color":"white",
                            "ytick.color":"white",
                            "axes.titlecolor":"white",
                            'legend.labelcolor': 'black',
                        })     
            sns.scatterplot(x=df[selected_scatter_columns],y=df[selected_scatter_columns2],hue=df["Target_Label"])      
            plt.title(f"{selected_scatter_columns} x {selected_scatter_columns2}")
            st.pyplot(plt.gcf())
        with st.sidebar:
            with st.spinner("Loading..."):
                time.sleep(1)
                st.success("Done!") 
    
    if "Boxplot" in visuals:
        selected_boxplot_columns = st.sidebar.selectbox("Selecione a primeira feature",all_columns)
        selected_boxplot_columns2 = st.sidebar.selectbox("Selecione a segunda feature",all_columns)
        st.empty()
        colboxplot1,colboxplot2,colboxplot3 = st.columns([0.3,2.5,0.3])
        with colboxplot2:
            sns.set_style("darkgrid") # Config do plot, cores e etc.
            sns.set(rc={"figure.facecolor":"0E1117",
                            "axes.labelcolor":"white",
                            "xtick.color":"white",
                            "ytick.color":"white",
                            "axes.titlecolor":"white",
                            'legend.labelcolor': 'black',
                        })
            sns.boxplot(x=df[selected_boxplot_columns],y=df[selected_boxplot_columns2],hue=df["Target_Label"])
            plt.title(f"{selected_boxplot_columns} x {selected_boxplot_columns2}")
            st.pyplot(plt.gcf())
        with st.sidebar:
            with st.spinner("Loading..."):
                time.sleep(1)
                st.success("Done!")
                                
    if "Describe" in visuals:
        with st.sidebar:
                with st.spinner("Loading..."):
                    time.sleep(1)
                    st.success("Done!")        
        st.dataframe(df.describe())


    hide_st_style="""
            <style>
            head{visibility:hidden;}
            footer{visibility:hidden;}
            </style>
              """
    st.markdown(hide_st_style,unsafe_allow_html=True)




if __name__=="__main__":
    main()
