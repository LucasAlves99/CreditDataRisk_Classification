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
# ------------------------------------------------------------------------------------------------
# carregar Animação
def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)

graph = load_lottiefile("lottiefile/graph.json")


# ------------------------------------------------------------------------------------------------
# Funções
def carregar_modelo():
    with open("./Modelo/XGB_Classifier.sav","rb") as f:
        model = pickle.load(f)
    return model

def carregar_scaler():
    with open("./Scaler/Scaler.sav","rb") as f:
        scaler = pickle.load(f)
    return scaler

def carregarLabelEncoderppc():
    with open("./Encoders/lblPropriedade_da_casa","rb") as f:
        lblPropriedade_da_casa = pickle.load(f)
    return lblPropriedade_da_casa 

def carregarLabelEncoderie():
    with open("./Encoders/lblIntencao_de_emprestimo","rb") as f:
        lblIntencao_de_emprestimo = pickle.load(f)
    return lblIntencao_de_emprestimo 


def carregarOneHotEncoder ():
    with open("./Encoders/OHEncoder","rb") as f:
        OHEncoder = pickle.load(f)
    return OHEncoder


df = Homepage.carregar_df()

# -----------------------------------------------------------------------------------------------------------
def main():
    # Animação
    
    col1,col2 = st.columns([4,6])
    with col1:
        col1.markdown("<h1 style='text-align: center; color: white;padding-top: 40%;'>Previsão de Inadimplência</h1>", unsafe_allow_html=True)
    with col2:
        col2.markdown("<h1 style='padding-left:20%;'>\n</h1>",unsafe_allow_html=True)
        st_lottie(graph,speed=1,reverse=False,loop=True,quality="high",
                    height=400,
                    width=400,)
       
    st.markdown("---")

    cols = ["Idade","Renda_Anual",
    "Tempo_de_trabalho_em_anos","Montante_do_emprestimo",
    "Taxa_de_juro","Propriedade_da_casa","Intencao_de_emprestimo",
    "Grau_do_emprestimo"]

    oc = ["Idade","Renda_Anual","Tempo_de_trabalho_em_anos","Montante_do_emprestimo","Taxa_de_juro"]

    # sidebar inputs
    intencao = st.sidebar.select_slider("Selecione a intenção do empréstimo",df["Intencao_de_emprestimo"].unique())
    grau_emprestimo = st.sidebar.select_slider("Selecione o grau do empréstimo",df["Grau_do_emprestimo"].unique())
    tipo_casa = st.sidebar.select_slider("Selecione a Propriedade da casa",df["Propriedade_da_casa"].unique())
    idade = st.sidebar.number_input("Digite a idade",min_value=20,max_value=100,value=20,)
    renda_anual = st.sidebar.number_input("Digite a renda anual",min_value=4800,max_value=1000000,value=50000)
    tempo_de_trabalho = st.sidebar.number_input("Tempo de trabalho em anos",min_value=0,max_value=50,value=3)
    montante_do_emprestimo = st.sidebar.number_input("Montante do emprestimo",min_value=500,max_value=50000,value=10000)
    taxa_de_juro = st.sidebar.number_input("Taxa de juro",min_value=0,max_value=30,value=5)
        

    btn = st.sidebar.button("Predict")
    if btn:
        modelo = carregar_modelo()
        scaler = carregar_scaler()
        oheGrau_emprestimo = carregarOneHotEncoder()
        lbl_Propriedade_casa = carregarLabelEncoderppc()
        lbl_Intencao_emprestimo = carregarLabelEncoderie()

    # One Hot Encoding
        df_ohe = pd.DataFrame([grau_emprestimo],columns=["Grau_do_emprestimo"])
        ohe = oheGrau_emprestimo.transform(df_ohe).todense()
        column_names=["Grau_do_emprestimo_A","Grau_do_emprestimo_B","Grau_do_emprestimo_C","Grau_do_emprestimo_D","Grau_do_emprestimo_E",
              "Grau_do_emprestimo_F","Grau_do_emprestimo_G"]
        df_ohe = pd.DataFrame(ohe,columns=column_names)
    # Label Encoding
    #Propriedade_da_casa
        lblpc = pd.DataFrame([tipo_casa],columns=["Propriedade_da_casa"])
        lblpc = lbl_Propriedade_casa.transform(lblpc)
        lblpc = pd.DataFrame(lblpc,columns=["Propriedade_da_casa"])
    # Intencao_de_emprestimo
        lblie = pd.DataFrame([intencao],columns=["Intencao_de_emprestimo"])
        lblie = lbl_Intencao_emprestimo.transform(lblie)
        lblie = pd.DataFrame(lblie,columns=["Intencao_de_emprestimo"])
    # Restantes
        other_cols = pd.DataFrame([[idade,renda_anual,tempo_de_trabalho,montante_do_emprestimo,taxa_de_juro]],columns=["Idade","Renda_Anual","Tempo_de_trabalho_em_anos","Montante_do_emprestimo","Taxa_de_juro"])
    # Juntando tudo no dataframe final
        final_df = pd.concat([other_cols,lblpc,lblie,df_ohe],axis=1)
        columns=["Grau_do_emprestimo","Propriedade_da_casa","Intencao_de_emprestimo","Idade","Renda_Anual","Tempo_de_trabalho_em_anos","Montante_do_emprestimo","Taxa_de_juro"]

    # Scalonando o resultado dos inputs
        scaler = carregar_scaler()
        final_df=scaler.transform(final_df)

    # predict
        model = carregar_modelo()
        result_proba = model.predict_proba(final_df)
        result = model.predict(final_df)
        with st.sidebar:
            with st.spinner("Loading..."):
                time.sleep(1)
                st.success("Done!")
        if result == 1:
            st.success('Inadimplente', icon="❌")
            # st.markdown("<h2 style='text-align: center; color: white;padding-top: 40%;'>Inadimplente</h2>", unsafe_allow_html=True)
            # Previsão com probabilidade
            # st.write(f"Inadimplente {result_proba[0][1]*100:.2f} %")
        else: 
            st.success('Não Inadimplente', icon="✅")
            # st.markdown("<h2 style='text-align: center; color: white;padding-top: 40%;'>Não Inadimplente</h2>", unsafe_allow_html=True)
            # Previsão com probabilidade
            # st.write(f"Não inadimplente {result_proba[0][0]*100:.2f} %")            



    hide_st_style="""
        <style>
        head{visibility:hidden;}
        footer{visibility:hidden;}
        </style>
             """
    st.markdown(hide_st_style,unsafe_allow_html=True)

if __name__=='__main__':
    main()