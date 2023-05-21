# Classificação de Risco de Crédito - Inadimplência

[![NPM](https://img.shields.io/npm/l/react)](https://github.com/LucasAlves99/CreditDataRisk_Classification/blob/main/LICENSE) 

# Sobre o problema
Projeto de previsão de inadimplência , que tem como objetivo, reduzir o número de clientes que não pagam os empréstimos.

Essa Classificação se dá da seguinte forma. Baseado nos dados de clientes inadimplentes e adimplentes passados, o alogritmo consegue efetuar 
uma previsão de um próximo cliente, dizendo se ele pode ser um possível cliente inadimplente ou não.

## Detalhes do Projeto
O projeto está dividido em 7 partes:
### Parte 1: Entendimento dos dados

* Comecei renomeando as colunas para ter uma melhor interpretação ao decorrer do projeto.

* Bucando valores nulos, uniques de cada coluna e quais features eram categóricas e númericas , além da descrição dos dados.

### Parte 2: EDA

* Plot de gráficos de frequência das features categóricas e numéricas,correlação e a utilização do scatter plot para as features com maiores correlação.

* Plot com focos no target (identificando que a classe está desbalanceada, além de observar alguns comportamentos que são mais ocorrentes em clientes inadimplentes, como idade entre 20 e 25 anos e taxa de juro de 15% por exemplo)

### Parte 3: Pre-Processamento
* Tratamento dos nulos.
* Indentificação de outliers com o modelo já treinado do Pyod https://pyod.readthedocs.io/en/latest/
    * Análise univariada e bivariada para observar se ainda restavam outliers e excluí-los 
* Encoding 
    * One Hot Encoding para atributos categóricos ordinais (representam grandeza)   
    * Label Encoding para os atributos qualitativos nominais restantes
* Utilização do Smote para o Balanceamento de classe 
* Split dos dados
* Utilização do StandardScaler para o escalonamento dos dados
### Parte 4: Modelagem
* Majority Learning , modelo base 78%
* Feature Selection 
* Tunning dos parâmetros com , GridSearch/Cross Validation
* Testes de Hipótese
    * Shapiro Wilk Test para analisar se os resultados estão normalizados 
    * Teste Anova e Tukey para observar se os modelos são parecidos, nesse caso não são, então faz diferença a escolha do algoritmo
* Treinando os modelos com seus melhores parâmetros e Modelos com parâmetros default

### Parte 5: Serialização
* Salvei dois modelos, um foi uma Rede Neural com parâmetros default ( 88% acurácia , 88% f1-score) e o outro XGBoost com tunning nos parâmetros (95% acurácia, 95% f1-score)
* Salvando o DataFrame, encoders e scalers para ser utilizado posteriormente na interface gráfica streamlit

### Parte 6: Streamlit
* O web aplicativo streamlit possui 3 páginas
   * *Homepage* : Página de entrada 
   * *EDA* : Análise Rápida com vários tipos de plots
   * *Predict* : Previsão
### Parte 7: Deploy
* Deploy no render: https://la99creditriskclassification.onrender.com/

# Primeiros passos

## Pre-requisitos

- Instalar as bibliotecas

```
pip install -r requirements.txt
```

## Executar o Projeto

- Entrar na pasta do projeto pelo cmd

```
cd caminho_do_projeto
```

- Executar o programa

```
streamlit run Homepage.py
```
![image](https://user-images.githubusercontent.com/50807648/226122129-964dee2b-095c-4221-9c22-f25a47461839.png)

- Ou acesse : https://la99creditriskclassification.onrender.com/
# Construído com
* [Pandas](https://pandas.pydata.org/) - Manipulação de Dados
* [Scikit-learn](https://scikit-learn.org/stable/) - Modelagem
* [Seaborn](https://seaborn.pydata.org/index.html) - Visualização de Dados
* [Matplotlib](https://matplotlib.org/) - Visualização 
* [Plotlty](https://plotly.com/) - Visualização
* [Pickle](https://docs.python.org/3/library/pickle.html) - Serialização de Objetos (Salvar objetos para utiliza-los posteriormente)
* [Streamlit](https://streamlit.io/) - Interface Gráfica
* [Render](https://render.com/) - Deploy

# Telas do Projeto
* Homepage
![image](https://github.com/LucasAlves99/CreditDataRisk_Classification/assets/50807648/9bacc05b-730c-4e6c-830e-6b6fdc602fc3)
* EDA
![image](https://user-images.githubusercontent.com/50807648/230182936-02d59e74-f6e6-418e-994b-1e3a6570224f.png)
* Predict
![image](https://user-images.githubusercontent.com/50807648/230183185-f830b570-2603-4476-8781-2ad89d780dfa.png)

# Modelo Aplicado
O classificador XGB com 95% de acurácia e 95% no f1-score overfittou. Após alguns testes com dados fictícios de pessoas que possuem uma idade acima de pessoas inadimplentes(inadimplentes , segundo a base, em sua maioria, possuem de 20 a 25 anos),uma renda anual de 500.000 por ano, 5 anos de trabalho e ainda assim pedindo 2000 de empréstimo e tendo-o negado.  Acabei por Utilizar redes neurais sem parâmetros e obtive um resultado satisfatório, com 88% de acurácia e 88% de f1-score, acertando em relação a previsão de pessoas com o perfil parecido com o descrito acima.

# Conclusão
O foco desse projeto era implementar um projeto completo , end-to-end do inicio ao deploy, utilizando todo conhecimento já adquirido mas com um foco maior na parte de modelagem. 

# O que eu aprendi com esse Projeto
* Interface Streamlit
   * Multipages  
* Deploy
* O que fazer em casos de Overfitting (Diminuir o Bias)







