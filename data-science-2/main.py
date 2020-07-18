#!/usr/bin/env python
# coding: utf-8

# # Desafio 4 - Funções de Probabilidade
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set(style='darkgrid')


# In[3]:


df = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
df.head()


# In[6]:


df.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[100]:


def q1():
    sample_height = get_sample(df, 'height', n=3000)
    shap = stats.shapiro(sample_height)
    return shap[1] >= 0.05


# In[101]:


q1()


# In[102]:


sample_height = get_sample(df, 'height', n=3000)
shap = stats.shapiro(sample_height)
print('Statistic:{}'.format(shap[0]))
print('P-valor:{}'.format(shap[1]))


# In[103]:


# Para n=100 o teste não aceita a hipotese nula


# In[104]:


plt.hist(sample_height, bins=25);


# In[105]:


import pylab
import statsmodels.api as sm

sm.qqplot(sample_height, line='45', fit=True)
pylab.show()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[117]:


def q2():
    sample_height = get_sample(df, 'height', n=3000)
    jb = stats.jarque_bera(sample_height)
    return jb[1] >= 0.05


# In[118]:


q2()


# In[108]:


sample_height = get_sample(df, 'height', n=3000)
jb = stats.jarque_bera(sample_height)
print('Statistic:{}'.format(jb[0]))
print('P-valor:{}'.format(jb[1]))


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[123]:


def q3():
    sample_weight = get_sample(df, 'weight', n=3000)
    ap = stats.normaltest(sample_weight)
    return ap[1] >= 0.05


# In[124]:


q3()


# In[125]:


sample_weight = get_sample(df, 'weight', n=3000)
ap = stats.normaltest(sample_weight)
print('Statistic:{}'.format(ap[0]))
print('P-valor:{}'.format(ap[1]))


# In[112]:


plt.hist(sample_weight, bins=25);


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[113]:


def q4():
    sample_weight = get_sample(df, 'weight', n=3000)
    log_sample_weight = np.log(sample_weight)
    ap2 = stats.normaltest(log_sample_weight)
    return ap2[1] >= 0.05


# In[114]:


q4()


# In[115]:


sample_weight = get_sample(df, 'weight', n=3000)
log_sample_weight = np.log(sample_weight)
ap2 = stats.normaltest(log_sample_weight)
print('Statistic:{}'.format(ap2[0]))
print('P-valor:{}'.format(ap2[1]))


# In[116]:


plt.hist(log_sample_weight, bins=25);


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[150]:


def q5():
    df5 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]

    bra = df5.height[df5['nationality']=='BRA']
    usa = df5.height[df5['nationality']=='USA']

    t_stati, t_pvalue = stats.ttest_ind(bra, usa, equal_var=False, nan_policy="omit")
    return t_pvalue >= 0.05


# In[151]:


q5()


# In[152]:


df5 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]
bra = df5.height[df5['nationality']=='BRA']
usa = df5.height[df5['nationality']=='USA']

t_stati, t_pvalue = stats.ttest_ind(bra, usa, equal_var=False, nan_policy="omit")
print('Statistic:{}'.format(t_stati))
print('P-valor:{}'.format(t_pvalue))


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[155]:


def q6():
    df6 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]

    bra = df6.height[df6['nationality']=='BRA']
    can = df6.height[df6['nationality']=='CAN']

    t_stati, t_pvalue = stats.ttest_ind(bra, can, equal_var=False, nan_policy="omit")
    return t_pvalue >= 0.05


# In[156]:


q6()


# In[157]:


df6 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]
bra = df6.height[df6['nationality']=='BRA']
can = df6.height[df6['nationality']=='CAN']

t_stati, t_pvalue = stats.ttest_ind(bra, can, equal_var=False, nan_policy="omit")
print('Statistic:{}'.format(t_stati))
print('P-valor:{}'.format(t_pvalue))


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[165]:


def q7():
    df7 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]

    usa = df7.height[df7['nationality']=='USA']
    can = df7.height[df7['nationality']=='CAN']

    t_stati, t_pvalue = stats.ttest_ind(usa, can, equal_var=False, nan_policy="omit")
    return round(t_pvalue, 8)


# In[166]:


q7()


# In[164]:


df7 = df[df['nationality'].isin(['BRA', 'USA', 'CAN'])]
usa = df7.height[df7['nationality']=='USA']
can = df7.height[df7['nationality']=='CAN']

t_stati, t_pvalue = stats.ttest_ind(usa, can, equal_var=False, nan_policy="omit")
print('Statistic:{}'.format(t_stati))
print('P-valor:{}'.format(t_pvalue))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
# 
# - Atletas brasileiros são diferentes dos americanos e canadenses (altura), mas EUA e Canada tem atletas com alturas semelhantes entre si.
