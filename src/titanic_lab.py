#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.listdir("data")


# In[4]:


get_ipython().system('pip install openpyxl')


# In[5]:


import pandas as pd
df = pd.read_excel("data/titanic3.xlsx")
df.head()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_excel("data/titanic3.xlsx")
df.head()


# In[8]:


datos = df[['pclass','survived','sex','age','sibsp','parch','embarked']]
datos.head()


# In[9]:


datos['age'] = datos['age'].fillna(datos['age'].median())


# In[10]:


bins = [-np.inf,15,25,35,45,55,65,75,np.inf]
labels = ["0-15","16-25","26-35","36-45","46-55","56-65","66-75","76-85"]

datos['Fac_Edad'] = pd.cut(datos['age'], bins=bins, labels=labels)


# In[11]:


datos['pclass'] = datos['pclass'].astype('category')
datos['survived'] = datos['survived'].astype('category')
datos['sex'] = datos['sex'].astype('category')
datos['embarked'] = datos['embarked'].astype('category')
datos['Fac_Edad'] = datos['Fac_Edad'].astype('category')


# In[12]:


print("Media:", datos['age'].mean())
print("Mediana:", datos['age'].median())
print("Varianza:", datos['age'].var())
print("Desviación:", datos['age'].std())


# In[13]:


plt.hist(datos['age'], bins=20)
plt.title("Distribución edades")
plt.show()


# In[14]:


datos['survived'].value_counts().plot(kind='bar')
plt.title("Sobrevivientes")
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

X = datos[['pclass','age']]
X = pd.get_dummies(X)

y = datos['survived'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3, random_state=1024)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train,y_train)

pred = modelo.predict(X_test)


# In[16]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[ ]:




