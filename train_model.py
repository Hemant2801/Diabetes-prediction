#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# # Data collection and analysis

# In[2]:


#lloading dataset
df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Diabetes detection/diabetes.csv')


# In[3]:


df.head()


# In[4]:


#shape of the dataset
df.shape


# In[5]:


#getting the statistical measure of the dataset
df.describe()


# In[6]:


df['Outcome'].value_counts() #give the total count of different values in the column
'''
0 ----> non-diabetic
1 ----> diabetic
'''


# In[7]:


#let us check the mean of the data across the outcome column
df.groupby('Outcome').mean()


# In[8]:


#sepearting data and labels
X = df.drop(columns = 'Outcome', axis = 1)
Y = df['Outcome']


# Standardize the data

# In[9]:


scaler = StandardScaler()
standardize_data = scaler.fit(X)
standardize_data = scaler.transform(X)


# In[10]:


standardize_data


# In[11]:


X = standardize_data
y = df['Outcome']


# split the data in training and testing data

# In[12]:


x_train , x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = y, random_state = 2)


# # TRAINING THE MODEL

# In[13]:


classifier = svm.SVC(kernel = 'linear')


# In[14]:


#training the SUPPORT VECTOR MACHINE CLASSIFIER
classifier.fit(x_train, y_train)


# evaluating the model

# In[15]:


#accuracy score on the training data
x_train_predict = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)


# In[16]:


print('ACCURAY SCORE OF THE TRAINING DATA : ', training_data_accuracy)


# In[17]:


#accuracy score on the testing data
x_test_predict = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)


# In[18]:


print('ACCURAY SCORE OF THE TESTING DATA : ', test_data_accuracy)


# # MAKING A PREDICTING SYSTEM

# In[24]:


input_data = input()
input_array = [float(i) for i in input_data.split(',')]

#convert the input into numpy array
input_array = np.asarray(input_array)

#reshape the array as we are predicting only on one instance
reshaped_array = input_array.reshape(1, -1)

#standardize the input data
std_data = scaler.transform(reshaped_array)
print(std_data)

#prediction
prediction = classifier.predict(std_data)
print(prediction)
if prediction == 0:
    print('NON-DIABETIC')
else:
    print('DIABETIC')


# In[ ]:




