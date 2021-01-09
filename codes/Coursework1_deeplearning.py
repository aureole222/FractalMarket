#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data=pd.read_csv('Dataset_A.csv',header=None)
data.columns=['Y','Sell_Price_1','Sell_Volume_1','Buy_Price_1','Buy_Volume_1','Sell_Price_2','Sell_Volume_2','Buy_Price_2','Buy_Volume_2','day1','day2','day3','day4','day5']
data.head()


# In[163]:


plt.figure(figsize=(5,5))
plt.imshow(data.corr(),cmap='YlGnBu')
plt.colorbar()
plt.show()


# ## Step 1: EDA 
# #### Look at the distribution and the correlation between factors, as well as check the distribution of data (graph produced in r)
# 
# 
# ![sell_price_1.png](attachment:sell_price_1.png)
# ![buy_volume_2.png](attachment:buy_volume_2.png)
# ![buy_price_2.png](attachment:buy_price_2.png)
# ![sell_volume_1.png](attachment:sell_volume_1.png)
# 
# ---
# 
# We could see volume data is heavily tailed, we should take log transformation 

# In[109]:


print(data.isnull().any())
cleaned_data=data.copy()
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.hist(cleaned_data.loc[:,'Sell_Volume_1'].values)
plt.subplot(2,2,2)
plt.hist(cleaned_data.loc[:,'Sell_Volume_2'].values)
plt.subplot(2,2,3)
plt.hist(cleaned_data.loc[:,'Buy_Volume_1'].values)
plt.subplot(2,2,4)
plt.hist(cleaned_data.loc[:,'Buy_Volume_2'].values)
plt.show()


# In[110]:


cleaned_data.loc[:,'Sell_Volume_1']=np.log(data['Sell_Volume_1'])
cleaned_data.loc[:,'Sell_Volume_2']=np.log(data['Sell_Volume_2'])
cleaned_data.loc[:,'Buy_Volume_1']=np.log(data['Buy_Volume_1'])
cleaned_data.loc[:,'Buy_Volume_2']=np.log(data['Buy_Volume_2'])
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.hist(cleaned_data.loc[:,'Sell_Volume_1'].values)
plt.subplot(2,2,2)
plt.hist(cleaned_data.loc[:,'Sell_Volume_2'].values)
plt.subplot(2,2,3)
plt.hist(cleaned_data.loc[:,'Buy_Volume_1'].values)
plt.subplot(2,2,4)
plt.hist(cleaned_data.loc[:,'Buy_Volume_2'].values)
plt.show()


# ## Step 2: Train/Test data split and normalisation 

# In[111]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.python.data import Dataset
import tensorflow.keras as keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


# In[112]:


x=cleaned_data.iloc[:,1:]
y=cleaned_data.iloc[:,[0]]
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)


# In[113]:


x_train[x_train.columns[0:8]].head()


# In[114]:


#Select numerical columns which needs to be normalized
train_norm = x_train[x_train.columns[0:8]]
test_norm = x_test[x_test.columns[0:8]]
# Normalize Training Data 
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
#Converting numpy array to dataframe
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
print (x_train.head())
# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_train.head())


# ## Step 3: feature selection methods

# In[28]:


from sklearn.ensemble import RandomForestRegressor
clf_rf = RandomForestRegressor(n_estimators=100)   
clr_rf = clf_rf.fit(x_train,np.array(y_train.values).reshape(-1,))
importances = clr_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]),x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[45]:


def generate_model(n_hidden,n_size,act,x_train):
    model = keras.Sequential(
        [keras.layers.Dense(n_size, activation=act,input_shape=(x_train.shape[1],)) for i in range(n_hidden)] + 
        [keras.layers.Dense(1, activation= 'sigmoid')])
    return model
result_train=[]
result_test=[]

model=generate_model(1,100,'relu',x_train.iloc[:,[1,2,3,4,5,6,7,8,9]])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model.fit(x_train.iloc[:,[1,2,3,4,5,6,7,8,9]], y_train,epochs= 5, batch_size = 60)
prediction=model.predict(x_test.iloc[:,[1,2,3,4,5,6,7,8,9]])
# evaluate the model
_, train_acc = model.evaluate(x_train.iloc[:,[1,2,3,4,5,6,7,8,9]], y_train, verbose=0)
_, test_acc = model.evaluate(x_test.iloc[:,[1,2,3,4,5,6,7,8,9]], y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))



model=generate_model(1,100,'relu',x_train)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model.fit(x_train, y_train,epochs= 5, batch_size = 60)
prediction=model.predict(x_test)
# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# #### accuracy was lower after removing features, I will keep all the features

# ## Step 4: Model construction and hyper-parameter tuning 
# 
# #### Look at number of hidden layer

# In[165]:


def generate_model(n_hidden,n_size,act):
    model = keras.Sequential(
        [keras.layers.Dense(n_size, activation=act,input_shape=(x_train.shape[1],)) for i in range(n_hidden)] + 
        [keras.layers.Dense(1, activation= 'sigmoid')])
    return model
result_train=[]
result_test=[]
for k in range(1,6):
    model=generate_model(k,100,'relu')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history2 = model.fit(x_train, y_train,epochs= 5, batch_size = 60)
    prediction=model.predict(x_test)
    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    result_train.append(train_acc)
    result_test.append(test_acc)
plt.plot([i for i in range(1,6)],result_train,label='training data')
plt.plot([i for i in range(1,6)],result_test,label='testing data')
plt.legend()
plt.title('Accuracy rate with different number of layers')
plt.show()


# #### look at number of nodes in each hidden layer

# In[167]:


result_train=[]
result_test=[]
for i in range(2):    
    for k in range(1,7):
        model=generate_model(2,25*k,'relu')
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history2 = model.fit(x_train, y_train,epochs= 5, batch_size = 60)
        prediction=model.predict(x_test)
        # evaluate the model
        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        result_train.append(train_acc)
        result_test.append(test_acc)


# In[168]:


result_train2=(np.array(result_train[:6])+np.array(result_train[6:]))/2
result_test2=(np.array(result_test[:6])+np.array(result_test[6:]))/2
plt.plot([25*i for i in range(1,7)],result_train2,label='training data')
plt.plot([25*i for i in range(1,7)],result_test2,label='testing data')
plt.title('Accuracy rate with different number of hidden neurons per layer')
plt.legend()
plt.show()


# In[173]:


result_train=[]
result_test=[]
for i in range(2):
    for k in range(1,5):
        model=generate_model(2,125,'relu')
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history2 = model.fit(x_train, y_train,epochs= 5*k, batch_size = 30)
        prediction=model.predict(x_test)
        # evaluate the model
        _, train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        result_train.append(train_acc)
        result_test.append(test_acc)


# In[174]:


result_train2=(np.array(result_train[:4])+np.array(result_train[4:]))/2
result_test2=(np.array(result_test[:4])+np.array(result_test[4:]))/2
plt.plot([5*i for i in range(1,5)],result_train2,label='training data')
plt.plot([5*i for i in range(1,5)],result_test2,label='testing data')
plt.title('Accuracy with different number of iterations')
plt.legend()
plt.show()
# optimal number of iteration is 10


# In[119]:


model=generate_model(2,125,'relu')
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model.fit(x_train, y_train,epochs= 10, batch_size = 30)
prediction=model.predict(x_test)
# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# In[104]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### find the optimal cutoff score

# In[72]:


acc=[]
i_list=[i*0.01 for i in range(101)]
for i in i_list:
    acc.append(accuracy_score(y_test,(prediction > i).astype(np.int)))
plt.plot(i_list,acc)


# ### Make final prediction

# In[125]:


new_data=pd.read_csv('Dataset_B_nolabels.csv',header=None)
new_data.columns=['Sell_Price_1','Sell_Volume_1','Buy_Price_1','Buy_Volume_1','Sell_Price_2','Sell_Volume_2','Buy_Price_2','Buy_Volume_2','day1','day2','day3','day4','day5']
new_data.head()
new_data.loc[:,'Sell_Volume_1']=np.log(new_data['Sell_Volume_1'])
new_data.loc[:,'Sell_Volume_2']=np.log(new_data['Sell_Volume_2'])
new_data.loc[:,'Buy_Volume_1']=np.log(new_data['Buy_Volume_1'])
new_data.loc[:,'Buy_Volume_2']=np.log(new_data['Buy_Volume_2'])

test_norm=new_data[new_data.columns[0:8]]
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
new_data.update(testing_norm_col)
prediction=(model.predict(new_data) > 0.5).astype(np.int).reshape(-1,)


# In[140]:


with open('saved_prediction.txt', 'w') as f:
    for item in prediction:
        f.write("%s\n" % item)





