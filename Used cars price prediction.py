#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the data

# In[2]:


df=pd.read_excel("D:\\cardetails.xlsx")
df.head()


# # understanding of data

# In[3]:


df.info()


# In[4]:


# no.of rows and columns of the dataset
df.shape


# In[5]:


# Column names
df.columns


# In[6]:


# Droping unnecessary column
df.drop(columns=["torque"],inplace=True)
df.head()


# # Data preprocessing

# In[7]:


# Any null values
df.isnull().sum()


# In[8]:


# droping the null values bcz our data is so big 
df.dropna(inplace=True)
df.shape


# In[9]:


# Duplicate checking
df.duplicated().sum()


# In[10]:


# Droping duplicate records
df.drop_duplicates(inplace=True)
df.shape


# In[11]:


# again understanding the dataframe
df.info()


# # Data Analysis

# In[12]:


# unique values of each column
for col in df.columns:
    print("unique values of" +col)
    print(df[col].unique())
    print("=============")


# In[13]:


# reducing cars full name to its company name ( by using def function)
def get_brand_name(car_name):
    car_name=car_name.split(" ")[0]
    return car_name.strip()


# In[14]:


get_brand_name("Maruti Swift Dzire VDI")


# In[15]:


df["name"]=df["name"].apply(get_brand_name)
df["name"]


# In[16]:


def clean_data(value):
    value=value.split(" ")[0]
    value=value.strip()
    if value=='':
        value=0
    return float(value)


# In[17]:


df["mileage"]=df["mileage"].apply(clean_data)
df["max_power"]=df["max_power"].apply(clean_data)
df["engine"]=df["engine"].apply(clean_data)


# In[18]:


# checking if the data is chaged or not
for col in df.columns:
    print("unique values of" +col)
    print(df[col].unique())
    print("=============")


# In[19]:


df.head()


# In[20]:


# converting categorical data to numerical
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["name"]=encoder.fit_transform(df["name"])
df["fuel"]=encoder.fit_transform(df["fuel"])
df["seller_type"]=encoder.fit_transform(df["seller_type"])
df["transmission"]=encoder.fit_transform(df["transmission"])
df["owner"]=encoder.fit_transform(df["owner"])


# In[21]:


df


# In[22]:


# resetting the index
df.reset_index(inplace=True)
df


# In[23]:


df.info()


# # EDA

# In[24]:


plt.figure(figsize=(12,8))
sns.boxplot(df)
plt.show()


# In[25]:


plt.figure(figsize=(30,15))
sns.barplot(x=df["km_driven"],y=df["selling_price"])
plt.show()


# In[26]:


plt.figure(figsize=(12,8))
sns.barplot(x=df["year"],y=df["selling_price"])
plt.show()


# In[27]:


df.corr()


# In[28]:


df.describe()


# In[29]:


# seperating independent and dependent features
x=df.drop(columns=["selling_price"],axis=1)
y=df["selling_price"]


# In[30]:


x.shape


# In[31]:


y.shape


# # splitting the data into train data and test data

# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)


# # modelling

# In[33]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# # evalution

# In[34]:


ypred_train=model.predict(x_train)
from sklearn.metrics import r2_score
print("r2_score:",r2_score(y_train,ypred_train))


# In[35]:


ypred_train


# In[36]:


x_train.shape


# In[37]:


x_train.head(1)


# In[38]:


ip_model=pd.DataFrame([[3791,4,2013,5315,1,1,1,0,25.44,936.0,57.6,5.0]],columns=["index","name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
ip_model


# In[39]:


model.predict(ip_model)


# In[40]:


ypred_test=model.predict(x_test)


# In[41]:


ypred_test


# In[42]:


x_test.head(1)


# In[43]:


op_model=pd.DataFrame([[2940,20,2019,15000,1,1,1,0,25,1248,88,5]],columns=["index","name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
op_model


# # prediction on test data

# In[44]:


model.predict(op_model)

