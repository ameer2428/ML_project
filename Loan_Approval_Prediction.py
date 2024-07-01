#!/usr/bin/env python
# coding: utf-8

# # Loan_Approval_Prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the data by Python pandas library

# In[2]:


df=pd.read_excel("D:\\Loan_Approval_Prediction.xlsx")
df.head()


# In[3]:


# no.of rows and columns
df.shape


# In[4]:


# column names
df.columns


# In[5]:


# summary of a DataFrame
df.info()


# # Data cleaning

# In[6]:


# checking for null_values
df.isnull().sum()


# In[7]:


# checking for outliers
plt.figure(figsize=(12,8))
sns.boxplot(df)


# In[8]:


# fill the null values of numerical datatype
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[9]:


# Fill the null values of object datatype
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[10]:


# Again checking for null_values
df.isnull().sum()


# In[11]:


# no.of people who took loan by gender
print('Number of people who took loan by gender')
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data = df, palette='Set1')


# In[12]:


# no.of people who took loan by married
print('Number of people who took loan by Married')
print(df['Married'].value_counts())
sns.countplot(x='Married',data = df, palette='Set1')


# In[13]:


# no.of people who took loan by education
print('Number of people who took loan by Education')
print(df['Education'].value_counts())
sns.countplot(x='Education',data = df, palette='Set1')


# In[14]:


# Total Applicant Income

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[15]:


# Apply Log Transformation
df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'] + 1)
sns.distplot(df['ApplicantIncomelog'])


# In[16]:


# Apply log transfermation
df['LoanAmountlog'] = np.log(df['LoanAmount'] + 1)
sns.distplot(df['LoanAmountlog'])


# In[17]:


df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term'] + 1)
sns.distplot(df['Loan_Amount_Term_log'])


# In[18]:


df['Total_Income_log'] = np.log(df['Total_Income'] + 1)
sns.distplot(df['Total_Income_log'])


# In[19]:


df.head()


# In[20]:


# drop unnecessary columns
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income',"Loan_ID"]
df = df.drop(columns = cols, axis = 1)
df.head()


# In[21]:


# Encoding Technique : Label Encoding, One Hot Encoding

from sklearn.preprocessing import LabelEncoder
#cols = ['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
le =  LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Education"] =  le.fit_transform(df["Education"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df.head()


# In[22]:


df["Dependents"].unique()


# In[23]:


df["Dependents"].replace({"3+":3},inplace=True)
df


# In[24]:


df.dtypes


# # Splitting Dependent and independent features

# In[25]:


# Splitting Independent and Dependent Features
x = df.drop(columns = ['Loan_Status'],axis = 1)
y = df['Loan_Status']


# In[26]:


x


# In[27]:


y


# # Train_Test_Split

# In[28]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)


# # Modelling and Evalution

# In[29]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[30]:


df.dtypes


# In[31]:


## Logistic Regression
model1 = LogisticRegression()
model1.fit(x_train,y_train)
y_pred_model1 = model1.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_model1)


# In[32]:


# Accuracy : the ratio of the correctly predicted values to total values
print("Accuracy score of Logistic Regression:", accuracy*100)


# In[33]:


score=cross_val_score(model1,x,y,cv=5)
score


# In[34]:


print("CV of Logistic Regression:",np.mean(score)*100)


# In[35]:


# Decision Tree Classifier

model2 = DecisionTreeClassifier()
model2.fit(x_train,y_train)
y_pred_model2 = model2.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_model2)
print("Accuracy score of Decision Tree: ", accuracy*100)


# In[36]:


score = cross_val_score(model2,x,y,cv=5)
print("Cross Validation score of Decision Tree: ",np.mean(score)*100)


# In[37]:


# Random Forest Classifier
model3 = RandomForestClassifier()
model3.fit(x_train,y_train)
y_pred_model3 = model3.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_model3)
print("Accuracy score of Random Forest: ", accuracy*100)


# In[38]:


#KNearestNeighbors model
model4 = KNeighborsClassifier(n_neighbors=3)
model4.fit(x_train,y_train)
y_pred_model4 = model4.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_model4)
print("Accuracy score of KNeighbors: ", accuracy*100)


# # classification report

# In[39]:


from sklearn.metrics import classification_report

def generate_classification_report(model_name,y_test,y_pred):
  report = classification_report(y_test,y_pred)
  print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_test,y_pred_model1)
generate_classification_report(model2,y_test,y_pred_model2)
generate_classification_report(model3,y_test,y_pred_model3)
generate_classification_report(model4,y_test,y_pred_model4)


# # Imbalance to balance

# In[40]:


# checking for balance or imbalance
df['Loan_Status'].value_counts()


# In[41]:


pip install imbalanced-learn


# In[42]:


from imblearn.over_sampling import RandomOverSampler


# In[43]:


oversample = RandomOverSampler(random_state=42)
x_resampled, y_resampled = oversample.fit_resample(x,y)
df_resampled = pd.concat([pd.DataFrame(x_resampled,columns=x.columns),pd.Series(y_resampled,name="Loan_status")],axis=1)


# In[45]:


x_resampled


# In[46]:


y_resampled


# In[47]:


y_resampled.value_counts()


# # splitting the resampled data and modeling,evalution and model selection

# In[48]:


# Train_Test_split on resampled Data
x_resampled_train, x_resampled_test, y_resampled_train, y_resampled_test = train_test_split(x_resampled,y_resampled,test_size = 0.25,random_state=42)


# In[49]:


## Logistic Regression
model1 = LogisticRegression()
model1.fit(x_resampled_train,y_resampled_train)
y_pred_model1 = model1.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model1)
accuracy*100


# In[50]:


## Decision Tree Classifier

model2 = DecisionTreeClassifier()
model2.fit(x_resampled_train,y_resampled_train)
y_pred_model2 = model2.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model2)
print("Accuracy score of Decision Tree: ", accuracy*100)


# In[51]:


## Random Forest Classifier
model3 = RandomForestClassifier()
model3.fit(x_resampled_train,y_resampled_train)
y_pred_model3 = model3.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model3)
print("Accuracy score of Random Forest: ", accuracy*100)


# In[52]:


#KNearestNeighbors model
model4 = KNeighborsClassifier(n_neighbors=3)
model4.fit(x_resampled_train,y_resampled_train)
y_pred_model4 = model4.predict(x_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model4)
print("Accuracy score of KNeighbors: ", accuracy*100)


# # classification report of resampled data

# In[53]:


from sklearn.metrics import classification_report

def generate_classification_report(model_name,y_test,y_pred):
  report = classification_report(y_test,y_pred)
  print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_resampled_test,y_pred_model1)
generate_classification_report(model2,y_resampled_test,y_pred_model2)
generate_classification_report(model3,y_resampled_test,y_pred_model3)
generate_classification_report(model4,y_resampled_test,y_pred_model4)


# In[ ]:




