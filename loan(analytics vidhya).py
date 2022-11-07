#!/usr/bin/env python
# coding: utf-8

# # LOAN PREDICTION

# **Predict Loan Eligibility for Dream Housing Finance company**
# 
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers. 

# **Data Collection from Analytics Vidhya**
# * The link for the dataset is given below:
#     https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/#ProblemStatement

# In[1]:


#Importing some necessary libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[2]:


os.chdir(r"C:\Users\riama\Desktop\Machine_learning")


# In[3]:


loan=pd.read_csv("loan_train.csv")


# In[4]:


loan.shape


# In[5]:


loan.head()


# In[6]:


#setting the loan id as index and not dropping that bcz later on we need to access the data from here itself 
#that who is eligible for loan or not
loan.set_index('Loan_ID',inplace=True)


# In[7]:


loan.info()


# **As per the above info there are 7 categorical and 5 numerical**

# In[8]:


loan.isnull().sum()


# **As we can see that there are missing values hence we will fill that accordingly for categorical will use mode or will replace that either by unknown as per the requirements and for the variable will fill median because mean does is not good if there are outliers in the data hence will fill median accordingly**

# #### Univariate Analysis

# In[9]:


#creating the function for the numerical variable 
def univar_num(y):
    missing= y.isnull().sum()
    min1=round(y.min(),2)
    max1=round(y.max(),2)
    mean=round(y.mean(),2)
    var=round(y.var(),2)
    std=round(y.std(),2)
    range1=round(max1-min1,2)
    q1= round(y.quantile(.25),2)
    q2=round(y.quantile(.5),2)
    q3=round(y.quantile(.75),2)
    skew=round(y.skew(),2)
    kurt=round(y.kurt(),2)
    myval={"Missing Value":missing,"Minimum value": min1,"Maximum value": max1, "Mean value": mean,"Variance": var,"Standard Deviation": std,"Range": range1,
          "Quantile1":q1,"Quantile2": q2,"Quantile3":q3,"Skewness": skew,"kurtosis":kurt}
    sns.histplot(y)
    plt.show()
    sns.boxplot(data=loan,y=y)
    plt.show()
    return myval


# In[10]:


#creating the function for the categorical variables 
def univar(data,y):
    unique_count= data[y].nunique()
    missing= data[y].isnull().sum()
    unique_cat= list(data[y].unique())
    f1=pd.DataFrame(data[y].value_counts())
    f1.rename(columns={y:"Count"},inplace=True)
    f2=pd.DataFrame(data[y].value_counts(normalize=True))
    f2.rename(columns={y:"percentage"},inplace=True)
    f2["percentage"]=round(f2["percentage"]*100,2)  #(f2["percentage"]*100).round(2).astype(str)+"%" (if we want to add percentage
                                                    #then we have to convert it to string then will add %symbol)
    ff=pd.concat([f1,f2],axis=1)
    
    myvalue= {"missing":missing,"unique category": unique_cat,"unique_count":unique_count}
    print(f"value count and %\n",ff)
    sns.countplot(data=data,x=y)
    return myvalue


# In[11]:


loan.columns


# In[12]:


#finding those variable where column is object based 
loan.dtypes[loan.dtypes=="object"].index


# **I am treating the missing values along with the column check**

# In[13]:


#'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'Property_Area', 'Loan_Status'
univar(loan, "Gender")


# In[14]:


loan["Gender"].isnull().sum()


# **As per the above we can see that there are 13 missing value hence I am replacing that with unknown column **

# In[15]:


loan["Gender"]=loan["Gender"].replace(np.nan,"unknown")


# In[16]:


loan["Gender"].isnull().sum()


# In[17]:


#'Married', 'Dependents', 'Education',
#        'Self_Employed', 'Property_Area', 'Loan_Status'
univar(loan,'Married')


# In[18]:


loan[loan["Married"].isnull()]


# **As per the above I am replacing the missing value with unknown **

# In[19]:


loan["Married"]=loan["Married"].replace(np.nan,"unknown")


# In[20]:


univar(loan,"Married")


# In[21]:


#'Dependents', 'Education',
#        'Self_Employed', 'Property_Area', 'Loan_Status'
univar(loan,"Dependents")


# In[22]:


loan[loan["Dependents"].isnull()]


# In[23]:


loan["Dependents"]=loan["Dependents"].replace(np.nan,"unknown")


# **I have replaced the missing value with the unknown and cannot fill mode here bcz if we see the data where there are missing values we can see that there are people who are not yet married hence we do not about them and can't fill mode like this **

# In[24]:


univar(loan,"Dependents")


# In[25]:


# 'Education',
#        'Self_Employed', 'Property_Area', 'Loan_Status'
univar(loan,"Education")


# In[26]:


# 'Self_Employed', 'Property_Area', 'Loan_Status'
univar(loan,"Self_Employed")


# In[27]:


loan[loan["Self_Employed"].isnull()]


# In[28]:


loan["Self_Employed"]=loan["Self_Employed"].replace(np.nan,"unknown")


# In[29]:


univar(loan,"Self_Employed")


# In[30]:


#'Property_Area', 'Loan_Status'
univar(loan,"Property_Area")


# In[31]:


univar(loan,'Loan_Status')


# ###### I am replacing it in numerical variable 

# In[32]:


loan["Loan_Status"]=loan["Loan_Status"].map({"Y":1,"N":0})


# In[33]:


univar(loan,"Loan_Status")


# In[35]:


# Checking the variable where column are numerical 


# In[34]:


loan.dtypes[loan.dtypes!="object"].index


# In[36]:


# 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Loan_Status'
univar_num(loan["ApplicantIncome"])


# In[37]:


#'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Loan_Status'
univar_num(loan["CoapplicantIncome"])


# In[38]:


#'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Loan_Status'
univar_num(loan["LoanAmount"])


# **As from the above info I can check that there are missing variable in LoanAmount column hence filling median for that **

# In[39]:


loan[loan["LoanAmount"].isnull()]


# In[40]:


loan["LoanAmount"].median()


# In[41]:


loan["LoanAmount"].fillna(loan["LoanAmount"].median(),inplace=True)


# In[42]:


loan["LoanAmount"].isnull().sum()


# In[43]:


# 'Loan_Amount_Term', 'Credit_History', 'Loan_Status'
univar_num(loan["Loan_Amount_Term"])


# There are missing value hence filling median value accordingly 

# In[44]:


loan[loan["Loan_Amount_Term"].isnull()]


# In[45]:


loan["Loan_Amount_Term"].fillna(loan["Loan_Amount_Term"].median(),inplace=True)


# In[46]:


#'Credit_History', 'Loan_Status'
univar_num(loan["Credit_History"])


# In[47]:


univar(loan,"Credit_History")


# In[48]:


loan[loan["Credit_History"].isnull()]


# In[49]:


loan["Credit_History"].fillna(loan["Credit_History"].median(),inplace=True)


# In[50]:


loan.isnull().sum()


# **All missing value has been removed**

# ##### Bivariate Analysis

# In[51]:



#Cat-Cat
# 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'Property_Area'
print(pd.crosstab(loan["Gender"],loan["Loan_Status"],normalize="index"))
print(pd.crosstab(loan["Gender"],loan["Loan_Status"],normalize="index").plot(kind="bar"))


# In[52]:


pd.crosstab(loan["Married"],loan["Loan_Status"],normalize="index")


# As per the above we can say that who are married they are getting loan 

# In[53]:


pd.crosstab(loan["Dependents"],loan["Loan_Status"],normalize="index")


# As per the above we can say that who have 2 dependents they are getting approval for the loan

# In[54]:


pd.crosstab(loan["Education"],loan["Loan_Status"],normalize="columns")


# from this we can say that those are graduate getting approval for the loan 

# In[55]:


pd.crosstab(loan["Self_Employed"],loan["Loan_Status"],normalize="columns")


# In[56]:


pd.crosstab(loan["Property_Area"],loan["Loan_Status"],normalize="all")


# In[57]:


#Cat-Num
loan.dtypes[loan.dtypes!="object"].index


# In[58]:


# ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History'
print(loan.groupby(["Loan_Status"]).agg({"ApplicantIncome":["mean","min","max"]}))
print(loan.groupby(["Loan_Status"]).agg({"ApplicantIncome":["mean","min","max"]}).plot(kind="bar"))


# In[59]:


print(loan.groupby(["Loan_Status"]).agg({"CoapplicantIncome":["mean","min","max"]}))
print(loan.groupby(["Loan_Status"]).agg({"CoapplicantIncome":["mean","min","max"]}).plot(kind="bar"))


# In[60]:


print(loan.groupby(["Loan_Status"]).agg({"LoanAmount":["mean","min","max"]}))
print(loan.groupby(["Loan_Status"]).agg({"LoanAmount":["mean","min","max"]}).plot(kind="bar"))


# In[61]:


print(loan.groupby(["Loan_Status"]).agg({"Loan_Amount_Term":["mean","min","max"]}))
print(loan.groupby(["Loan_Status"]).agg({"Loan_Amount_Term":["mean","min","max"]}).plot(kind="bar"))


# In[62]:


print(loan.groupby(["Loan_Status"]).agg({"Credit_History":["mean","min","max"]}))
print(loan.groupby(["Loan_Status"]).agg({"Credit_History":["mean","min","max"]}).plot(kind="bar"))


# ##### Outliers

# In[63]:


loan.describe(percentiles=[.01,.02,.03,.04,.05,.25,.5,.75,.9,.95,.96,.97,.98,.99]).T


# as checked there are outliers hence treating them with the Turkey method

# In[64]:


var="ApplicantIncome"
q1=loan[var].quantile(.25)
q2=loan[var].quantile(.75)
IQR=q2-q1
lower=q1-1.5*(IQR)
upper=q2+1.5*(IQR)
loan[var]=np.where(loan[var]>=upper,upper,loan[var])
loan[var]=np.where(loan[var]<=lower,lower,loan[var])


# In[65]:


sns.boxplot(data=loan,y="ApplicantIncome")


# In[66]:


sns.boxplot(data=loan,y="CoapplicantIncome")


# In[67]:


var="CoapplicantIncome"
q1=loan[var].quantile(.25)
q2=loan[var].quantile(.75)
IQR=q2-q1
lower=q1-1.5*(IQR)
upper=q2+1.5*(IQR)
loan[var]=np.where(loan[var]>=upper,upper,loan[var])
loan[var]=np.where(loan[var]<=lower,lower,loan[var])


# In[68]:


sns.boxplot(data=loan,y="CoapplicantIncome")


# In[69]:


var="LoanAmount"
q1=loan[var].quantile(.25)
q2=loan[var].quantile(.75)
IQR=q2-q1
lower=q1-1.5*(IQR)
upper=q2+1.5*(IQR)
loan[var]=np.where(loan[var]>=upper,upper,loan[var])
loan[var]=np.where(loan[var]<=lower,lower,loan[var])


# In[70]:


sns.boxplot(data=loan,y="LoanAmount")


# In[71]:


sns.boxplot(data=loan,y="Loan_Amount_Term")


# In[72]:


var="Loan_Amount_Term"
q1=loan[var].quantile(.25)
q2=loan[var].quantile(.75)
IQR=q2-q1
lower=q1-1.5*(IQR)
upper=q2+1.5*(IQR)
loan[var]=np.where(loan[var]>=upper,upper,loan[var])
loan[var]=np.where(loan[var]<=lower,lower,loan[var])


# ##### Dummy Creation

# In[73]:


loan_dum=pd.get_dummies(data=loan,drop_first=True)


# In[74]:


loan_dum.drop(columns=["Loan_Amount_Term"],inplace=True)


# #### Multicollinearity

# In[75]:


cr=loan_dum.corr()
cr1=cr[abs(cr)>0.7]
plt.figure(figsize=[10,11])
sns.heatmap(cr1,annot=True,cmap="coolwarm")


# ##### Model Development

# In[76]:


y=loan_dum["Loan_Status"]
x=loan_dum.drop(columns=["Loan_Status"])


# In[77]:


# from sklearn import metrics
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# In[78]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


# In[79]:


#Standardizing 
mn=MinMaxScaler()
mn_fit=mn.fit(x_train)
x_train_mn=mn_fit.transform(x_train)
x_test_mn=mn_fit.transform(x_test)

x_train_mn=pd.DataFrame(x_train_mn,columns=x_train.columns)
x_test_mn=pd.DataFrame(x_test_mn,columns=x_test.columns)


# In[80]:


logR=LogisticRegression(max_iter=1000)
logR.fit(x_train_mn,y_train)


# In[81]:


print('Train Score : ', logR.score(x_train_mn,y_train))
print('Test Score : ', logR.score(x_test_mn,y_test))


# In[82]:


#predicting
pred_train=logR.predict(x_train_mn)
pred_test=logR.predict(x_test_mn)


# In[83]:


def classification(act,pred,probs):
    ac1=metrics.accuracy_score(act,pred)
    rc1=metrics.recall_score(act,pred)
    pc1=metrics.precision_score(act,pred)
    f1=metrics.f1_score(act,pred)
    roc1=metrics.roc_auc_score(act,pred)
    result={"Accuracy":ac1,"Recall":rc1,"Precission":pc1,"F1score": f1, "AUC": roc1}
    
    fpr,tpr,threshold=metrics.roc_curve(act,probs)
    plt.plot([0,1],[0,1],"k--")  #0,1 are the edges starting from 0 to 1 and k-- is for the line which is coming in graph and 
                                 #k is for black color
    plt.plot(fpr,tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    return result


# In[84]:


prob_train_1=logR.predict_proba(x_train_mn)[:,1]
prob_test_1=logR.predict_proba(x_test_mn)[:,1]


# In[85]:


classification(y_train,pred_train,prob_train_1)


# In[86]:


classification(y_test,pred_test,prob_test_1)


# **As per the above info we can see that recall is 98% for train and 97% for test 
# * Accuracy for the train is 80%
# * Accuracy for the test is 82%**

# ##### Decision Tree

# In[87]:


dt=DecisionTreeClassifier()
dt.fit(x_train_mn,y_train)


# In[88]:


print('Train Accuracy :',round(dt.score(x_train_mn,y_train),3))
print('Test Accuracy :',round(dt.score(x_test_mn,y_test),3))


# In[89]:


#finding the best parameters 
params={
    'criterion': ['gini','entropy'],
    'max_depth': [5,7,9,10,11],
    'min_samples_split': [10,15,20,50,100,200,250],
    'min_samples_leaf': [5,10,15,20,50,80,100]}

dtg=DecisionTreeClassifier()
gd_search=GridSearchCV(estimator=dtg,param_grid=params,cv=10,n_jobs=-1,verbose=2)
gd_search.fit(x_train_mn,y_train)


# In[90]:


gd_search.best_estimator_


# In[91]:


gd_search.best_params_


# In[92]:


dt1=DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_leaf=15,min_samples_split=200)
dt1.fit(x_train_mn,y_train)

print("Train Accuracy:" ,dt1.score(x_train_mn,y_train))
print("Test Accuracy:" ,dt1.score(x_test_mn,y_test))


# In[93]:


def classification(act,pred,probs):
    ac1=metrics.accuracy_score(act,pred)
    rc1=metrics.recall_score(act,pred)
    pc1=metrics.precision_score(act,pred)
    f1=metrics.f1_score(act,pred)
    roc1=metrics.roc_auc_score(act,pred)
    result={"Accuracy":ac1,"Recall":rc1,"Precission":pc1,"F1score": f1, "AUC": roc1}
    
    fpr,tpr,threshold=metrics.roc_curve(act,probs)
    plt.plot([0,1],[0,1],"k--")  #0,1 are the edges starting from 0 to 1 and k-- is for the line which is coming in graph and 
                                 #k is for black color
    plt.plot(fpr,tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    return result


# In[94]:


pred_train_dt=dt1.predict(x_train_mn)
pred_test_dt=dt1.predict(x_test_mn)


# In[95]:


prob_train_dt=dt1.predict_proba(x_train_mn)[:,1]
prob_test_dt=dt1.predict_proba(x_test_mn)[:,1]


# In[96]:


classification(y_train,pred_train_dt,prob_train_dt)


# In[97]:


classification(y_test,pred_test_dt,prob_test_dt)


# ** As per the above we can say that recall is 98% for train and 97% for test
# - Accuracy is 80% for train
# - Accuracy is 82% for test

# ##### Random Forest

# In[98]:


rf=RandomForestClassifier()
rf.fit(x_train_mn,y_train)


# In[99]:


print("Train Accuracy: ", rf.score(x_train_mn,y_train))
print("Test Accuracy: ", rf.score(x_test_mn,y_test))


# In[100]:


params={"n_estimators":[100,150,200],
       "criterion":["gini","entropy"],
       "max_depth":[9,11,13],
       "min_samples_split":[50,100],
       "min_samples_leaf":[5,10,15],
       "max_features":["sqrt","log2"],
       "bootstrap":[True]
       }
rf1=RandomForestClassifier()
gs_rf=GridSearchCV(estimator=rf1,param_grid=params,cv=10,n_jobs=-1,verbose=1)
gs_rf.fit(x_train_mn,y_train)


# In[101]:


gs_rf.best_estimator_


# In[102]:


gs_rf.best_params_


# In[106]:


rf2=RandomForestClassifier(max_depth=9,max_features="sqrt",min_samples_leaf=5,min_samples_split=50,n_estimators=200)


# In[107]:


rf2.fit(x_train_mn,y_train)


# In[108]:


print("Train Accuracy: ", rf2.score(x_train_mn,y_train))
print("Test Accuracy: ", rf2.score(x_test_mn,y_test))


# In[109]:


def classification(act,pred,probs):
    ac1=metrics.accuracy_score(act,pred)
    rc1=metrics.recall_score(act,pred)
    pc1=metrics.precision_score(act,pred)
    f1=metrics.f1_score(act,pred)
    roc1=metrics.roc_auc_score(act,pred)
    result={"Accuracy":ac1,"Recall":rc1,"Precission":pc1,"F1score": f1, "AUC": roc1}
    
    fpr,tpr,threshold=metrics.roc_curve(act,probs)
    plt.plot([0,1],[0,1],"k--")  #0,1 are the edges starting from 0 to 1 and k-- is for the line which is coming in graph and 
                                 #k is for black color
    plt.plot(fpr,tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    return result


# In[110]:


pred_train_rf=rf2.predict(x_train_mn)
pred_test_rf=rf2.predict(x_test_mn)


# In[111]:


prob_train_rf=rf2.predict_proba(x_train_mn)[:,1]
prob_test_rf=rf2.predict_proba(x_test_mn)[:,1]


# In[112]:


classification(y_train,pred_train_rf,prob_train_rf)#


# In[113]:


classification(y_test,pred_test_rf,prob_test_rf)


# ##### SVM(Support Vector Machines)

# In[116]:


from sklearn import svm
svm1=svm.LinearSVC()
svm1.fit(x_train_mn, y_train)


# In[117]:


print('Train Accuracy :', svm1.score(x_train_mn, y_train))
print('Test Accuracy :', svm1.score(x_test_mn, y_test))


# In[118]:


svm2=svm.SVC()     
svm2.fit(x_train_mn,y_train)


# In[119]:


print('Train Accuracy :', svm2.score(x_train_mn, y_train))
print('Test Accuracy :', svm2.score(x_test_mn, y_test))


# In[120]:


##Hyperparameter
params={'kernel':['linear','poly','rbf'],
         'degree': [2,3,4],
         'gamma': [0.1, 1, .001],
         'C': [0.001,0.01,0.1,1,10,100,200]
        }
sv=svm.SVC()
svm_rs=GridSearchCV(sv, params, cv=10, n_jobs=-1, verbose=True)
svm_rs.fit(x_train_mn,y_train)


# In[121]:


svm_rs.best_params_


# In[122]:


svm_rs.best_estimator_


# In[127]:


sv1=svm.SVC(C=0.01, degree=2, gamma=1, kernel='poly',probability=True)                    
sv1.fit(x_train_mn, y_train)

print('Train Accuracy : ', sv1.score(x_train_mn,y_train))
print('Test Accuracy : ', sv1.score(x_test_mn,y_test))


# In[128]:


pred_train_svm=sv1.predict(x_train_mn)
pred_test_svm=sv1.predict(x_test_mn)


# In[129]:


prob_train_svm=sv1.predict_proba(x_train_mn)[:,1]
prob_test_svm=sv1.predict_proba(x_test_mn)[:,1]


# In[130]:


def classification(act,pred,probs):
    ac1=metrics.accuracy_score(act,pred)
    rc1=metrics.recall_score(act,pred)
    pc1=metrics.precision_score(act,pred)
    f1=metrics.f1_score(act,pred)
    roc1=metrics.roc_auc_score(act,pred)
    result={"Accuracy":ac1,"Recall":rc1,"Precission":pc1,"F1score": f1, "AUC": roc1}
    
    fpr,tpr,threshold=metrics.roc_curve(act,probs)
    plt.plot([0,1],[0,1],"k--")  #0,1 are the edges starting from 0 to 1 and k-- is for the line which is coming in graph and 
                                 #k is for black color
    plt.plot(fpr,tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    return result


# In[131]:


classification(y_train,pred_train_svm,prob_train_svm)


# In[132]:


classification(y_test,pred_test_svm,prob_test_svm)


# * **The Accuracy of Logistic Regression is 83% for test and for train it's 80%**
#     * Precision Value for those whose loan is `not approved`: Train is 78% and Test is 88%
#     * Recall Value for those whose loan is `not approved`:Train is 98% and Test is 42%
# 
#    
#      * Precision Value for those whose loan is `approved`: Train is 78% and Test is 82%
#      * Recall Value for those whose loan is `approved`: Train is 98% and Test is 98%
#     

# In[140]:


print(f"Train classification report for logistic\n",metrics.classification_report(y_train,pred_train))
print(f"Test classification report for logistic\n",metrics.classification_report(y_test,pred_test))


# * **The Accuracy of Decision Tree is 83% for test and for train it's 80%**
# * Precision Value for those whose loan is `not approved`: Train is 93% and Test is 88%
# * Recall Value for those whose loan is `not approved`:Train is 43% and Test is 42%
#  
# 
# * Precision Value for those whose loan is `approved`: Train is 78% and Test is 82%
# * Recall Value for those whose loan is `approved`: Train is 98% and Test is 98%

# In[141]:


print(f"Train classification report for Decision Tree\n",metrics.classification_report(y_train,pred_train_dt))
print(f"Test classification report for Decision Tree\n",metrics.classification_report(y_test,pred_test_dt))


# * **The Accuracy of Random Forest is 83% for test and for train it's 80%**
# * Precision Value for those whose loan is `not approved`: Train is 93% and Test is 88%
# * Recall Value for those whose loan is `not approved`:Train is 43% and Test is 42%
#  
# 
# * Precision Value for those whose loan is `approved`: Train is 78% and Test is 82%
# * Recall Value for those whose loan is `approved`: Train is 98% and Test is 98%

# In[142]:


print(f"Train classification report for Random Forest\n",metrics.classification_report(y_train,pred_train_rf))
print(f"Test classification report for Random Forest\n",metrics.classification_report(y_test,pred_test_rf))


# * **The Accuracy of Support Vector Machine is 83% for test and for train it's 80%**
# * Precision Value for those whose loan is `not approved`: Train is 93% and Test is 88%
# * Recall Value for those whose loan is `not approved`:Train is 43% and Test is 42%
#  
# 
# * Precision Value for those whose loan is `approved`: Train is 78% and Test is 82%
# * Recall Value for those whose loan is `approved`: Train is 98% and Test is 98%

# In[144]:


print(f"Train classification report for Support Vector Machine\n",metrics.classification_report(y_train,pred_train_svm))
print(f"Test classification report for Support Vector Machine\n",metrics.classification_report(y_test,pred_test_svm))

