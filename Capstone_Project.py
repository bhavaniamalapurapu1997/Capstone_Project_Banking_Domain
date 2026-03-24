#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import os
rcParams['figure.figsize']=20,10


# In[31]:


sql= "select * from acc_ord_card_disp_client_dist aocdcd join loan_transactions lt on lt.account_id= aocdcd.account_id" 


# In[32]:


import mysql.connector as connection
try:
    mydb=connection.connect(host='Abhi',database='capstone_prj',user='root',passwd='Bhavani@1997',use_pure=True)
    df=pd.read_sql(sql,mydb)
    mydb.close()
except Exception as e:
    mydb.close()
    print(str(e))


# In[33]:


display(df)


# In[34]:


display(df.shape)


# In[35]:


df.columns


# In[36]:


print(df.isnull().sum())


# In[37]:


display (df['status'].unique())


# In[38]:


display(df['status'])


# In[39]:


display (df['status'].value_counts())


# In[40]:


df.status=pd.DataFrame(df.status.map({'A':0,'B':1,'C':2,'D':3}),columns=['status'])
display(df.status)


# In[41]:


display (df['status'].value_counts())


# In[44]:


x=df[['loan_amount','duration','payments']].values
display(x)


# In[46]:


y=df['status'].values
display(y)


# In[51]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=143)
display(x_train.shape)
display(y_train.shape)
display(x_test.shape)
display(y_test.shape)


# In[52]:


accuracy=[]
for i in range(1,15):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn=knn.fit(x_train,y_train)
    y_pred=knn.predict(x_train)
    acc=r2_score(y_train,y_pred)
    accuracy.append(acc)
display(accuracy)
plt.plot(range(1,15),accuracy)
plt.show()
    
    


# In[53]:


knn=KNeighborsClassifier(n_neighbors=3)
knn=knn.fit(x_train,y_train)
y_pred=knn.predict(x_train)
acc=r2_score(y_train,y_pred)
display(acc)


# In[54]:


knn=KNeighborsClassifier(n_neighbors=10)
knn=knn.fit(x_train,y_train)
y_pred=knn.predict(x_train)
acc=r2_score(y_train,y_pred)
display(acc)


# In[55]:


from sklearn.metrics import confusion_matrix,classification_report
cr=classification_report(y_train,y_pred)
display(cr)
cm=confusion_matrix(y_train,y_pred)
display(cm)


# In[73]:


from sklearn.tree import DecisionTreeClassifier
Tree=DecisionTreeClassifier(criterion='entropy')
Tree=Tree.fit(x_test,y_test)
t_pred=Tree.predict(x_test)
print(pd.DataFrame(t_pred).groupby(0).agg({0:np.size}))
acc=r2_score(y_test,t_pred)
cr=classification_report(y_test,t_pred)
cm=confusion_matrix(y_test,t_pred)
display(acc,' ',cr,' ',cm)


# In[72]:


from sklearn.tree import DecisionTreeClassifier
trr=DecisionTreeClassifier(criterion='gini')
trr.fit(x_test,y_test)
trr_pred=trr.predict(x_test)
print(pd.DataFrame(trr_pred).groupby(0).agg({0:np.size}))
acc=r2_score(y_test,trr_pred)
cr=classification_report(y_test,trr_pred)
cm=confusion_matrix(y_test,trr_pred)
display(acc,' ',cr,' ',cm)


# In[69]:


from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=50,criterion='entropy')
rc.fit(x_test,y_test)
rc_pred=trr.predict(x_test)
print(pd.DataFrame(rc_pred).groupby(0).agg({0:np.size}))
acc=r2_score(y_test,rc_pred)
cr=classification_report(y_test,rc_pred)
cm=confusion_matrix(y_test,rc_pred)
display(acc,' ',cr,' ',cm)


# In[74]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_test,y_test)
lr_pred=trr.predict(x_test)
print(pd.DataFrame(lr_pred).groupby(0).agg({0:np.size}))
acc=r2_score(y_test,lr_pred)
cr=classification_report(y_test,lr_pred)
cm=confusion_matrix(y_test,lr_pred)
display(acc,' ',cr,' ',cm)


# In[79]:


from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
sql="select * from loan_transactions lt join acc_ord_card_disp_client_dist as aocdcd on lt.account_id=aocdcd.account_id"
import mysql.connector as connection
try:
    mydb=connection.connect(host='Abhi',user='root',passwd='Bhavani@1997',database='capstone_prj',use_pure=True)
    df=pd.read_sql(sql,mydb)
    mydb.close()
except Exception as e:
    mydb.close()
    print(str(e))


# In[80]:


display(df.head())


# In[81]:


display(df.shape)


# In[82]:


display(df.columns)


# In[83]:


print(df.isnull().sum())


# In[87]:


x=df[['loan_amount','balance']].values
display(x)


# In[89]:


from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
minmax_x=minmax.fit_transform(x)
print(minmax_x)


# In[92]:


wcss=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(minmax_x)
    wcss.append(kmeans.inertia_)
print(wcss)


# In[94]:


plt.plot(range(1,20),wcss)
plt.title("Elbow method")
plt.show()


# In[104]:


kmeans=KMeans(n_clusters=6,init='k-means++')
kmeans.fit(minmax_x)
y_kmeans=kmeans.labels_
display(y_kmeans)


# In[105]:


plt.scatter(minmax_x[y_kmeans==0,0],minmax_x[y_kmeans==0,1],s=100,c='r',label='Cluster-1')
plt.scatter(minmax_x[y_kmeans==1,0],minmax_x[y_kmeans==1,1],s=100,c='b',label='Cluster-2')
plt.scatter(minmax_x[y_kmeans==2,0],minmax_x[y_kmeans==2,1],s=100,c='g',label='Cluster-3')
plt.scatter(minmax_x[y_kmeans==3,0],minmax_x[y_kmeans==3,1],s=100,c='k',label='Cluster-4')
plt.scatter(minmax_x[y_kmeans==4,0],minmax_x[y_kmeans==4,1],s=100,c='y',label='Cluster-5')
plt.scatter(minmax_x[y_kmeans==5,0],minmax_x[y_kmeans==5,1],s=100,c='c',label='Cluster-6')
plt.show()


# In[135]:


x_final=pd.concat([df.iloc[:,1],pd.DataFrame(x),pd.DataFrame(y_kmeans)],axis=1)
x_final.columns=['account_id','loan_amount','balance','cluster']
x_final.to_excel('C:\\Users\\adima\\OneDrive\\Documents\\Cohort 128_ML_ Day 45Projects\\MY SQL Project Python\\KMeans.xlsx',index=None)
display(x_final)


# In[111]:


x_hc=df[['loan_amount','balance']].values
display(x_hc)
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
minmax_x=minmax.fit_transform(x_hc)
display(minmax_x)


# In[118]:


import scipy.cluster.hierarchy as sch
Dendro=sch.dendrogram(sch.linkage(minmax_x,method='centroid'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("ED")
plt.show()


# In[119]:


dendrogram=sch.dendrogram(sch.linkage(minmax_x,method='single'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('ED')
plt.show()


# In[120]:


dendrogram=sch.dendrogram(sch.linkage(minmax_x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('ED')
plt.show()


# In[122]:


dendrogram=sch.dendrogram(sch.linkage(minmax_x,method='average'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('ED')
plt.show()


# In[128]:


from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=2,metric="euclidean",linkage='ward')
y_hc=hc.fit_predict(minmax_x)
display(y_hc)
pd.DataFrame(y_hc).value_counts()


# In[129]:


plt.scatter(minmax_x[y_hc==0,0],minmax_x[y_hc==0,1],s=100,c='r',label='cluster-1')
plt.scatter(minmax_x[y_hc==1,0],minmax_x[y_hc==1,1],s=100,c='b',label='cluster-2')


# 

# In[130]:


from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=2,metric="euclidean",linkage='single')
y_hc=hc.fit_predict(minmax_x)
display(y_hc)
pd.DataFrame(y_hc).value_counts()
plt.scatter(minmax_x[y_hc==0,0],minmax_x[y_hc==0,1],s=100,c='r',label='cluster-1')
plt.scatter(minmax_x[y_hc==1,0],minmax_x[y_hc==1,1],s=100,c='b',label='cluster-2')


# In[131]:


from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=2,metric="euclidean",linkage='average')
y_hc=hc.fit_predict(minmax_x)
display(y_hc)
pd.DataFrame(y_hc).value_counts()
plt.scatter(minmax_x[y_hc==0,0],minmax_x[y_hc==0,1],s=100,c='r',label='cluster-1')
plt.scatter(minmax_x[y_hc==1,0],minmax_x[y_hc==1,1],s=100,c='b',label='cluster-2')


# In[136]:


xhc_final=pd.concat([df.iloc[:,1],pd.DataFrame(x_hc),pd.DataFrame(y_hc)],axis=1)
xhc_final.columns=['account_id','loan_amount','balance','cluster']
xhc_final.to_excel("C:\\Users\\adima\\OneDrive\Documents\\Cohort 128_ML_ Day 45Projects\\MY SQL Project Python\\xhc_final.xlsx",index=None)


# In[134]:


display(df)


# In[ ]:




