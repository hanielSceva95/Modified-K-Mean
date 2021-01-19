import csv
import numpy as np
import random
from numpy.random import seed
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#I've converted the .data file to .csv which is read and replaced ? to NaN in the files
COLUMNS_COUNT = 2

with open('water-treatment.data', 'r') as f:
    columns = [next(f).strip() for line in range(COLUMNS_COUNT)]
temp_df = pd.read_csv('water-treatment.data', skiprows=COLUMNS_COUNT, header=None, delimiter=';', skip_blank_lines=True)
even_df = temp_df.iloc[::2].reset_index(drop=True)
odd_df = temp_df.iloc[1::2].reset_index(drop=True)
df = pd.concat([even_df, odd_df], axis=1)
df.columns = columns
df.to_csv('out.csv', index=False)
text = open("out.csv", "r")
text = ''.join([i for i in text]) \
    .replace("?", "NaN")
x = open("out.csv","w")
x.writelines(text)
x.close()

reader=pd.read_csv('water-treatment.csv',header=None,delimiter=',');
df=pd.DataFrame(reader)
# print('Before Cleaning Up the DataSet\n')
# print(df)

#Calculating the mean of each Column and Replacing "NaN" with the Corresponding mean values
for i in range(1,39):
    mean = df.loc[:,i].mean()
    # print('The mean of column :'+str(i))
    # print(mean)
    df.loc[:,i].fillna(mean, inplace=True)

for i in range(1,39):
    for j in range(0,527):
        mean=df.loc[:,i].mean();
        stdevi=df.loc[:,i].std();
        df.loc[j,i]=(df.loc[j,i]-mean)/stdevi;

# print('\n')
# print('After Cleaning Up the DataSet and performing Normalization\n')
# print(df)

#Dropping the Date Column
# print('\n')
# print('After Dropping\n')
df.drop(df.columns[0], axis=1, inplace=True)
# print(df)

# Implementing K-Means with K as 4
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(df)


# Adjusting the Clustering output from 0-3 to 1-4
for i in range(len(pred_y)):
    if pred_y[i]==0:
        pred_y[i]=1
    elif pred_y[i]==1:
        pred_y[i]=2
    elif pred_y[i]==2:
        pred_y[i]=3
    else:
        pred_y[i]=4

# Adjusting the Output to the desired form so that the Clusters get renamed and appear in order
l1=[]
l2=[]
cnt=0
for k in pred_y:
    if not k in l1:
        l1.append(k)
        cnt=cnt+1
        l2.append(cnt)
for k in range(len(pred_y)):
    for k1 in range(len(l1)):
        if (pred_y[k]==l1[k1]):
            pred_y[k]=l2[k1]
            break

# print('Clustering Output After Ordering In Specified Order')
# print(pred_y)


# Implementing Principal Component Analysis with 2 components On the Normalized DataFrame
df1 = StandardScaler().fit_transform(df)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df1)
principalDf = pd.DataFrame(data = principalComponents
                           , columns = ['principal component 1', 'principal component 2'])
# print('PRINCIPAL COMPONENT ANALYSIS\n')
# print('Principal Components\n')
# print(principalDf)

# K-Means Implementation on the PCA applied Dataset
kmeans1 = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y1 = kmeans1.fit_predict(principalDf)


# Adjusting the Clustering output from 0-3 to 1-4
for i in range(len(pred_y1)):
    if pred_y1[i]==0:
        pred_y1[i]=1
    elif pred_y1[i]==1:
        pred_y1[i]=2
    elif pred_y1[i]==2:
        pred_y1[i]=3
    else:
        pred_y1[i]=4


# Adjusting the Output to the desired form so that the Clusters get renamed and appear in order
l1=[]
l2=[]
cnt=0
for k in pred_y1:
    if not k in l1:
        l1.append(k)
        cnt=cnt+1
        l2.append(cnt)
for k in range(len(pred_y1)):
    for k1 in range(len(l1)):
        if (pred_y1[k]==l1[k1]):
            pred_y1[k]=l2[k1]
            break

# print('Clustering Output of K-Means With PCA')
# print(pred_y1)




x_train,x_test,y_train,y_test=train_test_split(df,pred_y,test_size=0.6,random_state = seed(2017))

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)

#
y_pred_1=logreg.predict(x_test)
print(y_test.shape,y_pred_1.shape)
acc1=accuracy_score(y_test, y_pred_1)
print('Accuracy of K-Means')
print(acc1)


x1_train,x1_test,y1_train,y1_test=train_test_split(principalDf,pred_y1,test_size=0.6,random_state = seed(201))

# instantiate the model (using the default parameters)
logreg1 = LogisticRegression()

# fit the model with data
logreg1.fit(x1_train,y1_train)

#
y_pred_2=logreg1.predict(x1_test)
print(y1_test.shape,y_pred_2.shape)
acc2=accuracy_score(y1_test, y_pred_2)
print('Accuracy of K-Means with PCA')
print(acc2)
