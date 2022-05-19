import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer,load_digits,load_iris,load_wine
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
st.title("ML project by KING")
# st.text("Welcome")

st.header("welcome")
st.write("""Explore various classifiers and datasets
        Which one is best?""")

dataset_name=st.sidebar.selectbox('Select Dataset',['Iris','Breast_cancer','Wine','Digit recognition'])
st.write(f"### {dataset_name} Dataset")

classifier_name=st.sidebar.selectbox('Select Classifier',['logistic','SVM','Random forest'])
st.write(f'### {classifier_name} Classifier')

def get_dataset(name):
    data=None
    if name=='Iris':
        data=load_iris()
    elif name=='Breast_cancer':
        data=load_breast_cancer()
    elif name=='Wine':
        data=load_wine()
    elif name=='Digit recognition':
        data=load_digits()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)
st.write('Shape of dataset:',X.shape)
st.write('Number of classes:',len(np.unique(y)))

def add_parameter(clf_name):
    params=dict()
    if clf_name=='SVM':
        C=st.sidebar.slider('C',0.01,100.0)
        params['C']=C
    elif clf_name=='logistic':
        C = st.sidebar.slider('C', 0.01, 100.0)
        params['C'] = C
    elif clf_name=='Random forest':
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params
params=add_parameter(classifier_name)
def get_classifier(clf_name,params):
    clf=None
    if(clf_name=='SVM'):
        clf=SVC(C=params['C'])
    elif clf_name=='logistic':
        clf=LogisticRegression(C=params['C'])
    elif clf_name=='Random forest':
        clf=RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    return clf

clf=get_classifier(classifier_name,params)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
score=clf.score(X_test,y_test)
acc=accuracy_score(y_test,y_pred)

# st.write(f'### prediction score= {score}')
st.write(f'### accuracy score= {acc}')

#graoh plotting
#as we know that a graph is best when it is plotted for 2 components only so we will use
#pca for getting those 2 principal components and then finding graphs from it

pca=PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel(f"Principal Component 1 ")
plt.ylabel(f"Principal Component 2 ")
plt.colorbar()
plt.plot()
st.pyplot(fig)
