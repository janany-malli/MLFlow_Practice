
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts

import dvc.api

path = 'data/Titanic+Data+Set.csv'
repo = 'C:/Users/Janany/mlopsdemo1'
version = 'v1'

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)

if __name__ == '__main__':
    print('Starting the model')
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name = 'titanic_model_1')

    mlflow.autolog()

    #Loading Dataset
    titanic_df = pd.read_csv(data_url)   

    #Performing EDA
    print(titanic_df.head())
    print(titanic_df.info())

    titanic_df.drop_duplicates(inplace =True)   #dropping duplicate rows

    # Checking for missing values in Age
    print(titanic_df[titanic_df['Age'].isna()])

    #Imputing null value in Age column with median
    titanic_df['Age'].replace(np.nan,titanic_df['Age'].median(),inplace=True)

    #dropping rows with null values in "Embarked" column as there are only two rows with null values
    titanic_df['Embarked'].fillna('S')

    # Dropping "Cabin" column as it has maximum missing values
    cols = ['PassengerId','Name','Ticket','Embarked','Cabin']
    titanic_df.drop(cols,axis=1,inplace=True)

    #One hot encoding 
    titanic_df = pd.get_dummies(titanic_df,drop_first=True)

    #log data params
    mlflow.log_param('data_url',data_url)
    mlflow.log_param('data_version',version)
    mlflow.log_param('input_rows',titanic_df.shape[0])
    mlflow.log_param('input_cols',titanic_df.shape[1])

    #splitting data into training and test set for independent attributes
    X = titanic_df.drop(['Survived'],axis='columns',inplace=False)
    y = titanic_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state =22)

    print(X_train.shape,X_test.shape)
    log_param("Train shape",X_train.shape)

    model_1 = DecisionTreeClassifier(criterion = "entropy", max_depth=20,min_samples_leaf=2)
    # #model_lr = LogisticRegression()
    # model_rf = RandomForestClassifier(max_features='auto',min_samples_leaf=2,min_samples_split=10,n_estimators=50)
    model_1.fit(X_train,y_train)
    print("Model trained")

    prediction =model_1.predict(X_test)

    # Performance on train and test data
    
    #Confusion Matrix
    con_matrix = confusion_matrix(y_test,prediction)
    print("Confusion Matrix: ",con_matrix)
    
    # Accuracy score
    train_accuracy = model_1.score(X_train,y_train)
    test_accuracy = model_1.score(X_test,y_test)
    acc_score = accuracy_score(y_test,prediction)
    print("Accuracy:",acc_score)

    #Precision score
    prec_score = precision_score(y_test,prediction)
    print("Precision:",prec_score)

    #Recall score
    rec_score = recall_score(y_test,prediction)
    print("RecallScore: ",rec_score)

    #F1 score
    f1_scr = f1_score(y_test,prediction)
    print("F1 score",f1_scr)

    #Logging Metrics
    log_metric("Accuracy",acc_score)
    log_metric("PrecisionScore",prec_score)
    log_metric("RecallScore",rec_score)
    log_metric("F1 score",f1_scr)

    mlflow.sklearn.log_model(model_1,"Model1_DT")
    # #mlflow.sklearn.log_model(model_lr,"Model_LR")
    # mlflow.sklearn.log_model(model_rf,"Model_RF")
    mlflow.log_artifact(data_url)