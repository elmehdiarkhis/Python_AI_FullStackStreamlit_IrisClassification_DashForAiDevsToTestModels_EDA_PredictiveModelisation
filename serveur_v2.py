from flask import Flask, request
import pandas 
import numpy
import streamlit
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, Binarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import requests

#Pour empecher les warnings
streamlit.set_option('deprecation.showPyplotGlobalUse', False)

#Load Data 
dataset_Dict = load_iris()
data = dataset_Dict.data

#columns
feature_names = dataset_Dict.feature_names

#classes
target_names = dataset_Dict.target_names

#Output
target = dataset_Dict.target
target = target.reshape(-1,1)

#Create DataFrame
df = pandas.DataFrame(data, columns=feature_names)

app = Flask(__name__)

@app.route('/Train_Test_Split', methods=['POST'])
def predict():
   
    X = df.values
    Y = target
    
    seed = request.form.get('seed')
    test_size = request.form.get('test_size')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(test_size), random_state=int(seed))


    transform = request.form.get('transform')

    if transform=='StandardScaler':
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif transform=='Normalizer':
        sc = Normalizer()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif transform=='MinMaxScaler':
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    elif transform=='Binarizer':
        sc = Binarizer()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


    model = request.form.get('model')

    if model=='LogisticRegression':

        # Retrieve the data from the request body or parameters
        penalty_value = request.form.get('penalty_value')
        c_value = request.form.get('c_value')
        solver_value = request.form.get('solver_value')
       
        model = LogisticRegression(penalty=penalty_value, solver= solver_value, C=float(c_value))
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

    elif model=='DecisionTree':
        # Retrieve the data from the request body or parameters
        criterion_value = request.form.get('criterion_value')
        splitter_value = request.form.get('splitter_value')
        max_depth_value = request.form.get('max_depth_value')

        model = DecisionTreeClassifier(criterion=criterion_value, splitter=splitter_value, max_depth=int(max_depth_value))
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

    elif model=='SVC':
        # Retrieve the data from the request body or parameters
        kernel_value = request.form.get('kernel_value')
        c_value = request.form.get('c_value')
        gamma_value = request.form.get('gamma_value')

        model = SVC(kernel=kernel_value, C=float(c_value), gamma=gamma_value)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

    elif model=='KNeighbors':
        # Retrieve the data from the request body or parameters
        n_neighbors_value = request.form.get('n_neighbors_value')
        weights_value = request.form.get('weights_value')
        algorithm_value = request.form.get('algorithm_value')

        model = KNeighborsClassifier(n_neighbors=int(n_neighbors_value), weights=weights_value, algorithm=algorithm_value)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

    elif model=='RandomForest':
        # Retrieve the data from the request body or parameters
        n_estimators_value = request.form.get('n_estimators_value')
        criterion_value = request.form.get('criterion_value')
        max_depth_value = request.form.get('max_depth_value')

        model = RandomForestClassifier(n_estimators=int(n_estimators_value), criterion=criterion_value, max_depth=int(max_depth_value))
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
    
    elif model=='XGBoost':
        # Retrieve the data from the request body or parameters
        n_estimators_value = request.form.get('n_estimators_value')
        learning_rate_value = request.form.get('learning_rate_value')
        max_depth_value = request.form.get('max_depth_value')

        model = XGBClassifier(n_estimators=int(n_estimators_value), learning_rate=float(learning_rate_value), max_depth=int(max_depth_value))
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)


    return {
        'accuracy': float(accuracy_score(Y_test, Y_pred)),
        'precision': float(precision_score(Y_test, Y_pred, average='macro')),
        'recall': float(recall_score(Y_test, Y_pred, average='macro')),
        'f1': float(f1_score(Y_test, Y_pred, average='macro')),
    }


@app.route('/K_fold_Cross_Validation', methods=['POST'])
def cross_validation():
       
    X = df.values
    Y = target

    transform = request.form.get('transform')

    if transform=='StandardScaler':
        sc = StandardScaler()
        X = sc.fit_transform(X)
    elif transform=='Normalizer':
        sc = Normalizer()
        X = sc.fit_transform(X)
    elif transform=='MinMaxScaler':
        sc = MinMaxScaler()
        X = sc.fit_transform(X)
    elif transform=='Binarizer':
        sc = Binarizer()
        X = sc.fit_transform(X)


    model = request.form.get('model')

    if model=='LogisticRegression':
        # Retrieve the data from the request body or parameters

        penalty_value = request.form.get('penalty_value')
        solver_value = request.form.get('solver_value')
        c_value = request.form.get('c_value')

        n_splits = request.form.get('n_splits')


        model = LogisticRegression(penalty=penalty_value, solver= solver_value, C=float(c_value))
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')

    elif model=='DecisionTree':
        # Retrieve the data from the request body or parameters
        criterion_value = request.form.get('criterion_value')
        splitter_value = request.form.get('splitter_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')


        model = DecisionTreeClassifier(criterion=criterion_value, splitter=splitter_value, max_depth=int(max_depth_value))
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')

    elif model=='SVC':
        # Retrieve the data from the request body or parameters
        kernel_value = request.form.get('kernel_value')
        c_value = request.form.get('c_value')
        gamma_value = request.form.get('gamma_value')

        n_splits = request.form.get('n_splits')


        model = SVC(kernel=kernel_value, C=float(c_value), gamma=gamma_value)
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')

    elif model=='KNeighbors':
        # Retrieve the data from the request body or parameters
        n_neighbors_value = request.form.get('n_neighbors_value')
        weights_value = request.form.get('weights_value')
        algorithm_value = request.form.get('algorithm_value')

        n_splits = request.form.get('n_splits')


        model = KNeighborsClassifier(n_neighbors=int(n_neighbors_value), weights=weights_value, algorithm=algorithm_value)
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')



    elif model=='RandomForest':
        # Retrieve the data from the request body or parameters
        n_estimators_value = request.form.get('n_estimators_value')
        criterion_value = request.form.get('criterion_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')


        model = RandomForestClassifier(n_estimators=int(n_estimators_value), criterion=criterion_value, max_depth=int(max_depth_value))
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')
        

    elif model=='XGBoost':
        # Retrieve the data from the request body or parameters
        learning_rate_value = request.form.get('learning_rate_value')
        n_estimators_value = request.form.get('n_estimators_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')


        model = XGBClassifier(learning_rate=float(learning_rate_value), n_estimators=int(n_estimators_value), max_depth=int(max_depth_value))
        #accuracy in cross validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits))
        #precision in cross validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='precision_macro')
        #recall in cross validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='recall_macro')
        #f1 in cross validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = int(n_splits), scoring='f1_macro')


    return {
        'accuracy': accuracies.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }

@app.route('/Repeated_K_fold_Cross_Validation', methods=['POST'])
def repeated_k_fold_cross_validation():

    X = df.values
    Y = target

    transform = request.form.get('transform')

    if transform=='StandardScaler':
        sc = StandardScaler()
        X = sc.fit_transform(X)
    elif transform=='Normalizer':
        sc = Normalizer()
        X = sc.fit_transform(X)
    elif transform=='MinMaxScaler':
        sc = MinMaxScaler()
        X = sc.fit_transform(X)
    elif transform=='Binarizer':
        sc = Binarizer()
        X = sc.fit_transform(X)


    model = request.form.get('model')

    if model=='LogisticRegression':
        # Retrieve the data from the request body or parameters

        penalty_value = request.form.get('penalty_value')
        solver_value = request.form.get('solver_value')
        c_value = request.form.get('c_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = LogisticRegression(penalty=penalty_value, solver=solver_value, C=float(c_value))
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')
        


    elif model=='DecisionTree':
        # Retrieve the data from the request body or parameters
        criterion_value = request.form.get('criterion_value')
        splitter_value = request.form.get('splitter_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = DecisionTreeClassifier(criterion=criterion_value, splitter=splitter_value, max_depth=int(max_depth_value))
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')


    elif model=='SVC':
        # Retrieve the data from the request body or parameters
        kernel_value = request.form.get('kernel_value')
        c_value = request.form.get('c_value')
        gamma_value = request.form.get('gamma_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = SVC(kernel=kernel_value, C=float(c_value), gamma=gamma_value)
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')


    elif model=='KNeighbors':
        # Retrieve the data from the request body or parameters
        n_neighbors_value = request.form.get('n_neighbors_value')
        weights_value = request.form.get('weights_value')
        algorithm_value = request.form.get('algorithm_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = KNeighborsClassifier(n_neighbors=int(n_neighbors_value), weights=weights_value, algorithm=algorithm_value)
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')


    elif model=='RandomForest':
        # Retrieve the data from the request body or parameters
        n_estimators_value = request.form.get('n_estimators_value')
        criterion_value = request.form.get('criterion_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = RandomForestClassifier(n_estimators=int(n_estimators_value), criterion=criterion_value, max_depth=int(max_depth_value))
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')

    elif model=='XGBoost':
        # Retrieve the data from the request body or parameters
        learning_rate_value = request.form.get('learning_rate_value')
        n_estimators_value = request.form.get('n_estimators_value')
        max_depth_value = request.form.get('max_depth_value')

        n_splits = request.form.get('n_splits')
        n_repeats = request.form.get('n_repeats')


        model = XGBClassifier(learning_rate=float(learning_rate_value), n_estimators=int(n_estimators_value), max_depth=int(max_depth_value))
        #accuracy in Repeated_K_fold_Cross_Validation
        accuracies = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0))
        #precision in Repeated_K_fold_Cross_Validation
        precision = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='precision_macro')
        #recall in Repeated_K_fold_Cross_Validation
        recall = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='recall_macro')
        #f1 in Repeated_K_fold_Cross_Validation
        f1 = cross_val_score(estimator = model, X = X, y = Y, cv = RepeatedKFold(n_splits=int(n_splits), n_repeats=int(n_repeats), random_state=0), scoring='f1_macro')



    return {
        'accuracy': accuracies.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }
    





if __name__ == '__main__':
    app.run(port=1101, debug=True)