import pandas 
import numpy
import matplotlib.pyplot as plt
import seaborn 
import streamlit
from sklearn.datasets import load_iris
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

# Streamlit---------------------------------------------------------------

#Set up the page
streamlit.set_page_config(page_title='Iris Classification', page_icon=':seedling:', layout='centered', initial_sidebar_state='auto')

streamlit.title('EDA and Predictive Modelisation')
streamlit.subheader('Exploratory Data Analysis')


#Define Sidebar 
options = ['EDA','Predictive Modelisation']
selected_option = streamlit.sidebar.selectbox('Select an option',options)



if selected_option=='EDA':
    
    streamlit.subheader('EDA : Exploratory Data Analysis and Data Visualization')
    streamlit.write('choose a plot frpm the options below')
    
    #Add option to show/hide data
    if streamlit.checkbox('Show Data'):
        streamlit.write(df)

    #Add options to show/hide missing values
    if streamlit.checkbox('Show Missing Values'):
        streamlit.write(df.isnull().sum())  

    #Add options to show/hide datatypes
    if streamlit.checkbox('Show Datatypes'):
        streamlit.write(df.dtypes)
    
    #Add options to show/hide  descriptiv stats
    if streamlit.checkbox('Show Descriptive Stats'):
        streamlit.write(df.describe())
    
    #Add options to show/hide  correlation
    if streamlit.checkbox('Show Correlation'):
        corr = df.corr('pearson')
        mask = numpy.triu(numpy.ones_like(corr, dtype=bool))
        seaborn.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
        streamlit.pyplot()
    
    #Add options to show/hide  histogram
    if streamlit.checkbox('Show Histogram'):
        for col in df.columns:
            fig = px.histogram(df, x=col, nbins=50, title=col)
            streamlit.plotly_chart(fig)

    #Add options to show/hide  Density Plot
    if streamlit.checkbox('Show Density Plot'):
        for col in df.columns:
            fig = px.density_contour(df, x=col, title=col)
            streamlit.plotly_chart(fig)
    
    #Add options to show/hide  scatter plot fot pair of features
    if streamlit.checkbox('Show Scatter Plot'):
        fig = px.scatter(df , x = feature_names[0], y = feature_names[1], color = target_names[target])
        streamlit.plotly_chart(fig)


elif selected_option=='Predictive Modelisation':
    
    streamlit.subheader('Predictive Modelisation')

    streamlit.write('choose a transforme type and model from the options below')


    transform_options = ['StandardScaler','Normalizer','MinMaxScaler','Binarizer']
    transform = streamlit.selectbox('Select a transform type',transform_options)

    parcourir_data_options = ['Train_Test_Split','K_fold_Cross_Validation','Repeated_K_fold_Cross_Validation']
    parcourir_data = streamlit.selectbox('Select a parcourir data type',parcourir_data_options)


    #Train_Test_Split options
    if parcourir_data=='Train_Test_Split':
        seed = streamlit.slider('Select a seed',min_value=1,max_value=100,value=1,step=1)
        test_size = streamlit.slider('Select a test size',min_value=0.1,max_value=0.9,value=0.2,step=0.1)
    #K_fold_Cross_Validation options
    elif parcourir_data=='K_fold_Cross_Validation':
        n_splits = streamlit.slider('Select a n_splits',min_value=2,max_value=10,value=5,step=1)

    #Repeated_K_fold_Cross_Validation options
    elif parcourir_data=='Repeated_K_fold_Cross_Validation':
        n_splits = streamlit.slider('Select a n_splits',min_value=2,max_value=10,value=5,step=1)
        n_repeats = streamlit.slider('Select a n_repeats',min_value=2,max_value=10,value=5,step=1)




    link = "http://localhost:1101/" + parcourir_data

    streamlit.write('Choisissez le modèle à utiliser :')
    model_options = ['LogisticRegression','DecisionTree', 'SVC', 'KNeighbors', 'RandomForest', 'XGBoost']
    model = streamlit.selectbox('Choisissez le modèle', model_options)

    transform_options = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'Binarizer']
    transform = streamlit.selectbox('Choisissez le type de transformation', transform_options)


    if model=='LogisticRegression':
        streamlit.write('Voici le résultat pour l\'algorithme Logistic Regression :')
        
        penalty_options = ['l2', 'none']
        penalty_value = streamlit.selectbox('Choisissez la pénalité', penalty_options)

        c_value = streamlit.slider('Choisissez la valeur de pénalité C', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        solver_options = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        solver_value = streamlit.selectbox('Choisissez le solver', solver_options)

        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'penalty_value': penalty_value,
                'c_value': c_value,
                'solver_value': solver_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'penalty_value': penalty_value,
                'c_value': c_value,
                'solver_value': solver_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'penalty_value': penalty_value,
                'c_value': c_value,
                'solver_value': solver_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }
        
        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])

    elif model=='DecisionTree':
        streamlit.write('Voici le résultat pour l\'algorithme Decision Tree Classifier :')
        
        criterion_options = ['gini', 'entropy']
        criterion_value = streamlit.selectbox('Choisissez le critère', criterion_options)

        splitter_options = ['best', 'random']
        splitter_value = streamlit.selectbox('Choisissez le splitter', splitter_options)

        max_depth_value = streamlit.slider('Choisissez la profondeur maximale de l\'arbre', min_value=1, max_value=50, value=10, step=1)

        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'criterion_value': criterion_value,
                'splitter_value': splitter_value,
                'max_depth_value': max_depth_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'criterion_value': criterion_value,
                'splitter_value': splitter_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'criterion_value': criterion_value,
                'splitter_value': splitter_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }


        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])

    elif model=='SVC':
        streamlit.write('Voici le résultat pour l\'algorithme SVC :')
        
        kernel_options = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        kernel_value = streamlit.selectbox('Choisissez le kernel', kernel_options)

        c_value = streamlit.slider('Choisissez la valeur de pénalité C', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        gamma_value = streamlit.selectbox('Choisissez la valeur de gamma', ['scale', 'auto'])
        
        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'kernel_value': kernel_value,
                'c_value': c_value,
                'gamma_value': gamma_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'kernel_value': kernel_value,
                'c_value': c_value,
                'gamma_value': gamma_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'kernel_value': kernel_value,
                'c_value': c_value,
                'gamma_value': gamma_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }

        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])

    elif model=='KNeighbors':
        streamlit.write('Voici le résultat pour l\'algorithme KNeighbors :')
        
        n_neighbors_value = streamlit.slider('Choisissez le nombre de voisins', min_value=1, max_value=50, value=10, step=1)

        weights_options = ['uniform', 'distance']
        weights_value = streamlit.selectbox('Choisissez le poids', weights_options)

        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        algorithm_value = streamlit.selectbox('Choisissez l\'algorithme', algorithm_options)
        
        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'n_neighbors_value': n_neighbors_value,
                'weights_value': weights_value,
                'algorithm_value': algorithm_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'n_neighbors_value': n_neighbors_value,
                'weights_value': weights_value,
                'algorithm_value': algorithm_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'n_neighbors_value': n_neighbors_value,
                'weights_value': weights_value,
                'algorithm_value': algorithm_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }



        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])


    elif model=='RandomForest':
        streamlit.write('Voici le résultat pour l\'algorithme RandomForest :')
        
        n_estimators_value = streamlit.slider('Choisissez le nombre d\'estimators', min_value=1, max_value=50, value=10, step=1)

        criterion_options = ['gini', 'entropy']
        criterion_value = streamlit.selectbox('Choisissez le critère', criterion_options)

        max_depth_value = streamlit.slider('Choisissez la profondeur maximale de l\'arbre', min_value=1, max_value=50, value=10, step=1)

        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'n_estimators_value': n_estimators_value,
                'criterion_value': criterion_value,
                'max_depth_value': max_depth_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'n_estimators_value': n_estimators_value,
                'criterion_value': criterion_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'n_estimators_value': n_estimators_value,
                'criterion_value': criterion_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }


        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])


    elif model=='XGBoost':
        streamlit.write('Voici le résultat pour l\'algorithme XGBoost :')
        
        n_estimators_value = streamlit.slider('Choisissez le nombre d\'estimators', min_value=1, max_value=50, value=10, step=1)

        learning_rate_value = streamlit.slider('Choisissez la valeur de learning rate', min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        max_depth_value = streamlit.slider('Choisissez la profondeur maximale de l\'arbre', min_value=1, max_value=50, value=10, step=1)

        # Define the data string
        if parcourir_data=='Train_Test_Split':
            data = {
                'model': model,
                'transform': transform,
                'learning_rate_value': learning_rate_value,
                'n_estimators_value': n_estimators_value,
                'max_depth_value': max_depth_value,
                'seed': seed,
                'test_size': test_size
            }
        elif parcourir_data=='K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'learning_rate_value': learning_rate_value,
                'n_estimators_value': n_estimators_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits
            }
        elif parcourir_data=='Repeated_K_fold_Cross_Validation':
            data = {
                'model': model,
                'transform': transform,
                'learning_rate_value': learning_rate_value,
                'n_estimators_value': n_estimators_value,
                'max_depth_value': max_depth_value,
                'n_splits': n_splits,
                'n_repeats': n_repeats
            }



        # Send the POST request to the server
        response = requests.post(link, data=data)

        res = response.json()
        streamlit.write('Accuracy Score : ',res['accuracy'])
        streamlit.write('Precision Score : ',res['precision'])
        streamlit.write('Recall Score : ',res['recall'])
        streamlit.write('F1 Score : ',res['f1'])
    
   



   







    


