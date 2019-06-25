import pandas as pd
import numpy as np
import random as random
import sklearn
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def load_and_process_data(nombreArchivo):

    datosSinProcesar = pd.read_csv( nombreArchivo + '.csv', header=None,
                           names=None )
        
    print(datosSinProcesar.shape)  # Número de filas y columnas
    #datosSinProcesar.head(10)

    from sklearn import preprocessing
    codificadores = []
    datosProcesados = pd.DataFrame()
    for variable, valores in datosSinProcesar.iteritems():
        le = preprocessing.LabelEncoder()
        le.fit(valores)
        print('Codificación de valores para {}: {}'.format(variable, le.classes_))
        codificadores.append(le)
        datosProcesados[variable] = le.transform(valores)

    #examples_codificado.head(10)
    return datosProcesados;

def bootstrapping(train_df):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=len(train_df))
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

def aplicaBootstrapping(datosProcesados):
    
    datosTrasBootstrapping = sklearn.utils.resample(datosProcesados, replace=True)
    
    return datosTrasBootstrapping;


def divide_data(df_bootstrapped):
    
    
#    When selecting multiple columns or multiple rows in this manner,
#    remember that in your selection e.g.[1:5], the rows/columns selected 
#    will run from the first number to one minus the second number. 
#    e.g. [1:5] will go 1,2,3,4., [x,y] goes from x to y-1.
    
    examples = df_bootstrapped.iloc[:, 0:len(df_bootstrapped.columns)-1]
    labels = df_bootstrapped.iloc[:, len(df_bootstrapped.columns)-1]
    
    return examples,labels

#   Function to implement the Random Subspace Method
def random_subspace(bootstrapped_df, random_subspace):
    
    n_columns = bootstrapped_df.shape[1]
    column_indices = list(range(n_columns - 1))    # Excluding the last column which is the label
    n_columns_choosed = int((n_columns-1)*random_subspace)
    
    if (random_subspace < 0) or (random_subspace > 1):
        print('If you want the Random Subspace Method to work, you have to choose a number betweet 0 and 1')
        
    else:
        column_indices = random.sample(population=column_indices, k=n_columns_choosed)
        
    return column_indices

def select_columns(data, column_indices):
    
    #potential_splits = {}
    columns_chosen = {}
    for column_index in column_indices:          
        values = data.iloc[:, column_index]
        #We can take only the unique values if we want, cause the bootstrapping can introduce some values that are equal
        #unique_values = np.unique(values)
        
        #potential_splits[column_index] = unique_values
        columns_chosen[column_index] = values
    columns_chosen = DataFrame.from_dict(columns_chosen)
    #potential_splits = DataFrame.from_dict(unique_values)
    
    return columns_chosen
    #return potential_splits


    #def meta_algorithm(train_df, n_trees, n_features, dt_max_depth):
    
def train_forest(file_name, n_trees, n_features, max_depth):
    
    processed_data = load_and_process_data(file_name)
    
    forest = []
    indices_chosen = []
    decisionTree = DecisionTreeClassifier(max_depth=max_depth)
    for i in range(n_trees):
        
        #First, we apply the Bootstrapping
        df_bootstrapped = bootstrapping(processed_data)
        #Second, we divide the data into features(examples) and labels
        examples, labels = divide_data(df_bootstrapped)
        #Third, we execute the Random Subspace Method to the examples
        column_indices = random_subspace(examples, n_features)
        columns_selected = select_columns(examples, column_indices)
        #Fourth, we execute the fitting
        tree = decisionTree.fit(columns_selected, labels)
        #tree = decisionTree.DecisionTreeClassifier(max_depth = dt_max_depth, )
        indices_chosen.append(column_indices)
        forest.append(tree)
    
    return forest, indices_chosen

def forest_predictions(file_name, forest, indices_chosen):
    
    processed_data = load_and_process_data(file_name)
    examples, labels = divide_data(processed_data)
    
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        columns_chosen = select_columns(examples, indices_chosen[i])
        predictions = forest[i].predict(columns_chosen)
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions, df_predictions

def balancearConjunto(nombreFichero):
    test_data = load_and_process_data('adult_test')
    examples_test, labels_test = divide_data(test_data)
    income_porcentajes = labels_test.value_counts(normalize=True)
    if(income_porcentajes.iloc[0] == 0,5):
        if(income_porcentajes.iloc[0] > 0,5):
            for i in labels_test.keys():
                if (labels_test[i] == 0):
                    labels_test.drop(i)
                    break

        if(income_porcentajes.iloc[1] > 0,5):
            for i in labels_test.keys():
                if (labels_test[i] == 1):
                    labels_test.drop(i)
                    break 
                    
    return labels_test;

def calcularAccuracy(labels, forest_predictions, labels_balanceado):
    tasa_acierto = accuracy_score(labels, forest_predictions)
    tasa_acierto_balanceado = balanced_accuracy_score(labels_balanceado, forest_predictions)
    return tasa_acierto, tasa_acierto_balanceado;
