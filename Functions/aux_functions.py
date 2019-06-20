import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

def divide_data(df_bootstrapped):
    
    
#    When selecting multiple columns or multiple rows in this manner,
#    remember that in your selection e.g.[1:5], the rows/columns selected 
#    will run from the first number to one minus the second number. 
#    e.g. [1:5] will go 1,2,3,4., [x,y] goes from x to y-1.
    
    examples = df_bootstrapped.iloc[:, 0:len(df_bootstrapped.columns)-1]
    labels = df_bootstrapped.iloc[:, len(df_bootstrapped.columns)-1]
    
    return examples,labels

    #def meta_algorithm(train_df, n_trees, n_features, dt_max_depth):
    
def meta_algorithm(train_df, n_trees, n_features):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, len(train_df))
        
       #tree = decisionTree.DecisionTreeClassifier(max_depth = dt_max_depth, )
        
        decisionTree = DecisionTreeClassifier()
        forest.append(tree)
    
    return forest
