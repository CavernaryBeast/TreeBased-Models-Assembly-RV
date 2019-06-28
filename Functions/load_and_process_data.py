import pandas

def load_and_process_data(nombreArchivo):

    datosSinProcesar = pandas.read_csv( nombreArchivo + '.csv', header=None,
                           names=None )
        
    print(datosSinProcesar.shape)  # Número de filas y columnas
    #datosSinProcesar.head(10)

    from sklearn import preprocessing
    codificadores = []
    datosProcesados = pandas.DataFrame()
    for variable, valores in datosSinProcesar.iteritems():
        le = preprocessing.LabelEncoder()
        le.fit(valores)
        print('Codificación de valores para {}: {}'.format(variable, le.classes_))
        codificadores.append(le)
        datosProcesados[variable] = le.transform(valores)

    #examples_codificado.head(10)
    return datosProcesados;