import sklearn

def aplicaBootstrapping(datosProcesados):
    
    datosTrasBootstrapping = sklearn.utils.resample(datosProcesados, replace=True)
    
    return datosTrasBootstrapping;
