# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import zipfile
import json
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

METRICS_PATH = 'files/output/metrics.json'

def load_data():
    with zipfile.ZipFile('files/input/train_data.csv.zip') as z:
        with z.open('train_default_of_credit_card_clients.csv') as f:
            train = pd.read_csv(f)

    with zipfile.ZipFile('files/input/test_data.csv.zip') as z:
        with z.open('test_default_of_credit_card_clients.csv') as f:
            test = pd.read_csv(f)

    return train, test

def cleaning_data(train, test):

    # Renombrar la columna "default payment next month" a "default"
    train = train.rename(columns={'default payment next month': 'default'})
    test = test.rename(columns={'default payment next month': 'default'})

    # Remuever la columna "ID"
    train = train.drop(columns=['ID'])
    test = test.drop(columns=['ID'])

    # Eliminar los registros con informacion no disponible
    train = train.dropna()
    test = test.dropna()

    # Agrupar valores de EDUCATION > 4 en la categoría "others"
    train['EDUCATION'] = train['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    test['EDUCATION'] = test['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return train, test


def split_data(train, test):
    # Paso 2
    x_train = train.drop(columns=['default'])
    y_train = train['default']

    x_test = test.drop(columns=['default'])
    y_test = test['default']

    return x_train, y_train, x_test, y_test

def use_pipeline():
    pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('rf', RandomForestClassifier())
    ])

    return pipeline

def optimize_hyperparameters(x_train,y_train,pipeline):
    param_grid = {
        'rf__n_estimators': [10, 50, 100],
        'rf__max_depth': [2, 10, 20]
        #,
        #'rf__min_samples_split': [2, 5, 10],
        #'rf__min_samples_leaf': [1, 2, 4],
        #'rf__bootstrap': [True, False]
    }

    grid = GridSearchCV(
        pipeline, 
        param_grid,
        cv=10, 
        scoring='balanced_accuracy'
    )
    grid.fit(x_train, y_train)
    
    return grid

def save_model(grid):
    # Ensure the directory exists
    model_dir = 'files/models'
    model_path = os.path.join(model_dir, 'model.pkl.gz')

    # Debugging statements to verify the directory and file path
    print(f"Saving model to {model_path}")
    assert os.path.exists(model_dir), f"Directory {model_dir} does not exist"

    # Save the model to a compressed file
    with gzip.open(model_path, 'wb') as file:
        # save with pickle
        pickle.dump(grid, file)
        #joblib.dump(grid, file)


def write_json_file(file_path, data):
    # Ensure the directory exists
    metrics_dir = 'files/output'
    os.makedirs(metrics_dir, exist_ok=True)

    print(data)
    # write to file
    with open(file_path, 'w') as file:
        # write to file
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')
    
    print(f"Metrics saved to {METRICS_PATH}")

def get_metrics(grid, x_train, y_train, x_test, y_test):

    dict_template = {
        'dataset': None,
        'precision': None,
        'balanced_accuracy': None,
        'recall': None,
        'f1_score': None
    }

    train_template = dict_template.copy()
    test_template = dict_template.copy()

    # Calculate metrics
    #train_score = grid.score(x_train, y_train)
    #test_score = grid.score(x_test, y_test)

    # Predictions
    y_train_pred = grid.predict(x_train)
    y_test_pred = grid.predict(x_test)

    # Metrics
    #train_report = classification_report(y_train, y_train_pred)
    #test_report = classification_report(y_test, y_test_pred)

    train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')

    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')


    # Fill in the template for train metrics
    train_template = {}
    train_template['dataset'] = 'train'
    train_template['precision'] = train_precision
    train_template['balanced_accuracy'] = train_balanced_accuracy
    train_template['recall'] = train_recall
    train_template['f1_score'] = train_f1

    # Fill in the template for test metrics
    test_template = {}
    test_template['dataset'] = 'test'
    test_template['precision'] = test_precision
    test_template['balanced_accuracy'] = test_balanced_accuracy
    test_template['recall'] = test_recall
    test_template['f1_score'] = test_f1


    # Add metrics to JSON file line
    # Read existing metrics if the file exists
    #if os.path.exists(METRICS_PATH):
    #   pass 
    #else:
    #    # create it
    #    with open(METRICS_PATH, 'w', encoding="utf-8") as f:
    #        json.dump([], f, indent=4)
        
    metrics = []
    # Append new metrics
    metrics.append(train_template)
    metrics.append(test_template)

    return metrics

def get_confusion_matrix(grid, x_train, y_train, x_test, y_test):
    
    cm_train = confusion_matrix(y_train, grid.predict(x_train))
    cm_test = confusion_matrix(y_test, grid.predict(x_test))

    # Convert NumPy int64 to native Python int
    cm_train = cm_train.tolist()
    cm_test = cm_test.tolist()

    # {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
    
    # train metrics
    train_metrics = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {
            'predicted_0': cm_train[0][0],
            'predicted_1': cm_train[0][1]
        },
        'true_1': {
            'predicted_0': cm_train[1][0],
            'predicted_1': cm_train[1][1]
        }
    }
    

    # test metrics
    test_metrics = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {
            'predicted_0': cm_test[0][0],
            'predicted_1': cm_test[0][1]
        },
        'true_1': {
            'predicted_0': cm_test[1][0],
            'predicted_1': cm_test[1][1]
        }
    }



    # Combine train and test metrics
    metrics = [
        train_metrics,
        test_metrics
    ]

    return metrics



def main():
    # Paso 0
    train, test = load_data()
    # Paso 1
    train,test = cleaning_data(train, test)
    # Paso 2
    x_train, y_train, x_test, y_test = split_data(train, test)
    # Paso 3
    pipeline = use_pipeline()
    # Paso 4
    grid = optimize_hyperparameters(x_train, y_train, pipeline)
    # Paso 5
    save_model(grid)
    # Paso 6
    metrics = get_metrics(grid, x_train, y_train, x_test, y_test)
    # Paso 7
    cm_metrics = get_confusion_matrix(grid, x_train, y_train, x_test, y_test)

    # Save metrics to JSON file
    existing_metrics = metrics + cm_metrics
    write_json_file(METRICS_PATH, existing_metrics)

    print('Proceso finalizado')

if  __name__ == '__main__':
    main()
