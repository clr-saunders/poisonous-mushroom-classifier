import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import click
import os
from matplotlib import pyplot as plt

@click.command()
@click.argument('x_train')
@click.argument('y_train')
@click.argument('x_test')
@click.argument('y_test')

def main(x_train, y_train, x_test, y_test):

    x_train = pd.read_csv(x_train)
    y_train = pd.read_csv(y_train)
    y_train = y_train['is_poisonous']
    x_test = pd.read_csv(x_test)
    y_test = pd.read_csv(y_test)
    y_test = y_test['is_poisonous']

    preprocessor = OneHotEncoder(handle_unknown="ignore")

    dc_pipe = make_pipeline(
        preprocessor, 
        DummyClassifier(random_state=123)
    )

    dc_pipe.fit(x_train, y_train)

    cross_val_dc = pd.DataFrame(
        cross_validate(
            dc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})

    svc_pipe = make_pipeline(
        preprocessor, 
        SVC(random_state=123)
    )

    svc_pipe.fit(x_train, y_train)

    cross_val_svc = pd.DataFrame(
        cross_validate(
            svc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})


    y_pred = svc_pipe.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    disp.figure_.savefig("img/confusion_matrix.png")


if __name__ == '__main__':
    main()