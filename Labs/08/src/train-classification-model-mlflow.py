# import libraries
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score ,roc_curve
import matplotlib.pyplot as plt

def main(args):
    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, args.max_iter, X_train, y_train)

    # evaluate model
    eval_model(model, X_test, y_test)


def get_data(path):
    print("Reading data...")
    df = pd.read_csv(path)
    
    return df

def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

def train_model(reg_rate, max_iteration ,X_train, y_train):
    mlflow.log_param("reg_rate", reg_rate)
    mlflow.log_param("max_iteration", max_iteration)
    print("Training model...")
    model = LogisticRegression(C=1/reg_rate, solver="liblinear", max_iter=max_iteration).fit(X_train, y_train)

    return model


def getAccuracyScore(Y_actual, Y_prediction):
    """Get acuracy score in pct

    Args:
        Y_actual (_type_): Y_test or Y actual values
        Y_prediction (_type_): predicetd value

    Returns:
        _type_: accuracy in pct format
    """
    acc_score = accuracy_score(Y_actual, Y_prediction) * 100
    mlflow.log_metric("accuracy", acc_score)
    return '{:.4f}%'.format(acc_score)

def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc_score = getAccuracyScore(y_test, y_hat)
    print('Accuracy:', acc_score)

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)

    tn,fp,fn,tp = confusion_matrix(y_test, y_hat).ravel()
    f1_score = tp/(tp+((fn+fp)/2))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)

    print(f"F1 Score: {'{:.4f}'.format(f1_score)}")
    print(f"Recall: {'{:.4f}'.format(recall)}")
    print(f"Precision: {'{:.4f}'.format(precision)}")

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, help='training data file path', default='diabetes.csv')
    parser.add_argument('--reg_rate', type=float, help='regularization rate')
    parser.add_argument('--max_iter', type=int, help='maximum iteration')
    args = parser.parse_args()
    main(args)
