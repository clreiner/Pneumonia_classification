import pandas as pd
import numpy as np


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



import keras
from keras.models import Sequential


import matplotlib.pyplot as plt



def evaluate(model, train_set, test_set):
    '''
        Displays the Train/Test loss, precision and recall scores, a classification report
            and a confusion matrix

            Parameters:
                    model: a model fit to the training data
                    train_set (DirectoryIterator): the training data
                    test_set (DirectoryIterator): the test data

    '''
    train_eval = model.evaluate(train_set, steps=train_set.n//train_set.batch_size+1)
    test_eval = model.evaluate(test_set, steps=test_set.n//test_set.batch_size+1)

    print(f'Train Loss: {train_eval[0]: .3f}, Train Accuracy: {train_eval[1]: .3f}')
    print(f'Test Loss: {test_eval[0]: .3f}, Test Accuracy: {test_eval[1]: .3f}')

    #Confution Matrix and Classification Report
    test_set.reset()
    Y_pred = np.round(model.predict_generator(test_set, test_set.n//test_set.batch_size+1))
    y_pred = Y_pred.tolist()
    cm = confusion_matrix(test_set.classes, y_pred)
    target_names = list(test_set.class_indices.keys())



    print('Classification Report')
    print(classification_report(test_set.classes, y_pred, target_names=target_names))

    print('Confusion Matrix')
    disp = ConfusionMatrixDisplay(cm,  display_labels=target_names)
    disp.plot(values_format = '')
    

def plot_history(history):
    '''
        Plots the history of a model

            Parameters:
                    history: the history of fitting a model to training data
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))


    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history["val_" + met])
        ax[i].set_title(f"Model {met.capitalize()}")
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel(met.capitalize())
        ax[i].legend(["Train", "Validation"])