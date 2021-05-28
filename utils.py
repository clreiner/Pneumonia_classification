import pandas as pd
import numpy as np


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



import keras
from keras.models import Sequential


import matplotlib.pyplot as plt


num_train_img = 5216
num_test_img = 624

batch_size = 64

def steps_per_epoch(batch_size):
    return num_train_img // batch_size+1

def val_steps(batch_size):
    return num_test_img // batch_size+1


def evaluate(model, train_set, test_set, steps_per_epoch=steps_per_epoch(batch_size), val_steps=val_steps(batch_size)):
    '''
        Displays the Train/Test loss and accuracy scores, a classification report
            and a confusion matrix

            Parameters:
                    model: a model fit to the training data
                    train_set (DirectoryIterator): the training data
                    test_set (DirectoryIterator): the test data

    '''
    train_eval = model.evaluate(train_set, steps=steps_per_epoch)
    test_eval = model.evaluate(test_set, steps=val_steps)

    print(f'Train Loss: {train_eval[0]: .3f}, Train Accuracy: {train_eval[1]: .3f}')
    print(f'Test Loss: {test_eval[0]: .3f}, Test Accuracy: {test_eval[1]: .3f}')

    #Confution Matrix and Classification Report
    Y_pred = np.round(model.predict_generator(test_set, val_steps))
    y_pred = Y_pred.tolist()
    cm = confusion_matrix(test_set.classes, y_pred)
    target_names = list(test_set.class_indices.keys())



    print('Classification Report')
    print(classification_report(test_set.classes, y_pred, target_names=target_names))

    print('Confusion Matrix')
    disp = ConfusionMatrixDisplay(cm,  display_labels=target_names)
    disp.plot(values_format = '')