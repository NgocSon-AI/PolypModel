import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(writer, cm, class_names, epoch):

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arrage(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max()/2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalignment="center", color = color)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion matrix', figure, epoch)    