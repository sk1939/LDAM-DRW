from __future__ import absolute_import
from __future__ import print_function
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from util import other_class

# includes for SL

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from util import get_lr_scheduler
from model import get_model
from loss import symmetric_cross_entropy
from callback_util import LoggerCallback, SGDLearningRateTracker

countMap = dict()
indexMap = dict()


# Mark data for deletion by marking as -1
def markForDeletion(data, startIndex, endIndex):
    for i in range(startIndex, endIndex):
        data[i] = -1
    return data


# Delete all data marked as -1
def deleteMarkedDataAndReturnNewData(data):
    data = data[data != -1]
    return data


# Balance Images
def balanceImages(images, deletedLabelIndexes):
    for index in deletedLabelIndexes:
        startIndex = (index - 1) * 28 * 28
        endIndex = index * 28 * 28
        images = markForDeletion(images, startIndex, endIndex)
    return deleteMarkedDataAndReturnNewData(images)


# Balance Labels
def balanceLabels(labelSet, minimumSampleCount):
    deletedIndexes = []
    for labels in countMap:
        if countMap[labels] > minimumSampleCount:
            while countMap[labels] != minimumSampleCount:
                indexToDelete = random.choice(indexMap[labels])
                startIndex = indexToDelete - 1
                endIndex = indexToDelete
                markForDeletion(labelSet, startIndex, endIndex)
                indexMap[labels].remove(indexToDelete)
                deletedIndexes.append(indexToDelete)
                countMap[labels] -= 1
    return deleteMarkedDataAndReturnNewData(labelSet), deletedIndexes


# Balance Images and Labels
def balanceDataset(images, labels, minimumSampleCount):
    labels, deletedIndexes = balanceLabels(labels, minimumSampleCount)
    images = balanceImages(images, deletedIndexes)
    return images, labels


# Add asymmetric noise
def addAsymmetricNoise(data, noise_ratio):
    source_class = [7, 2, 3, 5, 6]
    target_class = [1, 7, 8, 6, 5]
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(data == s)[0]
        n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        data[noisy_sample_index] = t
    return data


# Add symmetric noise
def addSymmetricNoise(data, noise_ratio):
    n_samples = data.shape[0]
    n_noisy = int(noise_ratio * n_samples / 100)
    class_index = [np.where(data == i)[0] for i in range(10)]
    class_noisy = int(n_noisy / 10)

    noisy_idx = []
    for d in range(10):
        noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
        noisy_idx.extend(noisy_class_index)

    for i in noisy_idx:
        data[i] = other_class(n_classes=10, current_class=data[i])

    return data


# Load from zip file -- Imbalanced data
training_images_unzipped = gzip.open('./train-images-idx3-ubyte.gz', 'r')
training_labels_unzipped = gzip.open('./train-labels-idx1-ubyte.gz', 'r')
test_images_unzipped = gzip.open('./t10k-images-idx3-ubyte.gz', 'r')
test_labels_unzipped = gzip.open('./t10k-labels-idx1-ubyte.gz', 'r')

image_size = 28
training_labels_unzipped.read(8)
label_buf = training_labels_unzipped.read(60000)
labelArray = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)

training_images_unzipped.read(16)
buf = training_images_unzipped.read(image_size * image_size * 60000)
imbalanced_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

test_labels_unzipped.read(8)
test_label_buf = test_labels_unzipped.read(10000)
test_labelArray = np.frombuffer(test_label_buf, dtype=np.uint8).astype(np.int32)

test_images_unzipped.read(16)
test_images_buf = test_images_unzipped.read(image_size * image_size * 10000)
test_images = np.frombuffer(test_images_buf, dtype=np.uint8).astype(np.float32)

for index, label in enumerate(labelArray):
    if label not in countMap:
        countMap[label] = 1
    else:
        countMap[label] += 1

    if label not in indexMap:
        indexMap[label] = list()
        indexMap[label].append(index)

    else:
        indexMap[label].append(index)

minimumSampleCount = countMap[min(countMap, key=countMap.get)]

training_labels_balanced_without_noise = np.copy(labelArray)
balanced_data, training_labels_balanced_without_noise = balanceDataset(imbalanced_data,
                                                                       training_labels_balanced_without_noise,
                                                                       minimumSampleCount)

label_array_imbalanced_asym = np.copy(labelArray)
label_array_imbalanced_asym = addAsymmetricNoise(label_array_imbalanced_asym, 40)

label_array_balanced_asym = np.copy(training_labels_balanced_without_noise)
label_array_balanced_asym = addAsymmetricNoise(label_array_balanced_asym, 40)

label_array_imbalanced_sym = np.copy(labelArray)
sym_noisy_imbalanced_labels = addSymmetricNoise(label_array_imbalanced_sym, 40)

label_array_balanced_sym = np.copy(training_labels_balanced_without_noise)
label_array_balanced_sym = addSymmetricNoise(label_array_balanced_sym, 40)

imbalanced_data = imbalanced_data.reshape(-1, image_size * image_size)
balanced_data = balanced_data.reshape(-1, image_size * image_size)
test_images = test_images.reshape(-1, image_size * image_size)

labelArray = np_utils.to_categorical(labelArray, 10)
label_array_imbalanced_asym = np_utils.to_categorical(label_array_imbalanced_asym, 10)
label_array_balanced_asym = np_utils.to_categorical(label_array_balanced_asym, 10)
label_array_imbalanced_sym = np_utils.to_categorical(label_array_imbalanced_sym, 10)
label_array_balanced_sym = np_utils.to_categorical(label_array_balanced_sym, 10)

test_labelArray = np_utils.to_categorical(test_labelArray, 10)


def get_data():
    return imbalanced_data, labelArray, labelArray, test_images, test_labelArray


def computeAndDisplayConfusionMatrix(test_data, prediction, classifier):
    cm = metrics.confusion_matrix(test_data, prediction, labels=classifier.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()


def displayROCCurve(test_labels, prediction, test_images, classifier):
    y_pred_proba = classifier.predict_proba(test_images)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
    auc = metrics.roc_auc_score(test_labels, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


def printStatistics(test_labels, prediction, dataset_type, classifier, test_images):
    print(dataset_type)
    print(f"Classification report for classifier {classifier}:\n"
          f"{metrics.classification_report(test_labels, prediction)}\n")
    computeAndDisplayConfusionMatrix(test_labels, prediction, classifier)
    # displayROCCurve(test_labels, prediction, test_images, classifier)


def predictWithSVM(data, label, dataset_type):
    oSvm = svm.SVC()
    oSvm.fit(data, label)
    prediction = oSvm.predict(test_images)
    printStatistics(test_labelArray, prediction, dataset_type, oSvm, test_images)


def predictWithLogisticRegression(data, label, dataset_type):
    oLogisticRegression = LogisticRegression()
    oLogisticRegression.fit(data, label)
    prediction = oLogisticRegression.predict(test_images)
    printStatistics(test_labelArray, prediction, dataset_type, oLogisticRegression, test_images)


##################SVM######################################
# predictWithSVM(imbalanced_data, labelArray, "Imbalanced without noise")
# predictWithSVM(balanced_data, training_labels_balanced_without_noise, "Balanced without noise")

# predictWithSVM(imbalanced_data, label_array_imbalanced_asym, "Imbalanced with asymmetric noise")
# predictWithSVM(imbalanced_data, label_array_imbalanced_sym, "Imbalanced with symmetric noise")

# predictWithSVM(balanced_data, label_array_balanced_asym, "Balanced with asymmetric noise")
# predictWithSVM(balanced_data, label_array_balanced_sym, "Balanced with symmetric noise")


###############################LOGISTIC REGRESSION######################################
# predictWithLogisticRegression(imbalanced_data, labelArray, "Imbalanced without noise")
# predictWithLogisticRegression(balanced_data, training_labels_balanced_without_noise, "Balanced without noise")

# predictWithLogisticRegression(imbalanced_data, label_array_imbalanced_asym, "Imbalanced with asymmetric noise")
# predictWithLogisticRegression(imbalanced_data, label_array_imbalanced_sym, "Imbalanced with symmetric noise")

# predictWithLogisticRegression(balanced_data, label_array_balanced_asym, "Balanced with asymmetric noise")
# predictWithLogisticRegression(balanced_data, label_array_balanced_sym, "Balanced with symmetric noise")


def train(dataset, model_name='sl', batch_size=128, epochs=50, noise_ratio=0, asym=False, alpha=1.0, beta=1.0):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param dataset:
    :param model_name:
    :param batch_size:
    :param epochs:
    :param noise_ratio:
    :return:
    """
    print('Dataset: %s, model: %s, batch: %s, epochs: %s, noise ratio: %s%%, asymmetric: %s, alpha: %s, beta: %s' %
          (dataset, model_name, batch_size, epochs, noise_ratio, asym, alpha, beta))

    # load data
    X_train, y_train, y_train_clean, X_test, y_test = get_data()
    print(y_train.shape)
    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    print("n_images", n_images, "num_classes", num_classes, "image_shape:", image_shape)

    # define P for forward and backward loss
    P = np.eye(num_classes)

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=num_classes)
    # model.summary()

    optimizer = SGD(lr=0.1, decay=1e-4, momentum=0.9)

    loss = symmetric_cross_entropy(alpha, beta)

    # model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy', 'Recall', 'Precision', 'f1_score']
    )

    ## do real-time updates using callbakcs
    callbacks = []

    # learning rate scheduler if use sgd
    lr_scheduler = get_lr_scheduler(dataset)
    callbacks.append(lr_scheduler)

    callbacks.append(SGDLearningRateTracker(model))

    # acc, loss, lid
    log_callback = LoggerCallback(model, X_train, y_train, y_train_clean, X_test, y_test, dataset, model_name,
                                  noise_ratio, asym, epochs, alpha, beta)
    callbacks.append(log_callback)

    # data augmentation
    datagen = ImageDataGenerator()
    datagen.fit(X_train)

    # train model
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=callbacks
                        )


imbalanced_data = imbalanced_data.reshape(-1, image_size, image_size, 1)
balanced_data = balanced_data.reshape(-1, image_size, image_size, 1)
test_images = test_images.reshape(-1, image_size, image_size, 1)

train('mnist', 'sl', 128, 10, 40, False, 0.01, 1.0)







