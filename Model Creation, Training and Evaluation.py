#Importing Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from os import listdir
from tcn import TCN
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

# Data variables from Data Processing
Non_cancer_train, Non_cancer_temp, Non_cancer_train_labels, Non_cancer_temp_labels=train_test_split(normalized_non_cancer, non_cancer_labels, test_size=0.3, stratify=non_cancer_labels, random_state=42)
Cancer_train, Cancer_temp, Cancer_train_labels, Cancer_temp_labels=train_test_split(normalized_cancer, cancer_labels, test_size=0.3, stratify=cancer_labels, random_state=42)
Non_cancer_Val, Non_cancer_test, Non_cancer_val_labels, Non_cancer_test_labels=train_test_split(Non_cancer_temp, Non_cancer_temp_labels, test_size=0.5, stratify=Non_cancer_temp_labels, random_state=42)
Cancer_val, Cancer_test, Cancer_val_labels, Cancer_test_labels=train_test_split(Cancer_temp,Cancer_temp_labels, test_size=0.5, stratify=Cancer_temp_labels, random_state=42)

training_data=np.concatenate((Non_cancer_train, Cancer_train), axis=0)
training_labels=np.concatenate((Non_cancer_train_labels, Cancer_train_labels), axis=0)
validation_data=np.concatenate((Non_cancer_Val,Cancer_val), axis=0)
validation_labels=np.concatenate((Non_cancer_val_labels, Cancer_val_labels), axis=0)
testing_data=np.concatenate((Non_cancer_test, Cancer_test), axis=0)
testing_labels=np.concatenate((Non_cancer_test_labels, Cancer_test_labels), axis=0)

training_labels=to_categorical(training_labels,2)
validation_labels=to_categorical(validation_labels,2)
testing_labels=to_categorical(testing_labels,2)

# Model Creation
model=keras.Sequential([
    layers.Conv2D(32,(3,3), padding='same', activation='relu', input_shape=(50,50,3)),
    layers.MaxPooling2D(strides=2),
    layers.Conv2D(64,(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((3,3), strides=2),
    layers.Conv2D(128,(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((3,3),strides=2),
    layers.Conv2D(256,(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((3,3),strides=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model.summary()

# Fitting the Data
optimizer=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=keras.callbacks.EarlyStopping( monitor='val_loss', patience=10, restore_best_weights=True)

# Training the Model
history=model.fit(training_data, training_labels, validation_data=(validation_data, validation_labels), epochs=25, batch_size=256, callbacks=[early_stop])

# Evaluating the Model
model.evaluate(testing_data, testing_labels)

# Graphs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# Plotting Confusion Matrix
pred=model.predict(testing_data)
pred_classes=np.argmax(pred, axis=1)
def single_label_conversion(one_hot_labels):
    return one_hot_labels.argmax(axis=1)
real_training_labels=single_label_conversion(training_labels)
real_validation_labels=single_label_conversion(validation_labels)
real_test_labels=single_label_conversion(testing_labels)

# Plotting the confusion Matrix
accuracy=accuracy_score(real_test_labels, pred_classes)
print(f'Accuracy: {accuracy:.2f}')

precision=precision_score(real_test_labels, pred_classes)
print(f'Precision: {precision:.2f}')

recall=recall_score(real_test_labels, pred_classes)
print(f'Recall: {recall:.2f}')

f1=f1_score(real_test_labels, pred_classes)
print(f'F1: {f1:.2f}')

cm=confusion_matrix(real_test_labels, pred_classes)
plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Non-Cancer", "Cancer"], yticklabels=["Non-Cancer","Cancer"])
plt.title('confusion Matrix')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Saving the Model
model.save('CNN.h5')
