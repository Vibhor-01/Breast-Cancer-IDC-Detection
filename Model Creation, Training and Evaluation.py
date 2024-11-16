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