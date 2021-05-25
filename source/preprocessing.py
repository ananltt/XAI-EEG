from EEGModels import EEGNet
from functions import *
import numpy as np

train_dataset, train_labels = load_dataset(train_ref, data_dir)
val_dataset, val_labels = load_dataset(val_ref, data_dir)
test_dataset, test_labels = load_dataset(test_ref, data_dir)
input_shape = (22, 1000, 1)

print("CNN")
model = CNN(input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss="mse",
              metrics=["accuracy"])
print(model.predict(test_dataset))
history = model.fit(train_dataset, train_labels, batch_size, num_epochs)
print(test_labels)

# model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1], kernLength=4)
# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
# history = model.fit(train_dataset, train_labels, batch_size, num_epochs)
