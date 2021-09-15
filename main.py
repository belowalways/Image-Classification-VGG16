import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16

path_list = ['[Task1]Training Images/0.buildings/',
             '[Task1]Training Images/1.forest/',
             '[Task1]Training Images/2.glacier/',
             '[Task1]Training Images/3.mountain/',
             '[Task1]Training Images/4.sea/',
             '[Task1]Training Images/5.street/']

N_CLASSES = 6
RESIZED_IMAGES = (224, 224, 3)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --------- Data Labeling ---------


# make image array with path list, image size
def make_img_arr(path_list, resize_to):

    images = []
    labels = []

    for i in range(len(path_list)):
        file_list = os.listdir(path_list[i])

        for file in file_list:
            img = cv2.imread(path_list[i] + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if resize_to:
                if resize_to[0] < img.shape[0]:
                    img = cv2.resize(img, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_AREA)
                else:
                    img = cv2.resize(img, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_CUBIC)

            images.append(img.astype(np.float32))
            labels.append(i)

    x_train = np.array(images, np.float32)
    y_train = np.array(labels, np.int)

    return x_train, y_train


x_train, y_train = make_img_arr(path_list, RESIZED_IMAGES)

# for normalize
x_train = x_train / 255.0


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=34)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, stratify=y_train, random_state=34)

print(x_train.shape, x_test.shape, x_valid.shape)

# train, test, valid = 6126:851:1532 (6:2:2)

# --------- Model ---------

transfer_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
transfer_model.trainable = False
transfer_model.summary()
finetune_model = models.Sequential()
finetune_model.add(transfer_model)
finetune_model.add(Flatten())
finetune_model.add(Dense(2048, activation="relu"))
finetune_model.add(Dropout(0.2))
finetune_model.add(Dense(1024, activation="relu"))
finetune_model.add(Dropout(0.4))
finetune_model.add(BatchNormalization())
finetune_model.add(Dense(6, activation="softmax"))
finetune_model.summary()

finetune_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"])
history = finetune_model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_valid, y_valid), validation_steps=4)

score = finetune_model.evaluate(x_test, y_test, verbose=2)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

finetune_model.save('model_name.h5')

# --------- Plot ---------

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
y_vloss = history.history["val_loss"]
y_loss = history.history["loss"]

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker=".", c="red", label="Trainset_acc")
plt.plot(x_len, val_acc, marker=".", c="lightcoral", label="val_accuracy")
plt.plot(x_len, y_vloss, marker=".", c="cornflowerblue", label="val_loss")
plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

plt.legend(loc="upper right")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss/acc")
plt.show()

