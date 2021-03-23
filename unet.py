import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import pprint
from sklearn import metrics
import math
from tensorflow.keras import layers, backend, Model, utils
import tensorflow as tf
import shutil

def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def convolution_block(x, n_filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(n_filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, n_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, n_filters, (3,3))
    x = convolution_block(x, n_filters, (3,3))
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model(input_shape, n_filters, DropoutRatio=0.3):
    input_layer = tf.keras.Input(shape=input_shape)
    # 101 -> 50
    conv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, n_filters * 1)
    conv1 = residual_block(conv1, n_filters * 1, True)
    pool1 = layers.MaxPool2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, n_filters * 2)
    conv2 = residual_block(conv2, n_filters * 2, True)
    pool2 = layers.MaxPool2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, n_filters * 4)
    conv3 = residual_block(conv3, n_filters * 4, True)
    pool3 = layers.MaxPool2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)


    # Middle
    convm = layers.Conv2D(n_filters * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, n_filters * 8)
    convm = residual_block(convm, n_filters * 8, True)

    # 12 -> 25
    deconv3 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, n_filters * 4)
    uconv3 = residual_block(uconv3, n_filters * 4, True)

    # 25 -> 50
    deconv2 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, n_filters * 2)
    uconv2 = residual_block(uconv2, n_filters * 2, True)

    # 50 -> 101
    deconv1 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, n_filters * 1)
    uconv1 = residual_block(uconv1, n_filters * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = layers.Activation('sigmoid')(output_layer_noActi)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()
    return model

# gg = build_model((256,256,3), 32)


class Dataset(object):
    def __init__(self, type):
        self.batch_sizes = 4
        self.input_sizes = 256
        self.annotations = os.listdir('data_cut/label')
        np.random.shuffle(self.annotations)
        self.total_data = self.annotations[:170] if type == "train" else self.annotations[170:]
        self.check_batch = 160 if type == "train" else 60
        self.num_batch = math.ceil(self.check_batch/self.batch_sizes)
        self.num = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            batch_image = np.zeros((self.batch_sizes, self.input_sizes, self.input_sizes, 3), dtype=np.float32)
            batch_target = np.zeros((self.batch_sizes, self.input_sizes, self.input_sizes, 1), dtype=np.float32)
            i = 0
            while i < self.batch_sizes:
                filename = self.total_data[self.num+i]
                target, image = self.preprocess(filename)
                batch_image[i, ...] = image
                batch_target[i, ...] = target
                i+=1
            self.num += self.batch_sizes
            return batch_image, batch_target
        else:
            self.num = 0
            np.random.shuffle(self.total_data)
            raise StopIteration

    def preprocess(self, img_list):
        with rasterio.open(os.path.join("data_cut/image", img_list)) as src:
            image = src.read()[:3].transpose(1,2,0)
        with rasterio.open(os.path.join("data_cut/label", img_list)) as aaa:
            target = aaa.read().transpose(1,2,0)
        return target, image/255

    def __len__(self):
        print(self.num_batch)
        return self.num_batch

# gg = Dataset('train')
# for i in gg:
#     print('=================')

trainset = Dataset('train')
testset = Dataset('test')

def F1_Score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.numpy().flatten()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
    return metrics.auc(fpr, tpr)

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        output = model(image_data)
        total_loss = bce(target, output)
        # F1_score = F1_Score(target, output)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_steps.assign_add(1)

        with writer.as_default():
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            # tf.summary.scalar("loss/F1_score", F1_score, step=global_steps)
        writer.flush()
        
    return global_steps.numpy(), total_loss.numpy()

def validate_step(image_data, target):
    output = model(image_data)
    total_loss = bce(target, output)
    # F1_score = F1_Score(target, output)
        
    return total_loss.numpy()

model = build_model(input_shape=(256,256,3), n_filters = 32)
model.load_weights("model/unet.h5")


TRAIN_LOGDIR = 'log'
TRAIN_EPOCHS = 30
best_val_loss = 200
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
total_steps = TRAIN_EPOCHS * steps_per_epoch

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs {gpus}')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass
    
if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

optimizer = tf.keras.optimizers.Adam(lr = 0.00001)
# optimizer = tf.keras.optimizers.Adamax(beta_1 = tf.Variable(0.825), beta_2 = tf.Variable(0.99685))

for epoch in range(TRAIN_EPOCHS):
    for image_data, target in trainset:
        results = train_step(image_data, target)
        cur_step = (results[0] - 2) %steps_per_epoch + 1
        print("epoch:{:2.0f} step:{:5.0f}/{}, total_loss:{:7.2f}"
                .format(epoch, cur_step, steps_per_epoch, results[1]))

    count, F1_score, total_val = 0., 0, 0
    for image_data, target in testset:
        results = validate_step(image_data, target)
        count += 1
        total_val += results
        
    with validate_writer.as_default():
        tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
        # tf.summary.scalar("validate_loss/F1_score", F1_score/count, step=epoch)
    validate_writer.flush()
    print("\n\ntotal_val_loss:{:7.2f}\n\n".format(total_val/count))

    if best_val_loss>total_val/count:
        save_directory = os.path.join("model", f"unet.h5")
        model.save_weights(save_directory)
        best_val_loss = total_val/count