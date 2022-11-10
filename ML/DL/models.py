import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger


# ================Densennet==================================
def DenseNet121():
    DenseNet1211 = keras.models.Sequential([
        keras.applications.densenet.DenseNet121(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=[256,128,3]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])
    # ====================Optimizer============================
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0005)
    DenseNet1211.compile(loss="sparse_categorical_crossentropy",
                metrics=["accuracy"], 
                optimizer=optimizer1)
    return DenseNet1211
# ====================efficenet 2=========================
def eff_v2():
    eff_v21 = keras.models.Sequential([
        keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=[256,128,3]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])
    # ====================Optimizer============================
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.0005)
    eff_v21.compile(loss="sparse_categorical_crossentropy",
                metrics=["accuracy"], 
                optimizer=optimizer2)
    return eff_v21
# ===================efficenet==============================
def eff1(lr):
    eff1 = keras.models.Sequential([
        keras.applications.efficientnet.EfficientNetB1(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=[256,128,3]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])
    # ====================Optimizer============================
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=lr)
    eff1.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], 
                optimizer=optimizer3)
    return eff1
# ===================efficenet==============================
def eff0(lr):
    eff1 = keras.models.Sequential([
        keras.applications.efficientnet.EfficientNetB0(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=[256,128,3]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])
    # ====================Optimizer============================
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=lr)
    eff1.compile(loss="categorical_crossentropy",
                metrics=["accuracy"], 
                optimizer=optimizer3)
    return eff1
# ==================VGG16==================================
# vgg16 = keras.models.Sequential([
#     keras.applications.vgg16.VGG16(include_top=False, 
#                                             weights='imagenet', 
#                                             input_shape=[256,128,3]),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(32, activation="relu"),
#     keras.layers.Dense(16, activation="relu"),
#     keras.layers.Dense(8, activation="softmax")
# ])
# # ====================Optimizer============================
# optimizer4 = tf.keras.optimizers.Adam(learning_rate=0.0005)
# vgg16.compile(loss="sparse_categorical_crossentropy",
#                metrics=["accuracy"], 
#                optimizer=optimizer4)
# # ====================VGG19================================
# vgg19 = keras.models.Sequential([
#     keras.applications.vgg19.VGG19(include_top=False, 
#                                             weights='imagenet', 
#                                             input_shape=[256,128,3]),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(32, activation="relu"),
#     keras.layers.Dense(16, activation="relu"),
#     keras.layers.Dense(8, activation="softmax")
# ])
# # ====================Optimizer============================
# optimizer5 = tf.keras.optimizers.Adam(learning_rate=0.0005)
# vgg19.compile(loss="sparse_categorical_crossentropy",
#                metrics=["accuracy"], 
#                optimizer=optimizer5)
# ===================Resnet==============================
# resnet = keras.models.Sequential([
#     keras.applications.resnet_v2.ResNet101V2(include_top=False, 
#                                             weights='imagenet', 
#                                             input_shape=[256,128,3]),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(32, activation="relu"),
#     keras.layers.Dense(16, activation="relu"),
#     keras.layers.Dense(8, activation="softmax")
# ])
# # ====================Optimizer============================
# optimizer6 = tf.keras.optimizers.Adam(learning_rate=0.0005)
# resnet.compile(loss="sparse_categorical_crossentropy",
#                metrics=["accuracy"], 
#                optimizer=optimizer6)
# ======================regnet=============================
def regnet():
    regnet1 = keras.models.Sequential([
        keras.applications.regnet.RegNetX002(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=[256,128,3]),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="softmax")
    ])
    # ====================Optimizer============================
    optimizer7 = tf.keras.optimizers.Adam(learning_rate=0.0005)
    regnet1.compile(loss="sparse_categorical_crossentropy",
                metrics=["accuracy"], 
                optimizer=optimizer7)
    return regnet1
# ====================mobilenet=============================
# mobilenet = keras.models.Sequential([
#     keras.applications.mobilenet_v2.MobileNetV2(include_top=False, 
#                                             weights='imagenet', 
#                                             input_shape=[256,128,3]),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(32, activation="relu"),
#     keras.layers.Dense(16, activation="relu"),
#     keras.layers.Dense(8, activation="softmax")
# ])
# # ====================Optimizer============================
# optimizer8 = tf.keras.optimizers.Adam(learning_rate=0.0005)
# mobilenet.compile(loss="sparse_categorical_crossentropy",
#                metrics=["accuracy"], 
#                optimizer=optimizer8)
