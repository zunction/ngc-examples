import time
import tensorflow.compat.v2 as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as models

def rn50(img_size=(224,224), num_class=2, weights="imagenet"):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3))
    base = models.ResNet50V2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def rn152(img_size=(224,224), num_class=2, weights="imagenet"):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3))
    base = models.ResNet152V2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def dn201(img_size=(224,224), num_class=2, weights="imagenet"):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3))
    base = models.DenseNet201(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model


def mobilenet(img_size=(224,224), num_class=2, weights="imagenet"):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3))
    base = models.MobileNetV2(input_tensor=input_layer, include_top=False, weights=weights)
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_class)(x)
    preds = layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model

