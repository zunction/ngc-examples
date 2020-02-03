import time
import tensorflow.compat.v2 as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as models

def return_model_builder(model_name):
    models = {
        "rn50": create_rn50,
        "rn152": create_rn152,
    }
    return models

def create_rn50(img_size=(224,224), img_dtype="float32", num_class=2):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=img_dtype)
    base = models.ResNet50V2(input_tensor=input_layer, include_top=False, weights='imagenet')
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    preds = layers.Dense(num_class, activation="softmax", dtype='float32')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model

def create_rn152(img_size=(224,224), img_dtype="float32", num_class=2):
    input_layer = layers.Input(shape=(img_size[0],img_size[1],3), dtype=img_dtype)
    base = models.ResNet152V2(input_tensor=input_layer, include_top=False, weights='imagenet')
    base.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    preds = layers.Dense(num_class, activation="softmax", dtype='float32')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=preds)
    return model

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
