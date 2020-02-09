import os
import time
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import transformers as xfmers

def return_glue_task(tokenizer, dataset_name, task_name, max_seq_len=512):    
    data, info = tfds.load(dataset_name, shuffle_files=False,
                           with_info=True)
    
    train_examples = info.splits["train"].num_examples
    valid_examples = info.splits["validation"].num_examples
    num_labels = info.features["label"].num_classes
    
    print("Task:", dataset_name, ":")
    print("\tTrain:", train_examples)
    print("\tValidation", valid_examples)

    print("\tConverting training dataset...")
    train_dataset = xfmers.glue_convert_examples_to_features(data["train"], tokenizer,
                                                             max_length=max_seq_len, task=task_name)

    print("\tConverting validation dataset...")
    valid_dataset = xfmers.glue_convert_examples_to_features(data["validation"], tokenizer,
                                                             max_length=max_seq_len, task=task_name)
    
    return {"train_dataset": train_dataset,
            "valid_dataset": valid_dataset,
            "train_examples": train_examples,
            "valid_examples": valid_examples,
            "num_labels": num_labels}


def _tf_get_model(pretrained_name):
    if "roberta" in pretrained_name:
        mapping = xfmers.TFRobertaForSequenceClassification
    elif "distilbert" in pretrained_name:
        mapping = xfmers.TFDistilBertForSequenceClassification
    elif "albert" in pretrained_name:
        mapping = xfmers.TFAlbertForSequenceClassification
    elif "bert" in pretrained_name:
        mapping = xfmers.TFBertForSequenceClassification
    elif "xlm" in pretrained_name:
        mapping = xfmers.TFXLMForSequenceClassification
    elif "xlnet" in pretrained_name:
        mapping = xfmers.TFXLNetForSequenceClassification
    else:
        raise NotImplementedError
    return mapping


def create_tokenizer(model_name):
    tokenizer = xfmers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def create_model(model_name, num_labels=2):
    config = xfmers.AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = _tf_get_model(model_name).from_pretrained(model_name, config=config)
    return model


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
