import os
import time
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import transformers as xfmers

def return_glue_task(tokenizer, dataset_name, task_name, max_seq_len=512, index=0, num_shards=1):    
    data, info = tfds.load(dataset_name, shuffle_files=False,
                           with_info=True)
    
    train_examples = info.splits["train"].num_examples
    valid_examples = info.splits["validation"].num_examples
    test_examples = info.splits["test"].num_examples
    num_labels = info.features["label"].num_classes
    
    print("Task:", dataset_name, ":")
    print("\tTrain:", train_examples)
    print("\tValid:", valid_examples)
    print("\tTest: ", test_examples)

    print("\t[1/3] Converting training dataset...")
    train_dataset = data["train"]
    train_dataset = train_dataset.shard(num_shards, index)
    train_dataset = xfmers.glue_convert_examples_to_features(train_dataset, tokenizer,
                                                             max_length=max_seq_len, task=task_name)

    print("\t[2/3] Converting validation dataset...")
    valid_dataset = data["validation"]
    valid_dataset = valid_dataset.shard(num_shards, index)
    valid_dataset = xfmers.glue_convert_examples_to_features(valid_dataset, tokenizer,
                                                             max_length=max_seq_len, task=task_name)
    
    print("\t[3/3] Converting test dataset...")
    test_dataset = data["validation"]
    test_dataset = xfmers.glue_convert_examples_to_features(test_dataset, tokenizer,
                                                            max_length=max_seq_len, task=task_name)
    
    
    return {"train_dataset": train_dataset,
            "valid_dataset": valid_dataset,
            "test_dataset": test_dataset,
            "train_examples": train_examples,
            "valid_examples": valid_examples,
            "test_examples": valid_examples, #test_examples,
            "shards": num_shards,
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
        
