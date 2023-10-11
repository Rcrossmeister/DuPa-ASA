from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import torch

imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)

#tokenized_imdb = imdb.map(preprocess_function, batched=True)
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
imdb_texts = imdb["text"]
# 使用分词器进行编码，直接调用 __call__ 方法
tokenized_texts = tokenizer(imdb_texts, truncation=True, padding=True, return_tensors="pt")

# 创建数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_labels = [1, 0]
train_dataset = CustomDataset(tokenized_texts, train_labels)



accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#id2label = {0: "NEGATIVE", 1: "POSITIVE"}
#label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
      )

training_args = TrainingArguments(
     output_dir="./bert-finetune",
     learning_rate=2e-5,
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
     load_best_model_at_end=True,
 )

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_imdb["train"],
     eval_dataset=tokenized_imdb["test"],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics=compute_metrics,
 )

trainer.train()

from transformers import create_optimizer
import tensorflow as tf

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

import tensorflow as tf

model.compile(optimizer=optimizer)  # No loss argument!

from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
from transformers.keras_callbacks import PushToHubCallback

push_to_hub_callback = PushToHubCallback(
     output_dir="my_awesome_model",
     tokenizer=tokenizer,
 )

from transformers.keras_callbacks import KerasMetricCallback

# 创建计算指标的回调
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

# 删除PushToHubCallback，只保留metric_callback
callbacks = [metric_callback]

# 开始模型训练
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)







