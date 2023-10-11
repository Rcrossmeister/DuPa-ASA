import tensorflow as tf
from transformers import AutoTokenizer, create_optimizer, TFAutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import torch

imdb = load_dataset("imdb")
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# 对 IMDb 文本进行编码
imdb_texts = imdb["text"]
tokenized_texts = tokenizer(imdb_texts, truncation=True, padding=True, return_tensors="tf")

# 训练集标签（示例数据，请根据实际情况替换）
train_labels = [1, 0]

# 创建自定义数据集
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_texts), train_labels))
num_epochs=3

# 初始化模型
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 优化器
batches_per_epoch = len(train_dataset) // batch_size
total_train_steps = batches_per_epoch * num_epochs
optimizer, _ = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

# 编译模型
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])

# 模型训练
model.fit(train_dataset.batch(batch_size), epochs=num_epochs)
