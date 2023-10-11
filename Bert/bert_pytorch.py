from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from transformers import TrainingArguments, Trainer
import evaluate

# 加载 IMDb 数据集
imdb = load_dataset("imdb")
print(imdb)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理函数
def preprocess_function(examples):
   return tokenizer(examples['text'], truncation=True, padding=True)

# 对 IMDb 文本进行编码
imdb_texts = imdb['train']['text']
tokenized_texts = tokenizer(imdb_texts, truncation=True, padding=True, return_tensors="pt")

# 定义自定义数据集
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

# 训练集标签（示例数据，请根据实际情况替换）
train_labels = [1, 0]

# 创建自定义数据集
train_dataset = CustomDataset(tokenized_texts, train_labels)

# 初始化评估器
accuracy = evaluate.load("accuracy")

# 计算评估指标的函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 初始化模型参数
model_name = "bert-base-uncased"
num_labels = 2

# 训练参数
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

# 初始化训练器
trainer = Trainer(
     model_name,
     args=training_args,
     train_dataset=train_dataset,
     data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
     compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

