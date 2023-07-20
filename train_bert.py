# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertForSequenceClassification
from transformers import TrainingArguments,DataCollatorWithPadding,Trainer
import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
num_class = 2
device = "cuda"
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
cache_dir = '.'
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT",cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained("GroNLP/hateBERT",cache_dir=cache_dir,num_labels=2,id2label=id2label, label2id=label2id,)

#load data
train_path='./DiaSafety_dataset/train.json'
test_path = './DiaSafety_dataset/test.json'
train_df = pd.read_json(train_path)
test_df = pd.read_json(test_path)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
print(train_dataset)
print(test_dataset)

#preprocess data
def preprocess_function(examples):
    max_len =512
    context_list = ["context:"+context for context in examples["context"]]
    response_list = ["response:"+response for response in examples["response"]]
    label_list = [1 if label=='Safe' else 0 for label in  examples["label"]]
    toknized_data = tokenizer(context_list,response_list,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=max_len,  
                    return_tensors='pt')
    toknized_data['label'] = label_list
    return toknized_data
tokenized_train_dataset = train_dataset.map(preprocess_function,batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function,batched=True)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./bert_on_DiaSafety",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
#     compute_metrics=custom_metrics,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model()
trainer.save_state()
