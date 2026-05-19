import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

# this is for me since i'm using my cpu to train the model
torch.set_num_threads(os.cpu_count())

model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small', device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

#tokenize the data
def tokenize(batch):
    tokenized_inputs = tokenizer(batch["input"], max_length=128, truncation=True, padding="max_length") #tokenize inputs
    tokenized_outputs = tokenizer(batch["output"], max_length=128, truncation=True, padding="max_length") #tokenize given outputs 
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"] #labels of inputs should be the outputs
    return tokenized_inputs


# tokenize data set 
dataset = load_dataset('json', data_files='data.json')
tokenized_dataset = dataset.map(tokenize, batched=True)

#TrainingArguments
training_arguments = TrainingArguments(
    output_dir="training_directory", 
    per_device_train_batch_size=8,
    num_train_epochs=60,
    learning_rate=3e-4
)
#set up the trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset["train"]
)

#train and save the model as tm1 (TechurModel1)
trainer.train()
model.save_pretrained("tm1")
tokenizer.save_pretrained("tm1")
