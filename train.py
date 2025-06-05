from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd

# Load CSV and convert to JSONL (optional)
df = pd.read_csv("dataset/sample_data.csv")
df = df.rename(columns={"input": "source", "output": "target"})
df[["source", "target"]].to_json("dataset/formatted_data.json", orient="records", lines=True)

# Load dataset
dataset = load_dataset(
    "csv",
    data_files="dataset/sample_data.csv",
    split="train",
    column_names=["input", "output"],
    delimiter=",",
    header=0,
)

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Clean output column to be strings (batched)
def clean_batch(batch):
    batch['output'] = [str(x) if x is not None else "" for x in batch['output']]
    return batch

dataset = dataset.map(clean_batch, batched=True)

# Preprocess
def preprocess(batch):
    model_inputs = tokenizer(
        batch["input"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["output"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)


# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_dir="./logs",
#    evaluation_strategy="no",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

