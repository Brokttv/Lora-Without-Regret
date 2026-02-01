from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import os


def get_dataloaders(batch_size=32):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("ag_news")
    
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=36)
    
    dataset = DatasetDict({
        "train": train_val_split["train"],
        "val": train_val_split["test"],
        "test": dataset["test"]
    })
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    

    num_workers = os.cpu_count()
    
    train_dataset = tokenized_dataset["train"].select(range(10000))
    val_dataset = tokenized_dataset["val"].select(range(2000))
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_data = DataLoader(tokenized_dataset["test"], batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_data, val_data, test_data
