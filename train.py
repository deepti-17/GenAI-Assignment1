import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from model import build_model

# ---------------- CONFIG ----------------
BLOCK_SIZE = 128
TRAIN_BATCH = 8
EVAL_BATCH = 16
NUM_EPOCHS = 3
LR = 5e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET ----------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset["train"] = dataset["train"].filter(lambda x: x["text"] and not x["text"].isspace())
dataset["validation"] = dataset["validation"].filter(lambda x: x["text"] and not x["text"].isspace())

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

def tokenize_fn(e):
    return tokenizer(e["text"], return_special_tokens_mask=True)

tokenized_train = dataset["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_valid = dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    chunks = [concatenated[i:i+BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
    return {"input_ids": chunks}

train_dataset = tokenized_train.map(group_texts, batched=True, remove_columns=tokenized_train.column_names)
valid_dataset = tokenized_valid.map(group_texts, batched=True, remove_columns=tokenized_valid.column_names)

def collate_fn(batch):
    ids = [x["input_ids"] for x in batch]
    return tokenizer.pad({"input_ids": ids}, return_tensors="pt")

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn)

# ---------------- MODEL ----------------
model = build_model()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = max(1, len(train_loader) * NUM_EPOCHS)
warmup_steps = int(0.06 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- TRAINING ----------------
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        coll_batch = [{"input_ids": b.tolist()} for b in batch["input_ids"]]
        mb = data_collator(coll_batch)

        input_ids = mb["input_ids"].to(device)
        labels = mb["labels"].to(device)

        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            coll_batch = [{"input_ids": b.tolist()} for b in batch["input_ids"]]
            mb = data_collator(coll_batch)
            ids = mb["input_ids"].to(device)
            labels = mb["labels"].to(device)
            out = model(ids, labels=labels)
            eval_loss += out.loss.item()

    print(f"  Eval Loss: {eval_loss/len(valid_loader):.4f}")

# ---------------- SAVE ----------------
model.save_pretrained("saved_model_pytorch")
tokenizer.save_pretrained("saved_model_pytorch/tokenizer")

print("Training complete. Model saved.")
