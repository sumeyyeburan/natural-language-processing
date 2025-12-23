!pip install -U transformers accelerate bitsandbytes datasets peft sentencepiece



from datasets import load_dataset

dataset = load_dataset("Naholav/CodeGen-Deep-5K")
dataset



dataset = dataset["train"].train_test_split(test_size=0.10, seed=42)
test_set = dataset["test"]

train_val = dataset["train"].train_test_split(test_size=0.0111, seed=42)
train_set = train_val["train"]
val_set = train_val["test"]

len(train_set), len(val_set), len(test_set)



def format_example(e):
    return {
        "text": (
            "You are an expert Python programmer. Please read the problem carefully before writing any Python code.\n"
            + e["input"]
            + "\n# Solution:\n"
            + e["solution"]
        )
    }

train_set = train_set.map(format_example)
val_set   = val_set.map(format_example)
test_set  = test_set.map(format_example)



train_set = train_set.remove_columns([c for c in train_set.column_names if c != "text"])
val_set   = val_set.remove_columns([c for c in val_set.column_names if c != "text"])
test_set  = test_set.remove_columns([c for c in test_set.column_names if c != "text"])

train_set[0]



from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)



lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qwen_lora_deep",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,
    warmup_ratio=0.03,

    learning_rate=2e-4,
    num_train_epochs=2,

    logging_steps=20,
    eval_steps=100,
    eval_strategy="steps",

    remove_unused_columns=False,

    save_steps=100,
    save_total_limit=5,
    fp16=True,
    report_to="none",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)



from transformers import Trainer, EarlyStoppingCallback
from transformers import TrainerCallback   # ← BU VAR
import json

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step: {state.global_step} | {logs}")


class SaveLossCallback(TrainerCallback):
    def __init__(self, filename):
        self.filename = filename
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.logs.append(logs)
            with open(self.filename, "w") as f:
                json.dump(self.logs, f, indent=2)

def collate(batch):
    texts = [example["text"] for example in batch]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        PrintLossCallback(),
        SaveLossCallback("deep_log_history.json"),
    ],
)



trainer.train()