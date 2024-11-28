import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np


def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred  # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Load a dataset (replace with your dataset)
dataset = load_dataset("ag_news")

# Check dataset format
print(dataset["train"][0])

print(dataset)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)


data = dataset['train'].train_test_split(test_size=0.1, seed=42)
data['val'] = data.pop("test")
data['test'] = dataset['test']

print(data)

pos_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().label.value_counts()[1])
neg_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().label.value_counts()[0])

# print(data['train'].to_pandas()['text'].str.len().max())
# print(data['train'].to_pandas()['text'].str.split().str.len().max())
#
# print(tokenizer(data['train'][0]['text']))
# print(len(tokenizer(data['train'][0]['text'])))


train_data = data["train"].remove_columns(['text'])
eval_data = data["val"].remove_columns(['text'])


model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    device_map='auto',
    num_labels=4
)
model.config.pad_token_id = model.config.eos_token_id

# FREEZE WEIGHTS
for param in model.parameters():
    param.requires_grad = False

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    inference_mode=False,  # Training mode
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.1  # Dropout for LoRA layers
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
print(model.print_trainable_parameters())


# Training
lr = 1e-4
batch_size = 8
num_epochs = 5

lora_trainer = WeightedCELossTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=transformers.TrainingArguments(
        output_dir="lora-classification",
        learning_rate=lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=False,
        gradient_checkpointing=True
    ),
    data_collator=transformers.DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

model.config.use_cache = False
lora_trainer.train()

# TRAINING
# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=train_data,
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=4,
#         warmup_steps=100,
#         max_steps=500,
#         learning_rate=2e-4,
#         logging_steps=1,
#         output_dir='outputs',
#         auto_find_batch_size=True
#     ),
#     data_collator=transformers.DataCollatorWithPadding(tokenizer)
# )
# model.config.use_cache = False
# trainer.train()

torch.save(model.state_dict(), 'lora_classifier.pt')
