# Download required libraries

# pip install -q git+https://github.com/huggingface/transformers
# pip install -q sentencepiece
# pip install -q accelerate
# pip install bitsandbytes
# pip install -q -U git+https://github.com/huggingface/peft.git
# pip install datasets
# pip install fire

from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "KBlueLeaf/guanaco-7b-leh-v2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(
    model_id,
    use_fast=False)


model = LlamaForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "model.embed_tokens",
        "lm_head",
    ]
)
model = get_peft_model(model, config).to(torch.float32)
print_trainable_parameters(model)

from datasets import load_dataset

val_set_size = 5

def generate_prompt(data_point):
    # Remove first line since we actually don't need it.
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

train_on_inputs = True
cutoff_len = 512

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result



def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

data = load_dataset("json", data_files="./finetune.json")
if val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(
        generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(
        generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from typing import List
import transformers
import torch
import fire
import sys
import os

batch_size = 16
micro_batch_size = 2
num_epochs = 10
learning_rate = 3e-4
output_dir = "./CRYSTAL-fine"
group_by_length = False
wandb_run_name = ""

gradient_accumulation_steps = batch_size // micro_batch_size

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

wandb_project = ""

use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        # ddp doesn't like gradient checkpointing
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        # uncomment this line if your gpu support bf16
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        optim='adamw_torch',
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)

lora_config = LoraConfig.from_pretrained(output_dir)
model = get_peft_model(model, lora_config)

while True:
  text = "Enter Query: "
  inputs = tokenizer(text, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_new_tokens=50)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))



