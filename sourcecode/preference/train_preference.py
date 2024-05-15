from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import wandb
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer, RewardConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a text regression model')
parser.add_argument('--train_file', type=str, help='Path to the training data', required=True)
parser.add_argument('--val_file', type=str, help='Path to the validation data', required=True)
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
# parser.add_argument('--text_col', type=str, help='Name of the text column', default='prompt')
# parser.add_argument('--scalar_col', type=str, help='Name of the scalar column', default='future_norm')
parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
parser.add_argument('--grad_accum', type=int, help='Gradient accumulation', default=1)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=2)
parser.add_argument('--log_train_every', type=int, help='Log train every', default=100)
# parser.add_argument('--eval_every', type=int, help='Evaluate every', default=1_000)
parser.add_argument('--eval_every', type=float, help='Evaluate every', default=0.2)
# output dir
parser.add_argument('--output_dir', type=str, help='Output directory', default='temp-output')
# parser.add_argument('--lora', type=bool, help='Use lora', default=False)
# lora flag
parser.add_argument('--lora', dest='lora', action='store_true')
# bits and bytes flag
parser.add_argument('--bits_and_bytes', dest='bits_and_bytes', action='store_true')
# parser.add_argument('--eval_every', type=int, help='Evaluate every', default=200)
# parser.add_argument('--loss', type=str, help='Loss function', default='mse')

args = parser.parse_args()

train_file = args.train_file
val_file = args.val_file
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
model_name = args.model_name
log_train_every = args.log_train_every
eval_every = args.eval_every
bits_and_bytes = args.bits_and_bytes

# wandb.init(project="reward_modeling", config={
#     "learning_rate": lr,
#     "epochs": epochs,
#     "batch_size": batch_size,
#     "model_name": model_name,
#     'log_train_every': log_train_every,
#     'eval_every': eval_every,
#     'loss': args.loss
# })

# if 'csv' in train_file:
if 'csv' in train_file:
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
elif 'tsv' in train_file:
    train_data = pd.read_csv(train_file, sep='\t')
    val_data = pd.read_csv(val_file, sep='\t')
else:
    raise ValueError('File type not supported')

# convert from chosen, rejected, and nmargin to 
# input_ids_chosen
# attention_mask_chosen
# input_ids_rejected
# attention_mask_rejected
# tokenize
tokenizer = AutoTokenizer.from_pretrained(model_name)
# set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = 'left'
def convert_data(data, tokenizer):
    chosen_tokenized = tokenizer(data['chosen'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    # to columns
    data['input_ids_chosen'] = chosen_tokenized['input_ids'].tolist()
    data['attention_mask_chosen'] = chosen_tokenized['attention_mask'].tolist()
    rejected_tokenized = tokenizer(data['rejected'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
    # to columns
    data['input_ids_rejected'] = rejected_tokenized['input_ids'].tolist()
    data['attention_mask_rejected'] = rejected_tokenized['attention_mask'].tolist()
    # if margin, convert to a tensor
    # if 'margin' in data.columns:
    #     data['margin'] = data['margin'].apply(lambda x: torch.tensor(x))
    return data

# convert the data
train_data = convert_data(train_data, tokenizer)[['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected', 'margin']]
# train_data = convert_data(train_data, tokenizer)[['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected']]
# to torch dataset
val_data = convert_data(val_data, tokenizer)[['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected', 'margin']]
# val_data = convert_data(val_data, tokenizer)[['input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected']]

# pandas df to torch dataset
class RewardDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.data.items()}
train_data = RewardDataset(train_data)
val_data = RewardDataset(val_data)

# need the num_labels for the model to output a scalar

if bits_and_bytes:
    from transformers import BitsAndBytesConfig
    # nf4_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, device_map='balanced', quantization_config=nf4_config)
    # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map='balanced')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",  # 4 bit
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32  # Ensure this matches your input tensor dtype
    )
    #model = AutoModelForCausalLM.from_pretrained(model_name, config=bnb_config, device_map='balanced')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, device_map='balanced', quantization_config=bnb_config)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, device_map='balanced', load_in_4bit=bits_and_bytes)
# parallelize the model
# model.parallelize()
# set num_labels to 1
# model.config.num_labels = 1

model.config.pad_token_id = tokenizer.eos_token_id
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# training_args = {
#     'output_dir': 'temp-output',
#     'overwrite_output_dir': True,
#     'per_device_train_batch_size': batch_size,
#     'per_device_eval_batch_size': batch_size,
#     'gradient_accumulation_steps': 1,
#     'learning_rate': lr,
#     'num_train_epochs': epochs,
#     'evaluation_strategy': 'steps',
#     'logging_strategy': 'steps',
#     'save_strategy': 'steps',
#     'eval_steps': eval_every,
#     'logging_steps': log_train_every,
#     'save_steps': eval_every,
#     'logging_dir': 'logs',
#     'save_total': 1,
#     'max_length': 512,
# }
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=lr,
    num_train_epochs=epochs,
    evaluation_strategy='steps',
    eval_steps=eval_every,
    # eval every .2 epoch
    # eval_steps=0.2,
    logging_strategy='steps',
    save_strategy='steps',
    logging_steps=log_train_every,
    save_steps=eval_every,
    logging_dir='logs',
    # save_total=1,
    # max_length=512,
    # pad_token_id=tokenizer.eos_token_id,
    # log to wandb
    report_to='wandb',
    # parallelize across all gpus
    # parallel_mode='pip
)


trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config if args.lora else None,
)
trainer.train()
