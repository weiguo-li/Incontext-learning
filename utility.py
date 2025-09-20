# from IPython import test
from multiprocess import context
import torch
import datasets
import random
from tqdm import tqdm
from transformers import (Trainer, TrainingArguments)
from torch.nn.utils.rnn import pad_sequence
from typing import List
import torch.nn.functional as F
import gc
import psutil, os

LABEL_MAPS = {
    "sst2": {
        "id2str": {0: "negative", 1: "positive"},
        "str2id": {"negative": 0, "positive": 1}
    },
    "sst5": {
        "id2str": {0: "terrible", 1: "bad", 2: "neutral", 3: "good", 4: "great"},
        "str2id": {"terrible": 0, "bad": 1, "neutral": 2, "good": 3, "great": 4}
    },
    # Add more data types here
}

def get_label_map(data_type, map_type):
    try:
        return LABEL_MAPS[data_type][map_type]
    except KeyError:
        raise NotImplementedError(f"Label map for {data_type} and {map_type} not found.")

def custom_collator(features, tokenizer):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def preprocess_function(examples, tokenizer,data_type="sst2"):
    if data_type == "sst2":
      label_map = get_label_map(data_type, "id2str")
      
      inputs = [
          f"Sentence: {sentence} Label: {label_map[label]}"
          for sentence, label in zip(examples["sentence"], examples["label"])
      ]
    elif data_type == "sst5":
      label_map =get_label_map(data_type, "id2str")
      inputs = [
          f"Sentence: {sentence} Label: {label_map[label]}"
          for sentence, label in zip(examples["text"], examples["label"])
      ]
    else:
      raise NotImplementedError



    # Tokenize with padding/truncation
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding=False,
        max_length=256,
        # return_tensors="pt"
        return_tensors=None
    )
    # Set labels equal to input_ids for causal LM loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized



def format_context(demonstrations,data_type="sst2") -> str:
  """
    demonstrations: a Dataset object or None
    convert a demontrations into a prompt 
  """
  prompt = ""

  if demonstrations == None or demonstrations == "":
    return prompt

  if data_type == "sst2":
    for example in demonstrations:
      sentence, label = example["sentence"],example["label"]
      prompt += f"Sentence: {sentence} Label: {get_label_map(data_type, 'id2str')[label]} "
  elif data_type == "sst5":
    for example in demonstrations:
      sentence, label = example["text"],example["label"]
      prompt += f"Sentence: {sentence} Label: {get_label_map(data_type, 'id2str')[label]} "
  else:
    raise NotImplementedError
  
  return prompt


def format_query(context_prompt, sentence,data_type="sst2"):

  if data_type == "sst2" or data_type == "sst5":
    return context_prompt + f"Sentence: {sentence} Label:"

  return None

def format_demonstraions_query(deomnstrations, query, data_type="sst2"):
  """
  demonstraions: a Dataset object or None
  query: a string
  demonstraions can be an empty string"""
  context_prompt = format_context(deomnstrations, data_type=data_type)
  return format_query(context_prompt, query, data_type=data_type)


def evaluate_demonstrations(model, tokenizer, demonstrations, test_data, data_type="sst2", batch_size=32):
  
  # ensure no noisy logs and consistent padding
  if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

  context_prompt = format_context(demonstrations, data_type=data_type)
  cnt = 0
  candidates = []

  # pre-extract to avoid attribute lookups in loop
  to_device = model.device

  if data_type in LABEL_MAPS:
    label_id2str = get_label_map(data_type, "id2str")
  else:
    raise NotImplementedError


  # optional: pre-materialize test as list for slicing
  test_list = list(test_data)

  # import math
  total = len(test_list)
  for start in tqdm(range(0, total, batch_size), leave=True):
    batch = test_list[start:start + batch_size]
    if data_type == "sst2":
      prompts = [format_query(context_prompt, ex["sentence"], data_type=data_type) for ex in batch]
    elif data_type == "sst5":
      prompts = [format_query(context_prompt, ex["text"], data_type=data_type) for ex in batch]
    else:
      raise NotImplementedError


    with torch.inference_mode():
      inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
      ).to(to_device)

      output_ids = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
      )

    # compute true input lengths per item to slice the generated token(s)
    # input_lengths = inputs["attention_mask"].sum(dim=1)

    # compare predicted labels
    for i, ex in enumerate(batch):
      gen_ids = output_ids[i, -1]  # should be length 1
      label_txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

      if label_txt.lower() == label_id2str[ex["label"]].lower():
        cnt += 1
        candidates.append(ex["idx"])
        # print("ggod")

  print(f"Accuracy is {cnt / len(test_list)}")
  return cnt / len(test_list), candidates

def evaluate_zeroshot(base_model, tokenizer, test_data, data_type = "sst2"):
  acc, candidate = evaluate_demonstrations(base_model, tokenizer, demonstrations="", test_data = test_data, data_type=data_type)
  return acc, candidate

def evaluate_finetuning(fine_tuned_model, tokenizer, test_data, data_type="sst2"):
  acc, candidate = evaluate_demonstrations(fine_tuned_model, tokenizer, demonstrations="", test_data=test_data, data_type=data_type)
  return acc, candidate

def data_selection(model, tokenizer, train_data, test_data, data_type="sst2", num_data_points:int=32, seed_max:int=100, batch_size:int=32):
  ideal_seed = 0
  max_acc = 0
  ideal_demo = None
  candidate = []
  for seed in tqdm(range(seed_max)):
    random.seed(seed)
    random_indices = random.sample(range(len(train_data)), k=num_data_points)
    demonstrations = train_data.select(random_indices)
    eval_acc, candidate = evaluate_demonstrations(model, tokenizer, demonstrations, test_data, data_type=data_type, batch_size=batch_size)
    if eval_acc > max_acc:
      max_acc = eval_acc
      ideal_seed = seed
      ideal_demo = demonstrations

    print(f"Seed : {seed} --> acc is : {eval_acc}(icl) ")
  return ideal_seed, candidate, ideal_demo


def select_demonstraions(train_data, num_data_points:int=32, best_seed:int=0):
  random.seed(best_seed)
  random_indices = random.sample(range(len(train_data)), k=num_data_points)
  demonstrations = train_data.select(random_indices)
  return demonstrations


def finetune_model_eval(model, tokenizer, train_dataset, test_dataset, num_data_points = 32, data_type:str = "sst2", best_seed = 0):
  
  random.seed(best_seed)
  random_indices = random.sample(range(len(train_dataset)), k=num_data_points)
  demonstrations = train_dataset.select(random_indices)  # used as the training data for one epoch 

  evaluate_fewshots  = evaluate_demonstrations
  _, cnt_zsl = evaluate_zeroshot(model, tokenizer, test_dataset, data_type = data_type)
  _, cnt_icl = evaluate_fewshots(model,tokenizer, demonstrations, test_dataset, data_type)


  training_args = TrainingArguments(
    output_dir=f"./qwen3_{data_type}_lm",
    eval_strategy="steps",
    eval_steps=64,
    # save_steps=500,
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    # gradient_accumulation_steps=,
    num_train_epochs=1,
    learning_rate=5e-5,
    # weight_decay=0.01,
    # save_total_limit=1,
    save_strategy = "no",
    # fp16=True,
    # push_to_hub=False,
    report_to="none"
)


  # ** To do fine tuning here only need to fine tune the Key and Value
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=demonstrations.map(preprocess_function, batched=True, remove_columns=demonstrations.column_names, fn_kwargs={"tokenizer": tokenizer, "data_type":data_type}),
      eval_dataset=test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names, fn_kwargs={"tokenizer": tokenizer, "data_type":data_type}),
      # tokenizer=tokenizer,
      data_collator=lambda features: custom_collator(features, tokenizer)

  )
  # Start training
  trainer.train()


  print("training finished!")

  _, cnt_ft = evaluate_finetuning(model, tokenizer, test_dataset, data_type=data_type)
  metric =len(((set(cnt_ft) - set(cnt_zsl)) & (set(cnt_icl)-set(cnt_zsl)))) \
    /(len((set(cnt_ft) - set(cnt_icl))) + 1e-5)

  print(f" The Rec2FT is {metric}")
  return (metric)



def extract_hiddenstates(model,tokenizer,test_data: List[str], batch_size=2, return_all=True):
  """
  change batch size according to your GPU memory
  return_all: if False, yields hidden states instead of accumulating them (memory efficient)
  """
  # make sure pad on left side

  if type(test_data) == str:
    test_data = [test_data]
  
  if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"

  # if not return_all:
  #   # Memory efficient generator mode - yields each batch's hidden states
  #   for i in tqdm(range(0, len(test_data), batch_size), leave=True):
  #     batch_data = test_data[i:i+batch_size]
  #     with torch.no_grad():
  #         inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(model.device)
  #         output = model(**inputs, output_hidden_states=True, pad_token_id=tokenizer.pad_token_id)
          
  #         # Process and yield immediately to avoid accumulation
  #         hidden_states = tuple(h.detach().cpu()[:, -1, :].clone() for h in output.hidden_states)
          
  #         print(f"RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
          
  #         # Clean up before yielding
  #         del inputs, output
  #         torch.cuda.empty_cache() if torch.cuda.is_available() else None
          
  #         yield hidden_states
          
  #         # Clean up after yielding
  #         del hidden_states
  #         gc.collect()
  #   return

  # Original mode - accumulate all hidden states
  all_hidden_states = []

  for i in tqdm(range(0, len(test_data), batch_size), leave=True):
    batch_data = test_data[i:i+batch_size]
    with torch.no_grad():
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(model.device)
        output = model(**inputs, output_hidden_states=True, pad_token_id=tokenizer.pad_token_id)
        
        # Properly detach and clone hidden states to break computation graph
        hidden_states = tuple(h.detach().cpu()[:, -1, :].clone() for h in output.hidden_states)
        all_hidden_states.append(hidden_states)
        
        print(f"RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
        
        # Explicitly delete all references
        del inputs, output
        # Delete the hidden_states tuple after appending to list
        del hidden_states
        
        # Clear caches
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()  # Run garbage collection

  return all_hidden_states





def extract_attentionweights(model,tokenizer,test_data: List[str]):
  pass





