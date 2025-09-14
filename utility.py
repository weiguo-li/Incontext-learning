from IPython import test
from multiprocess import context
import torch
import datasets
import random

label_map_sst2_id2str = {0: "negative", 1: "positive"}
label_map_sst2_str2id = {"Negative": 0, "Positive": 1}
label_map_sst5 = {}


def format_context(demonstrations,data_type="sst2") -> str:
  """
    convert a demontrations into a prompt 
  """
  prompt = ""

  if demonstrations == None:
    return prompt

  if data_type == "sst2":
    for example in demonstrations:
      sentence, label = example["sentence"],example["label"]
      prompt += f"Sentence: {sentence} Label: {label_map_sst2_id2str[label]} "
  
  return prompt


def format_query(context_prompt, sentence,data_type="sst2"):

  if data_type == "sst2":
    return context_prompt + f"Sentence: {sentence} Label:"

  return None


def evaluate_demonstrations(model,tokenizer, demonstrations, test_data, data_type="sst2"):
  
  context_prompt = format_context(demonstrations,data_type=data_type)
  cnt = 0
  candidates = []
  for example in test_data:
    sentence = example["sentence"]
    input_prompt = format_query(context_prompt, sentence, data_type=data_type)
    # print(input_prompt)
    with torch.no_grad():
      inputs = tokenizer(input_prompt, return_tensors = "pt").to(model.device)
      
      output_ids = model.generate(
      **inputs,
      max_new_tokens = 1, 
      do_sample = False
      )

    generated_only_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    Label = tokenizer.decode(generated_only_ids, skip_special_tokens = True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if Label.strip() == label_map_sst2_id2str[example["label"]]:
      cnt += 1
      candidates.append(example['idx'])

    # print(generated_text)
    # print(f"The labe is {Label}, the ground truth is {example["label"]}")
    # print("----\n")
  
  print(f"Accuracy is {cnt / len(test_data)}")
  return cnt/len(test_data), candidates

def evaluate_zeroshot(base_model, tokenizer, test_data, data_type = "sst2"):
  acc, candidate = evaluate_demonstrations(base_model, tokenizer, demonstrations="", test_data = test_data, data_type=data_type)
  return acc, candidate

def evaluate_finetuning(fine_tuned_model, tokenizer, test_data, data_type="sst2"):
  acc, candidate = evaluate_demonstrations(fine_tuned_model, tokenizer, demonstrations="", test_data=test_data, data_type=data_type)
  return acc, candidate

def data_selection(model, tokenizer, train_data, test_data,data_type = "sst2", num_data_points:int = 32, seed_max:int= 100):
  """
    to select 32 data points by default that can achieve the 
    best performance on the test_data.

    to find a random seed eg. from 0 to 100.

  """
  ideal_seed = 0
  max_acc = 0
  ideal_demo = None
  for seed in range(seed_max):

    random.seed(seed)

    random_indices = random.sample(range(len(train_data)), k = num_data_points)
    demonstrations = train_data.select(random_indices)
    eval_acc, candidate = evaluate_demonstrations(model, tokenizer, demonstrations, test_data, data_type=data_type)
    # return
    if eval_acc > max_acc:
      max_acc = eval_acc
      ideal_seed = seed
      ideal_demo = demonstrations

  
  return ideal_seed, candidate, ideal_demo