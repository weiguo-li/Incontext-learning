from prompt_toolkit import prompt
from utility import (
  evaluate_demonstrations as evaluate_fewshots,
  evaluate_zeroshot,
  evaluate_finetuning,
  extract_hiddenstates,
  extract_attentionweights,
  select_demonstraions, 
  format_query, 
  LABEL_MAPS, 
  format_demonstraions_query,
  preprocess_function,
  custom_collator
  # get_label_map
)
import torch.nn.functional as F
from typing import List, Tuple
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
from tqdm import tqdm
import random 

def reorganize_last_token_hiddenstates(hiddenstates: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # hiddenstates: List of tuples, each tuple is (num_layers,) of tensors (batch, seq_len, hidden_dim)
    num_layers = len(hiddenstates[0])
    layerwise_last_token = [[] for _ in range(num_layers)]  # list for each layer

    for batch_tuple in hiddenstates:
        for layer_idx, layer_tensor in enumerate(batch_tuple):
            # Get last token hidden state for each item in batch
            last_token = layer_tensor[:, -1, :]  # shape: (batch, hidden_dim)
            layerwise_last_token[layer_idx].append(last_token)

    # Concatenate along batch dimension for each layer
    layerwise_concat = tuple(torch.cat(layer_list, dim=0) for layer_list in layerwise_last_token)
    # Each element: shape (total_batch, hidden_dim)
    return layerwise_concat #



def Rec2FTP(base_model,fine_tuned_model, tokenizer, test_data, demonstrations, data_type = "sst2", best_seed = 0):
  _, cnt_zsl = evaluate_zeroshot(base_model, tokenizer, test_data, data_type = data_type)
  _, cnt_ft = evaluate_finetuning(fine_tuned_model, tokenizer, test_data, data_type)
  _, cnt_icl = evaluate_fewshots(base_model,tokenizer, demonstrations, test_data, data_type)

  # print(cnt_zsl)
  # print(cnt_ft)
  # print(cnt_icl)
  metric =len(((set(cnt_ft) - set(cnt_zsl)) & (set(cnt_icl)-set(cnt_zsl)))) \
   /(len((set(cnt_ft) - set(cnt_icl))) + 1e-5)


  return metric





def SimAOU(model,tokenizer,train_data, test_data, best_seed = 0, num_data_points = 32, data_type = "sst2", **kwargs):
  demonstrations = select_demonstraions(train_data, num_data_points=num_data_points, best_seed = best_seed)

  if data_type == "sst2":
    prompts_zsl = [format_demonstraions_query("", data_point, data_type) for data_point in test_data["sentence"] ]
    prompts_icl = [format_demonstraions_query(context, query, data_type) for context, query in zip([demonstrations]*len(test_data), test_data["sentence"])]
    prompts_ft = prompts_zsl
  elif data_type == "sst5":
    prompts_zsl = [format_demonstraions_query("", data_point, data_type) for data_point in test_data["text"] ]
    prompts_icl = [format_demonstraions_query(context, query, data_type) for context, query in zip([demonstrations]*len(test_data), test_data["text"])]
    prompts_ft = prompts_zsl  

  else:
    raise NotImplementedError

  icl_hiddensates = extract_hiddenstates(model, tokenizer, prompts_icl, batch_size=kwargs.get("batch_size_icl", 8)) # List[Tuple: (num_layers) * (batch_size, seq_len, hidden_size)]
  zsl_hiddensates = extract_hiddenstates(model, tokenizer, prompts_zsl, batch_size=kwargs.get("batch_size_zsl", 16)) # List[Tuple: (num_layers) * (batch_size, seq_len, hidden_size)]



  # train the model on demonstraions
  training_args = TrainingArguments(
    output_dir=f"./qwen3_{data_type}_lm",
    eval_strategy="steps",
    eval_steps=64,
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    save_strategy = "no",
    report_to="none"
)


  # ** To do fine tuning here only need to fine tune the Key and Value
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=demonstrations.map(preprocess_function, batched=True, remove_columns=demonstrations.column_names, fn_kwargs={"tokenizer": tokenizer, "data_type":data_type}),
      eval_dataset=test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names, fn_kwargs={"tokenizer": tokenizer, "data_type":data_type}),
      data_collator=lambda features: custom_collator(features, tokenizer)

  )
  # Start training
  trainer.train()
  print("training finished!")

  ft_hiddensates = extract_hiddenstates(model, tokenizer, prompts_ft, batch_size=8)

  def cal_sim(
    ft_hiddensates: List[Tuple[torch.Tensor]],
    icl_hiddensates: List[Tuple[torch.Tensor]],
    zsl_hiddensates: List[Tuple[torch.Tensor]],
    norm_type: str = "l2"
) -> float:
    ft_hiddensates = reorganize_last_token_hiddenstates(ft_hiddensates) # Tuple: (num_layers) * (total_batch, hidden_size)
    icl_hiddensates = reorganize_last_token_hiddenstates(icl_hiddensates)
    zsl_hiddensates = reorganize_last_token_hiddenstates(zsl_hiddensates)

    if norm_type == "l2":
    # L2 normalize
      ft_hiddensates = [layer / (layer.norm(dim=-1, keepdim=True) + 1e-5) for layer in ft_hiddensates]
      icl_hiddensates = [layer / (layer.norm(dim=-1, keepdim=True) + 1e-5) for layer in icl_hiddensates]
      zsl_hiddensates = [layer / (layer.norm(dim=-1, keepdim=True) + 1e-5) for layer in zsl_hiddensates]  
    elif norm_type == "layernorm":
      # Layer normalization
      ft_hiddensates = [(layer - layer.mean(dim=1, keepdim=True)) / (layer.std(dim=1, keepdim=True) + 1e-5) for layer in ft_hiddensates]
      icl_hiddensates = [(layer - layer.mean(dim=1, keepdim=True)) / (layer.std(dim=1, keepdim=True) + 1e-5) for layer in icl_hiddensates]
      zsl_hiddensates = [(layer - layer.mean(dim=1, keepdim=True)) / (layer.std(dim=1, keepdim=True) + 1e-5) for layer in zsl_hiddensates]
    else:
      raise NotImplementedError

    # compute delta
    delta_icl_zsl = [ (icl_layer - zsl_layer) for icl_layer, zsl_layer in zip(icl_hiddensates, zsl_hiddensates)]
    delta_ft_zsl = [ (ft_layer - zsl_layer) for ft_layer, zsl_layer in zip(ft_hiddensates, zsl_hiddensates)]
    random_generated = [ torch.randn_like(icl_layer) for icl_layer in icl_hiddensates]

    # compute the similarity with F.cosine_similarity
    sim_scores = [ F.cosine_similarity(delta_icl, delta_ft, dim=-1).mean().item() for delta_icl, delta_ft in zip(delta_icl_zsl, delta_ft_zsl)] # shape (num_layers,)
    sim_scores_random = [ F.cosine_similarity(delta_icl, rand_layer, dim=-1).mean().item() for delta_icl, rand_layer in zip(delta_icl_zsl, random_generated)] # shape (num_layers,)
  # sum over 
    sim_scores = sum(sim_scores) / len(sim_scores)
    sim_scores_random = sum(sim_scores_random) / len(sim_scores_random)

    return sim_scores, sim_scores_random


 
  sim_scores, sim_scores_random = cal_sim(ft_hiddensates, icl_hiddensates, zsl_hiddensates, norm_type = "l2")


  print(f"SimAOU(FT) score of  data type {data_type} is {sim_scores:.4f} \n SimAOU(Random) score of  data type {data_type} is {sim_scores_random:.4f}")

  return sim_scores
