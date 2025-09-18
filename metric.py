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
  # get_label_map
)

import random 




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





def SimAOU(model,tokenizer,train_data, test_data, best_seed = 0, num_data_points = 32, data_type = "sst2"):
  demonstrations = select_demonstraions(train_data, num_data_points=32, best_seed = best_seed)

  if data_type == "sst2":
    prompts_zsl = [format_demonstraions_query("", data_point, data_type) for data_point in test_data["sentence"] ]
    prompts_icl = [format_demonstraions_query(context, query, data_type) for context, query in zip([demonstrations]*len(test_data), test_data["sentence"])]
    prompts_ft = prompts_zsl
  elif data_type == "sst5":
    prompts_zsl = [ format_demonstraions_query("", data_point, data_type) for data_point in test_data["text"] ]
    prompts_icl = [format_demonstraions_query(context, query, data_type) for context, query in zip([demonstrations]*len(test_data), test_data["text"])]
    prompts_ft = prompts_zsl  

  else:
    raise NotImplementedError

  # print(prompts_icl[:5])
  # print(prompts_ft[:5])

  icl_hiddensates = extract_hiddenstates(model, tokenizer, prompts_icl,batch_size=4)
  zsl_hiddensates = extract_hiddenstates(model, tokenizer, prompts_zsl,batch_size=8)

  return icl_hiddensates, zsl_hiddensates

  # fine the model first on the demosntraion and then extract the hidden states
  # ft_hiddensates = extract_hiddenstates(model, tokenizer, prompts_ft)