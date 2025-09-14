from utility import (
  evaluate_demonstrations as evaluate_fewshots,
  evaluate_zeroshot,
  evaluate_finetuning,
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
   /len((set(cnt_ft) - set(cnt_icl)))


  return metric