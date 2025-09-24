# from prompt_toolkit import prompt
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
    custom_collator,
    extract_attn_weights,
    get_query_states,
    # get_label_map
)
import torch.nn.functional as F
from typing import List, Tuple
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
from tqdm import tqdm
import random
from customized_Qwen import AttentionWeightsCollector as AttentionCollector


def reorganize_last_token_hiddenstates(
    hiddenstates: List[Tuple[torch.Tensor]], all_position=False
) -> Tuple[torch.Tensor]:
    # hiddenstates: List of tuples, each tuple is (num_layers,) of tensors (batch, seq_len, hidden_dim)
    num_layers = len(hiddenstates[0])
    layerwise_last_token = [[] for _ in range(num_layers)]  # list for each layer

    for batch_tuple in hiddenstates:
        for layer_idx, layer_tensor in enumerate(batch_tuple):
            # Get last token hidden state for each item in batch
            if all_position == False:
                last_token = layer_tensor
            else:
                last_token = layer_tensor[:, -1, :]  # shape: (batch, hidden_dim)

            layerwise_last_token[layer_idx].append(last_token)

    # Concatenate along batch dimension for each layer
    layerwise_concat = tuple(
        torch.cat(layer_list, dim=0) for layer_list in layerwise_last_token
    )
    # Each element: shape (total_batch, hidden_dim)
    return layerwise_concat  #


def Rec2FTP(
    base_model,
    fine_tuned_model,
    tokenizer,
    test_data,
    demonstrations,
    data_type="sst2",
    best_seed=0,
):
    _, cnt_zsl = evaluate_zeroshot(
        base_model, tokenizer, test_data, data_type=data_type
    )
    _, cnt_ft = evaluate_finetuning(fine_tuned_model, tokenizer, test_data, data_type)
    _, cnt_icl = evaluate_fewshots(
        base_model, tokenizer, demonstrations, test_data, data_type
    )

    # print(cnt_zsl)
    # print(cnt_ft)
    # print(cnt_icl)
    metric = len(((set(cnt_ft) - set(cnt_zsl)) & (set(cnt_icl) - set(cnt_zsl)))) / (
        len((set(cnt_ft) - set(cnt_icl))) + 1e-5
    )

    return metric


def SimAOU(
    model,
    tokenizer,
    train_data,
    test_data,
    best_seed=0,
    num_data_points=32,
    data_type="sst2",
    **kwargs,
):
    demonstrations = select_demonstraions(
        train_data, num_data_points=num_data_points, best_seed=best_seed
    )
    prompts_zsl, prompts_icl, prompts_ft = create_prompts_for_data_type(
        demonstrations, test_data, data_type
    )

    icl_hiddensates = extract_hiddenstates(
        model, tokenizer, prompts_icl, batch_size=kwargs.get("batch_size_icl", 4)
    )  # List[Tuple: (num_layers) * (batch_size, seq_len, hidden_size)]
    zsl_hiddensates = extract_hiddenstates(
        model, tokenizer, prompts_zsl, batch_size=kwargs.get("batch_size_zsl", 16)
    )  # List[Tuple: (num_layers) * (batch_size, seq_len, hidden_size)]

    # train the model on demonstraions
    training_args = TrainingArguments(
        output_dir=f"./qwen3_{data_type}_lm",
        eval_strategy="steps",
        eval_steps=64,
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_strategy="no",
        report_to="none",
    )

    # ** To do fine tuning here only need to fine tune the Key and Value
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=demonstrations.map(
            preprocess_function,
            batched=True,
            remove_columns=demonstrations.column_names,
            fn_kwargs={"tokenizer": tokenizer, "data_type": data_type},
        ),
        eval_dataset=test_data.map(
            preprocess_function,
            batched=True,
            remove_columns=test_data.column_names,
            fn_kwargs={"tokenizer": tokenizer, "data_type": data_type},
        ),
        data_collator=lambda features: custom_collator(features, tokenizer),
    )
    # Start training
    trainer.train()
    print("training finished!")

    ft_hiddensates = extract_hiddenstates(
        model, tokenizer, prompts_ft, batch_size=kwargs.get("batch_size_ft", 16)
    )  # List[Tuple: (num_layers) * (batch_size, seq_len, hidden_size)]

    def cal_sim(
        ft_hiddensates: List[Tuple[torch.Tensor]],
        icl_hiddensates: List[Tuple[torch.Tensor]],
        zsl_hiddensates: List[Tuple[torch.Tensor]],
        norm_type: str = "l2",
    ) -> float:
        ft_hiddensates = reorganize_last_token_hiddenstates(
            ft_hiddensates
        )  # Tuple: (num_layers) * (total_batch, hidden_size)
        icl_hiddensates = reorganize_last_token_hiddenstates(icl_hiddensates)
        zsl_hiddensates = reorganize_last_token_hiddenstates(zsl_hiddensates)

        if norm_type == "l2":
            # L2 normalize
            ft_hiddensates = [
                layer / (layer.norm(dim=-1, keepdim=True) + 1e-5)
                for layer in ft_hiddensates
            ]
            icl_hiddensates = [
                layer / (layer.norm(dim=-1, keepdim=True) + 1e-5)
                for layer in icl_hiddensates
            ]
            zsl_hiddensates = [
                layer / (layer.norm(dim=-1, keepdim=True) + 1e-5)
                for layer in zsl_hiddensates
            ]
        elif norm_type == "layernorm":
            # Layer normalization
            ft_hiddensates = [
                (layer - layer.mean(dim=1, keepdim=True))
                / (layer.std(dim=1, keepdim=True) + 1e-5)
                for layer in ft_hiddensates
            ]
            icl_hiddensates = [
                (layer - layer.mean(dim=1, keepdim=True))
                / (layer.std(dim=1, keepdim=True) + 1e-5)
                for layer in icl_hiddensates
            ]
            zsl_hiddensates = [
                (layer - layer.mean(dim=1, keepdim=True))
                / (layer.std(dim=1, keepdim=True) + 1e-5)
                for layer in zsl_hiddensates
            ]
        else:
            raise NotImplementedError

        # compute delta
        delta_icl_zsl = [
            (icl_layer - zsl_layer)
            for icl_layer, zsl_layer in zip(icl_hiddensates, zsl_hiddensates)
        ]
        delta_ft_zsl = [
            (ft_layer - zsl_layer)
            for ft_layer, zsl_layer in zip(ft_hiddensates, zsl_hiddensates)
        ]
        random_generated = [
            torch.randn_like(icl_layer) for icl_layer in icl_hiddensates
        ]

        # compute the similarity with F.cosine_similarity
        sim_scores = [
            F.cosine_similarity(delta_icl, delta_ft, dim=-1).mean().item()
            for delta_icl, delta_ft in zip(delta_icl_zsl, delta_ft_zsl)
        ]  # shape (num_layers,)
        sim_scores_random = [
            F.cosine_similarity(delta_icl, rand_layer, dim=-1).mean().item()
            for delta_icl, rand_layer in zip(delta_icl_zsl, random_generated)
        ]  # shape (num_layers,)
        # sum over
        sim_scores = sum(sim_scores) / len(sim_scores)
        sim_scores_random = sum(sim_scores_random) / len(sim_scores_random)

        return sim_scores, sim_scores_random

    sim_scores, sim_scores_random = cal_sim(
        ft_hiddensates, icl_hiddensates, zsl_hiddensates, norm_type="l2"
    )

    print(
        f"SimAOU(FT) score of  data type {data_type} is {sim_scores:.10f} \n SimAOU(Random) score of  data type {data_type} is {sim_scores_random:.10f}"
    )

    return sim_scores, sim_scores_random


def SimAM(
    model,
    tokenizer,
    train_data,
    test_data,
    best_seed=0,
    num_data_points=32,
    data_type="sst2",
    **kwargs,
):
    """Similarity of Attention Maps"""
    demonstrations = select_demonstraions(
        train_data, num_data_points=num_data_points, best_seed=best_seed
    )
    prompts_zsl, prompts_icl, prompts_ft = create_prompts_for_data_type(
        demonstrations, test_data, data_type
    )

    zsl_attn_weights = extract_attn_weights(
        model, tokenizer, prompts_zsl, batch_size=kwargs.get("batch_size_zsl", 1)
    )

    icl_attentionweights = extract_attn_weights(
        model, tokenizer, prompts_icl, batch_size=kwargs.get("batch_size_icl", 1)
    )

    # train the model on demonstraions
    training_args = TrainingArguments(
        output_dir=f"./qwen3_{data_type}_lm",
        eval_strategy="steps",
        eval_steps=64,
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=demonstrations.map(
            preprocess_function,
            batched=True,
            remove_columns=demonstrations.column_names,
            fn_kwargs={"tokenizer": tokenizer, "data_type": data_type},
        ),
        eval_dataset=test_data.map(
            preprocess_function,
            batched=True,
            remove_columns=test_data.column_names,
            fn_kwargs={"tokenizer": tokenizer, "data_type": data_type},
        ),
        data_collator=lambda features: custom_collator(features, tokenizer),
    )
    # Start training
    trainer.train()
    print("training finished!")

    ft_attn_weights = extract_attn_weights(
        model, tokenizer, prompts_ft, batch_size=kwargs.get("batch_size_ft", 1)
    )

    # return ft_attn_weights, icl_attentionweights, zsl_attn_weights

    def cal_sim(
        ft_attn_weights: List[List[torch.Tensor]],
        icl_attn_weights: List[List[torch.Tensor]],
        zsl_attn_weights: List[List[torch.Tensor]],
    ) -> float:
        """
        ft_attn_weights, icl_attn_weights, zsl_attn_weights:
            List of length num_layers, each element is a list of torch.Tensor of shape (batch_size, num_heads, seq_len)
            each element is a tensor of shape (batch_size, num_heads, seq_len)
        """

        num_layers = len(ft_attn_weights)
        num_samples = len(
            ft_attn_weights[0]
        )  # assuming all have same number of samples this works for batch size 1

        accumlated_sim_ft_icl = 0.0
        accumlated_sim_zsl_icl = 0.0

        for layer_idx in range(num_layers):
            for sample_idx in range(num_samples):
                valid_length = ft_attn_weights[layer_idx][sample_idx].shape[-1]

                accumlated_sim_ft_icl += (
                    F.cosine_similarity(
                        ft_attn_weights[layer_idx][
                            sample_idx
                        ],  # (batch_size = 1,num_heads, seq_len)
                        icl_attn_weights[layer_idx][sample_idx][
                            :, :, -valid_length:
                        ],  # (batch_size = 1,num_heads, seq_len)
                        dim=-1,
                    )
                    .mean()
                    .item()
                )

                accumlated_sim_zsl_icl += (
                    F.cosine_similarity(
                        zsl_attn_weights[layer_idx][
                            sample_idx
                        ],  # (batch_size = 1,num_heads, seq_len)
                        icl_attn_weights[layer_idx][sample_idx][
                            :, :, -valid_length:
                        ],  # (batch_size = 1,num_heads, seq_len)
                        dim=-1,
                    )
                    .mean()
                    .item()
                )

        return accumlated_sim_ft_icl / (
            num_layers * num_samples
        ), accumlated_sim_zsl_icl / (num_layers * num_samples)  # type: ignore

    sim_scores_ft, sim_scores_bf_ft = cal_sim(
        ft_attn_weights, icl_attentionweights, zsl_attn_weights
    )
    print(
        f"SimAM(FT) score  is {sim_scores_ft:.10f} \n SimAM(Before FT) score is {sim_scores_bf_ft:.10f}"
    )
    return sim_scores_ft, sim_scores_bf_ft


def Kendall(
    model,
    tokenizer,
    train_data,
    test_data,
    data_type="sst2",
    best_seed=0,
    num_data_points=32,
    **kwargs,
):


    demonstrations = select_demonstraions(
        train_data, num_data_points=num_data_points, best_seed=best_seed
    )

    prompts_zsl, prompts_icl, _ = create_prompts_for_data_type(
        demonstrations, test_data, data_type
    )

    query_states_demonstration = get_query_states(model, tokenizer, demonstrations, data_type)
    query_states_test = get_query_states(model, tokenizer, prompts_zsl, data_type, query_type="query_zsl")
    inner_product = calculate_query_inner_product(query_states_demonstration, query_states_test)

    all_attn_weights2demonqueries = []
    batch_size = kwargs.get("batch_size", 1) # batch size can only be 1 for similicity of attention maps
    demon_area = inner_product[0].shape[0]
    for i in tqdm(range(0, len(prompts_icl), batch_size)):
        batch_data = prompts_icl[i : i + batch_size]
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, pad_token_id=tokenizer.pad_token_id)
            attn_weights = outputs.attentions  # Tuple of length (num_layers,) each is a tensor of Size(batch_size, num_heads, seq_len, seq_len)
            
            attn_weights2demonqueries = [item[0,:,-1,:demon_area].detach().clone().sum(dim = 0).unsqueeze(0) for item in attn_weights] # shape(num_heads,query_context) -> size(query_context)
            all_attn_weights2demonqueries.append(attn_weights2demonqueries)

    #concatenate all batches
    all_attn_weights2demonqueries = list(zip(*all_attn_weights2demonqueries)) # list of length num_layers, each is a list of tensor of shape (batch_size, query_context)
    all_attn_weights2demonqueries = [torch.cat(item, dim=0) for item in all_attn_weights2demonqueries] # list of length num_layers, each is a tensor of shape (total_batch, query_context)

    # randomly generated attention weights for control experiment
    random_all_attn_weights2demonqueries = [torch.rand_like(item).transpose(1,0) for item in all_attn_weights2demonqueries]

    kendall = calculate_kendall_tau_simple(inner_product, all_attn_weights2demonqueries)
    kendall_random = calculate_kendall_tau_simple(all_attn_weights2demonqueries, random_all_attn_weights2demonqueries)
    print(f"Kendall's tau of {data_type} is {kendall:.10f}, random baseline is {kendall_random:.10f}")

    return kendall, kendall_random


def create_prompts_for_data_type(demonstrations, test_data, data_type) -> Tuple[List[str], List[str], List[str]]:
    """
    Create zero-shot learning, in-context learning, and fine-tuning prompts based on data type.

    Args:
        demonstrations: Selected demonstration examples
        test_data: Test dataset
        data_type: Type of dataset ("sst2" or "sst5")

    Returns:
        tuple: (prompts_zsl, prompts_icl, prompts_ft)
    """
    if data_type == "sst2":
        text_field = "sentence"
    elif data_type == "sst5":
        text_field = "text"
    else:
        raise NotImplementedError(f"Data type '{data_type}' is not supported")

    # Zero-shot learning prompts (no demonstrations)
    prompts_zsl = [
        format_demonstraions_query("", data_point, data_type)
        for data_point in test_data[text_field]
    ]

    # In-context learning prompts (with demonstrations)
    prompts_icl = [
        format_demonstraions_query(context, query, data_type)
        for context, query in zip(
            [demonstrations] * len(test_data), test_data[text_field]
        )
    ]

    # Fine-tuning prompts (same as zero-shot)
    prompts_ft = prompts_zsl

    return prompts_zsl, prompts_icl, prompts_ft


def calculate_query_inner_product(query_states_demonstration, query_states_test):
    """
    Calculate inner product between demonstration and test query states.
    
    Args:
        query_states_demonstration: tuple of lists, each list contains tensors from q_proj layers
        query_states_test: tuple of lists, each list contains tensors from q_proj layers
    
    Returns:
        layer_wise_inner_products: list of inner products for each layer
    """
    num_layers = len(query_states_demonstration)
    layer_wise_inner_products = []
    
    for layer_idx in range(num_layers):
        # Get query states for this layer
        demo_queries = query_states_demonstration[layer_idx]  # List of tensor shape(1, seq_len, hidden_dim)
        test_queries = query_states_test[layer_idx]  # List of tensors

        demo_layer = demo_queries[0].squeeze(0)  # (seq_len, hidden_dim)

        # Concatenate all batches for test data
        test_layer = torch.cat([item[:,-1,:] for item in test_queries], dim=0)  # (total_batch, hidden_dim)
        
        # Calculate inner product
        inner_product = torch.matmul(demo_layer, test_layer.T)  # (seq_len, total_batch)
        layer_wise_inner_products.append(inner_product)

    return layer_wise_inner_products

def calculate_kendall_tau_simple(inner_product, all_attn_weights2demonqueries):
    """
    Simple Kendall's tau calculation between inner products and attention weights.
    """
    from scipy.stats import kendalltau
    import numpy as np
    
    layer_kendall_taus = []
    
    for layer_idx in range(len(inner_product)):
        inner_prod_layer = inner_product[layer_idx]  # (seq_len, total_batch)
        attn_weights_layer = all_attn_weights2demonqueries[layer_idx]  # (total_batch, seq_len)
        
        batch_taus = []
        for batch_idx in range(inner_prod_layer.shape[1]):
            inner_vec = inner_prod_layer[:, batch_idx].cpu().numpy()
            attn_vec = attn_weights_layer[batch_idx, :].cpu().numpy()
            
            tau, _ = kendalltau(inner_vec, attn_vec)
            if not np.isnan(tau):  # Skip NaN values
                batch_taus.append(tau)
        
        layer_avg_tau = np.mean(batch_taus) if batch_taus else 0
        layer_kendall_taus.append(layer_avg_tau)
    
    return np.mean(layer_kendall_taus)

