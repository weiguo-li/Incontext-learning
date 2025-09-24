# from IPython import test
from multiprocess import context
import torch
import datasets
import random
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import torch.nn.functional as F
import gc
import psutil, os

from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Qwen3Attention,
)
from transformers.utils.deprecation import deprecate_kwarg
from customized_Qwen import AttentionWeightsCollector as AttentionCollector

LABEL_MAPS = {
    "sst2": {
        "id2str": {0: "negative", 1: "positive"},
        "str2id": {"negative": 0, "positive": 1},
    },
    "sst5": {
        "id2str": {0: "terrible", 1: "bad", 2: "neutral", 3: "good", 4: "great"},
        "str2id": {"terrible": 0, "bad": 1, "neutral": 2, "good": 3, "great": 4},
    },
    # Add more data types here
}


def enable_kv_only_training(model):
    """
    Freeze all parameters except key and value weights in attention layers.
    Only key and value parameters will have gradients during training.
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only key and value parameters in attention layers
    for name, module in model.named_modules():
        if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            # Unfreeze key projection weights and biases
            if hasattr(module.k_proj, "weight"):
                module.k_proj.weight.requires_grad = True
            if hasattr(module.k_proj, "bias") and module.k_proj.bias is not None:
                module.k_proj.bias.requires_grad = True

            # Unfreeze value projection weights and biases
            if hasattr(module.v_proj, "weight"):
                module.v_proj.weight.requires_grad = True
            if hasattr(module.v_proj, "bias") and module.v_proj.bias is not None:
                module.v_proj.bias.requires_grad = True

    # Print summary of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(
        f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
    )


def get_label_map(data_type, map_type):
    try:
        return LABEL_MAPS[data_type][map_type]
    except KeyError:
        raise NotImplementedError(
            f"Label map for {data_type} and {map_type} not found."
        )


def custom_collator(features, tokenizer):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [
        torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
    ]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_function(examples, tokenizer, data_type="sst2"):
    if data_type == "sst2":
        label_map = get_label_map(data_type, "id2str")

        inputs = [
            f"Sentence: {sentence} Label: {label_map[label]}"
            for sentence, label in zip(examples["sentence"], examples["label"])
        ]
    elif data_type == "sst5":
        label_map = get_label_map(data_type, "id2str")
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
        return_tensors=None,
    )
    # Set labels equal to input_ids for causal LM loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def format_context(demonstrations, data_type="sst2") -> str:
    """
    demonstrations: a Dataset object or None
    convert a demontrations into a prompt
    """
    prompt = ""

    if demonstrations == None or demonstrations == "":
        return prompt

    if data_type == "sst2":
        for example in demonstrations:
            sentence, label = example["sentence"], example["label"]
            prompt += f"Sentence: {sentence} Label: {get_label_map(data_type, 'id2str')[label]} "
    elif data_type == "sst5":
        for example in demonstrations:
            sentence, label = example["text"], example["label"]
            prompt += f"Sentence: {sentence} Label: {get_label_map(data_type, 'id2str')[label]} "
    else:
        raise NotImplementedError

    return prompt


def format_query(context_prompt, sentence, data_type="sst2"):
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


def evaluate_demonstrations(
    model, tokenizer, demonstrations, test_data, data_type="sst2", batch_size=16
):
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
        batch = test_list[start : start + batch_size]
        if data_type == "sst2":
            prompts = [
                format_query(context_prompt, ex["sentence"], data_type=data_type)
                for ex in batch
            ]
        elif data_type == "sst5":
            prompts = [
                format_query(context_prompt, ex["text"], data_type=data_type)
                for ex in batch
            ]
        else:
            raise NotImplementedError

        with torch.inference_mode():
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(to_device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
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


def evaluate_zeroshot(base_model, tokenizer, test_data, data_type="sst2"):
    acc, candidate = evaluate_demonstrations(
        base_model,
        tokenizer,
        demonstrations="",
        test_data=test_data,
        data_type=data_type,
    )
    return acc, candidate


def evaluate_finetuning(fine_tuned_model, tokenizer, test_data, data_type="sst2"):
    acc, candidate = evaluate_demonstrations(
        fine_tuned_model,
        tokenizer,
        demonstrations="",
        test_data=test_data,
        data_type=data_type,
    )
    return acc, candidate


def data_selection(
    model,
    tokenizer,
    train_data,
    test_data,
    data_type="sst2",
    num_data_points: int = 32,
    seed_max: int = 100,
    batch_size: int = 32,
):
    ideal_seed = 0
    max_acc = 0
    ideal_demo = None
    candidate = []
    for seed in tqdm(range(seed_max)):
        random.seed(seed)
        random_indices = random.sample(range(len(train_data)), k=num_data_points)
        demonstrations = train_data.select(random_indices)
        eval_acc, candidate = evaluate_demonstrations(
            model,
            tokenizer,
            demonstrations,
            test_data,
            data_type=data_type,
            batch_size=batch_size,
        )
        if eval_acc > max_acc:
            max_acc = eval_acc
            ideal_seed = seed
            ideal_demo = demonstrations

        print(f"Seed : {seed} --> acc is : {eval_acc}(icl) ")
    return ideal_seed, candidate, ideal_demo


def select_demonstraions(train_data, num_data_points: int = 32, best_seed: int = 0):
    random.seed(best_seed)
    random_indices = random.sample(range(len(train_data)), k=num_data_points)
    demonstrations = train_data.select(random_indices)
    return demonstrations


def finetune_model_eval(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    num_data_points=32,
    data_type: str = "sst2",
    best_seed=0,
):
    random.seed(best_seed)
    random_indices = random.sample(range(len(train_dataset)), k=num_data_points)
    demonstrations = train_dataset.select(
        random_indices
    )  # used as the training data for one epoch

    evaluate_fewshots = evaluate_demonstrations
    _, cnt_zsl = evaluate_zeroshot(model, tokenizer, test_dataset, data_type=data_type)
    _, cnt_icl = evaluate_fewshots(
        model, tokenizer, demonstrations, test_dataset, data_type
    )

    training_args = TrainingArguments(
        output_dir=f"./qwen3_{data_type}_lm",
        eval_strategy="steps",
        eval_steps=64,
        # save_steps=500,
        logging_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        # gradient_accumulation_steps=,
        num_train_epochs=1,
        learning_rate=5e-5,
        # weight_decay=0.01,
        # save_total_limit=1,
        save_strategy="no",
        # fp16=True,
        # push_to_hub=False,
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
        eval_dataset=test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "data_type": data_type},
        ),
        # tokenizer=tokenizer,
        data_collator=lambda features: custom_collator(features, tokenizer),
    )
    # Start training
    trainer.train()

    print("training finished!")

    _, cnt_ft = evaluate_finetuning(model, tokenizer, test_dataset, data_type=data_type)
    metric = len(((set(cnt_ft) - set(cnt_zsl)) & (set(cnt_icl) - set(cnt_zsl)))) / (
        len((set(cnt_ft) - set(cnt_icl))) + 1e-5
    )

    print(f" The Rec2FT is {metric}")
    return metric


def extract_hiddenstates(
    model, tokenizer, test_data: List[str], batch_size=2, return_all=True
):
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
        batch_data = test_data[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(
                model.device
            )
            output = model(
                **inputs, output_hidden_states=True, pad_token_id=tokenizer.pad_token_id
            )

            # Properly detach and clone hidden states to break computation graph
            hidden_states = tuple(
                h.detach().cpu()[:, -1, :].clone() for h in output.hidden_states
            )
            # hidden_states = tuple(h.detach()[:, -1, :].clone() for h in output.hidden_states)
            all_hidden_states.append(hidden_states)

            # print(f"RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

            # Explicitly delete all references
            del inputs, output
            # Delete the hidden_states tuple after appending to list
            del hidden_states

            # Clear caches
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()  # Run garbage collection

    return all_hidden_states


def extract_attn_weights(model, tokenizer, test_data: List[str], batch_size=1):
    attn_collector = AttentionCollector(model)
    attn_collector.enable_collection()

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if type(test_data) == str:
        test_data = [test_data]

    attn_all = [[] for _ in range(model.config.num_hidden_layers)]
    for i in tqdm(range(0, len(test_data), batch_size), leave=True):
        batch_data = test_data[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(
                model.device
            )
            output = model(**inputs, pad_token_id=tokenizer.pad_token_id)

            attn_collector.collect_attn()
            attn_weights = attn_collector.get_all_attn()  # dictionary of layer_idx -> attention tensor(Size(batch_size, num_heads, seq_len, seq_len))
            for layer_idx in range(model.config.num_hidden_layers):
                layer_attention = attn_weights.get(
                    f"layer_{layer_idx}"
                )  # (batch_size, num_heads, seq_len, seq_len)
                if layer_attention is not None:
                    last_token_attention = (
                        layer_attention[:, :, -1, :].detach().cpu()
                    )  # (batch_size, num_heads, seq_len)
                    attn_all[layer_idx].append(last_token_attention)
                else:
                    print(f"Warning: No attention weights found for layer {layer_idx}")

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    attn_collector.disable_collection()

    return attn_all  # list of length num_layers, each element is a list of tensors (num_batches) of shape (batch_size, num_heads, seq_len)


def extract_attentionweights(
    model, tokenizer, test_data: List[str], batch_size=1, before_softmax=True
):
    """
    Extract attention weights/scores from the model - Working version for Qwen
    """
    if type(test_data) == str:
        test_data = [test_data]
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    layerwise_attention = [[] for _ in range(model.config.num_hidden_layers)]

    if before_softmax:
        # Use a different approach: monkey-patch the attention function
        raw_attention_scores = {}
        original_forward_funcs = {}

        def create_hooked_forward(layer_idx, original_forward):
            def hooked_forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                **kwargs,
            ):
                # Get Q, K, V projections
                bsz, q_len, _ = hidden_states.size()

                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                # Reshape for multi-head attention
                query_states = query_states.view(
                    bsz, q_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, self.num_key_value_heads, self.head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, self.num_key_value_heads, self.head_dim
                ).transpose(1, 2)

                # Handle GQA (Grouped Query Attention) if needed
                if self.num_key_value_heads != self.num_heads:
                    key_states = key_states.repeat_interleave(
                        self.num_heads // self.num_key_value_heads, dim=1
                    )
                    value_states = value_states.repeat_interleave(
                        self.num_heads // self.num_key_value_heads, dim=1
                    )

                # Apply rotary embeddings
                cos, sin = self.rotary_emb(value_states, seq_len=q_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

                # Compute raw attention scores
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / (self.head_dim**0.5)

                # Store raw scores BEFORE applying mask and softmax
                raw_attention_scores[layer_idx] = attn_weights.detach().cpu()

                # Apply attention mask if provided
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # Apply softmax
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_weights = torch.nn.functional.dropout(
                    attn_weights, p=self.attention_dropout, training=self.training
                )

                # Apply attention to values
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

                if output_attentions:
                    return attn_output, attn_weights
                else:
                    return attn_output, None

            return hooked_forward

        # Monkey-patch the attention layers
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
                original_forward_funcs[layer_idx] = attn_module.forward
                attn_module.forward = create_hooked_forward(
                    layer_idx, attn_module.forward
                ).__get__(attn_module, attn_module.__class__)
                print(f"Patched layer {layer_idx} attention forward method")

    for i in tqdm(range(0, len(test_data), batch_size), leave=True):
        batch_data = test_data[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(
                model.device
            )

            if before_softmax:
                # Clear previous scores
                raw_attention_scores.clear()

                # Forward pass will trigger our patched attention
                output = model(**inputs, pad_token_id=tokenizer.pad_token_id)

                print(
                    f"Captured raw attention scores for layers: {list(raw_attention_scores.keys())}"
                )

                # Extract raw scores
                for layer_idx in range(model.config.num_hidden_layers):
                    if layer_idx in raw_attention_scores:
                        attention_tensor = raw_attention_scores[layer_idx]
                        # Get last token attention scores: (batch_size, num_heads, seq_len)
                        layer_scores = attention_tensor[:, :, -1, :]
                        layerwise_attention[layer_idx].append(layer_scores)
                        print(
                            f"Layer {layer_idx}: Extracted raw scores with shape: {layer_scores.shape}"
                        )
                    else:
                        print(
                            f"Warning: No raw attention scores captured for layer {layer_idx}"
                        )
            else:
                # Use standard attention weights (after softmax)
                output = model(
                    **inputs,
                    output_attentions=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                attentions = output.attentions

                for layer_idx, layer_attention in enumerate(attentions):
                    last_token_attention = layer_attention[:, :, -1, :].detach().cpu()
                    layerwise_attention[layer_idx].append(last_token_attention)

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    # Restore original forward methods
    if before_softmax:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx in original_forward_funcs:
                layer.self_attn.forward = original_forward_funcs[layer_idx]
        print("Restored original attention forward methods")

    return layerwise_attention


def get_query_states(model, tokenizer, demonstraions, data_type="sst2", query_type = "demonstration") -> Tuple[List[torch.Tensor]]:
    query_states = tuple([] for _ in range(model.config.num_hidden_layers))
    hooks = []

    if query_type == "demonstration":
        text = format_context(demonstraions, data_type=data_type)
    elif query_type == "query_zsl":
        text = demonstraions


    # Hook directly on q_proj modules
    def hook_fn(name, layer_idx):
        def hook(module, input, output):
            query_states[layer_idx].append(output.detach().clone())

        return hook

    # Register hooks on all q_proj layers
    for name, module in model.named_modules():
        if "q_proj" in name:
            hooks.append(module.register_forward_hook(hook_fn(name, len(hooks))))


    if query_type == "demonstration":
        # Forward pass
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs, pad_token_id=tokenizer.pad_token_id)

    elif query_type == "query_zsl":
        batch_size = 4
        for i in tqdm(range(0, len(text), batch_size), leave=True):
            batch_data = demonstraions[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(batch_data, return_tensors="pt", padding=True).to(
                    model.device
                )
                model(**inputs, pad_token_id=tokenizer.pad_token_id)

    # Cleanup
    for hook in hooks:
        hook.remove()


    return query_states
