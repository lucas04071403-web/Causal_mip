import gc
import os
from typing import Literal
from torch.utils.data import RandomSampler
from rmu_layer_utils import get_rmu_layer_module
from utils import (
    data_process_clf_mllmu_batch,
    data_process_gen_mllmu_batch,
    data_process_clf_clear_batch,
    data_process_gen_clear_batch
)
from write_log import write_logger

from transformers import get_scheduler
from tqdm import tqdm
import torch
import datetime

time_today = datetime.datetime.now()
time_today = time_today.strftime('%Y%m%d')


def get_model_device(model):
    return next(model.parameters()).device


def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    model_device = get_model_device(model)
    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            if "Qwen" in model.config.architectures[0]:
                input_ids = inputs[0].to(model_device)
                attention_mask = inputs[1].to(model_device)
                pixel_values = inputs[2].to(model_device)
                image_grid_thw = inputs[3].to(model_device)
                labels = inputs[4].to(model_device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)
            elif "Llava" in model.config.architectures[0]:
                input_ids = inputs[0].to(model_device)
                attention_mask = inputs[1].to(model_device)
                pixel_values = inputs[2].to(model_device)
                labels = inputs[3].to(model_device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
            elif "Gemma" in model.config.architectures[0]:
                inputs = inputs[0]
                input_ids = inputs["input_ids"].to(model_device)
                attention_mask = inputs["attention_mask"].to(model_device)
                pixel_values = inputs["pixel_values"].to(model_device)
                labels = inputs["labels"].to(model_device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    else:
        if "Qwen" in model.config.architectures[0]:
            input_ids = inputs[0].to(model_device)
            attention_mask = inputs[1].to(model_device)
            pixel_values = inputs[2].to(model_device)
            image_grid_thw = inputs[3].to(model_device)
            labels = inputs[4].to(model_device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)
        elif "Llava" in model.config.architectures[0]:
            input_ids = inputs[0].to(model_device)
            attention_mask = inputs[1].to(model_device)
            pixel_values = inputs[2].to(model_device)
            labels = inputs[3].to(model_device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
        elif "Gemma" in model.config.architectures[0]:
            inputs = inputs[0]
            input_ids = inputs["input_ids"].to(model_device)
            attention_mask = inputs["attention_mask"].to(model_device)
            pixel_values = inputs["pixel_values"].to(model_device)
            labels = inputs["labels"].to(model_device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    hook_handle.remove()

    return cache[0]

def precompute_rmu_retain_activations(model, loader, layer_id):
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, RandomSampler):
        raise ValueError(
            "RMU retain activation caching requires a deterministic retain_loader order. "
            "Please disable shuffle/random sampling for retain_loader."
        )
    target_module = get_rmu_layer_module(model, layer_id, "frozen")
    cached_activations = []
    was_training = model.training
    model.eval()
    for batch in tqdm(loader, total=len(loader), desc="Caching frozen retain activations"):
        activations = forward_with_cache(model, batch, module=target_module, no_grad=True)
        cached_activations.append(activations.detach().cpu())
    if len(cached_activations) != len(loader):
        raise ValueError(
            f"Cached retain activations count ({len(cached_activations)}) does not match "
            f"retain_loader length ({len(loader)})."
        )
    if was_training:
        model.train()
    return cached_activations

def train(model, data_loader, optimizer, args, save=True, save_identifier="", skip_train_if_exists=False):
    _save_identifier = f"_{save_identifier}" if save_identifier else ""
    save_path = f"{args.output_file_path}model_caches/{args.model}{_save_identifier}_{args.dataset[:5]}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}.pth"
    if skip_train_if_exists and os.path.exists(save_path):
        from load_model import load_peft_model
        model = load_peft_model(args, trainable=True, identifier=save_identifier)
        print(f"Model already exists at {save_path}, skipping training.")
        model.to(args.device)
        return model

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(data_loader),
    )
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(data_loader, desc=f"Training {epoch + 1}/{args.epochs}")):
            optimizer.zero_grad()
            if "Qwen" in args.model or "llava" in args.model:
                if "Qwen" in args.model:
                    input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch
                if "llava" in args.model:
                    input_ids, attention_mask, pixel_values, labels, _ = batch
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels
                        )
            else:
                inputs, _ = batch
                for k in inputs.keys():
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(args.device)
                output = model(
                    **inputs
                )
            loss = output.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            if idx % 50 == 0:
                tqdm.write(f"Batch {idx + 1}/{len(data_loader)} finished, loss: {loss.item():.4f}")
            # tqdm.write(f"Batch {idx + 1}/{len(data_loader)} finished, loss: {loss.item():.4f}")
        epoch_loss /= len(data_loader)
        print(f"Epoch {epoch + 1} finished, loss: {epoch_loss:.4f}")
    # save lora model
    if save:
        model.save_pretrained(save_path)
    return model


def finetune(model, retain_loader, sampled_forget_loader, args, save=True, use_forget=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.finetune_epochs * len(retain_loader),
    )
    if use_forget:
        u = torch.randn(model.config.hidden_size).to(args.device)
        u = u / torch.norm(u)
        c = 2
        alpha = 0.001
    model.train()
    all_loss  = []
    for epoch in range(args.finetune_epochs):
        epoch_loss = 0
        for idx, (batch_retain, batch_forget) in tqdm(enumerate(zip(retain_loader, sampled_forget_loader)), postfix={"epoch": epoch}, total=len(retain_loader), dynamic_ncols=True):
            if "Qwen" in args.model or "llava" in args.model:
                if "Qwen" in args.model:
                    input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch_retain
                if "llava" in args.model:
                    input_ids, attention_mask, pixel_values, labels, _ = batch_retain
                optimizer.zero_grad()
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output_retain = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=True
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )                
                        output_retain = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels,
                            output_hidden_states=True
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output_retain = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=True
                        )
            else:
                inputs, _ = batch_retain
                for k in inputs.keys():
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(args.device)
                output_retain = model(
                    **inputs
                )
            loss_retain = output_retain.loss

            if use_forget:
                input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch_forget
                optimizer.zero_grad()
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output_forget = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=True
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )
                        output_forget = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels,
                            output_hidden_states=True
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output_forget = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=True
                        )
                h_forget = output_forget.hidden_states[7]
                loss_unlearn = torch.mean(torch.norm(h_forget - c * u, dim=1)**2)

                loss = loss_retain + alpha * loss_unlearn
            else:
                loss = loss_retain
                loss_unlearn = torch.tensor(0.0, device=args.device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            all_loss.append(loss.item())
            if idx % 50 == 0:
                tqdm.write(f"Retain Set Batch {idx + 1}/{len(retain_loader)} finished, loss: {loss.item():.4f}, loss_retain: {loss_retain.item():.4f}, loss_unlearn: {loss_unlearn.item():.4f}")
    # save lora model
    if save:
        model.save_pretrained(f"{args.output_file_path}model_caches/{args.model}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}_finetune.pth")

    import json
    with open(f'{args.output_file_path}/loss_{args.this_run_id}.json', 'w') as f:
        json.dump(all_loss, f)
    print('saved')


    return model


def adaptive_rmu_finetune(updated_model, frozen_model, retain_loader, forget_loader, args, save=True):
    def get_params(model, layer_ids, param_ids):
        params = []
        for layer_id in layer_ids:
            for i, p in enumerate(model.model.layers[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)
        return params
    layer_id = args.rmu_layer_id
    steering_coeff = args.rmu_steering_coeff
    alpha = args.rmu_alpha
    beta = args.rmu_beta
    optimizer = torch.optim.AdamW(updated_model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                                 num_training_steps=args.finetune_epochs * len(retain_loader))
    frozen_retain_cache = frozen_model if isinstance(frozen_model, list) else None
    if frozen_model is None:
        frozen_retain_cache = precompute_rmu_retain_activations(updated_model, retain_loader, layer_id)
    elif frozen_retain_cache is not None and len(frozen_retain_cache) != len(retain_loader):
        raise ValueError(
            f"Frozen retain cache length ({len(frozen_retain_cache)}) does not match "
            f"retain_loader length ({len(retain_loader)})."
        )

    if frozen_model is not None and not isinstance(frozen_model, list):
        frozen_module = get_rmu_layer_module(frozen_model, layer_id, "frozen")
    updated_module = get_rmu_layer_module(updated_model, layer_id, "updated")
    
    if hasattr(updated_model.config, 'hidden_size'):
        hidden_size = updated_model.config.hidden_size
    elif hasattr(updated_model.config, 'text_config') and hasattr(updated_model.config.text_config, 'hidden_size'):
        hidden_size = updated_model.config.text_config.hidden_size
    else:
        raise ValueError("Model configuration does not contain 'hidden_size' or 'text_config.hidden_size'.")
    
    random_vector = torch.rand(1, 1, hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * steering_coeff

    all_loss = []

    updated_model.train()
    for epoch in range(args.finetune_epochs):
        coeffs = args.rmu_coeffs
        for idx, (batch_r, batch_f) in tqdm(enumerate(zip(retain_loader, forget_loader)),
                                            total=len(retain_loader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            updated_forget_activations = forward_with_cache(
                    updated_model, batch_f, module=updated_module, no_grad=False
            ).to(updated_model.device)
            # if idx == 0:
            #     coeffs = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item() * 5.0 
            # else:
            #     pass
            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_activations, control_vec * coeffs
            )
            updated_retain_activations = forward_with_cache(
                updated_model, batch_r, module=updated_module, no_grad=False
            ).to(updated_model.device)
            if frozen_retain_cache is not None:
                frozen_retain_activations = frozen_retain_cache[idx].to(updated_model.device)
            else:
                frozen_retain_activations = forward_with_cache(
                    frozen_model, batch_r, module=frozen_module, no_grad=True
                ).to(updated_model.device)

            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_activations, frozen_retain_activations
            )

            # Update model
            loss = alpha * retain_loss + beta * unlearn_loss
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            all_loss.append(loss.item())
            if idx % 10 == 0:
                tqdm.write(f"[{epoch}/{args.finetune_epochs}] Step {idx} | loss: {loss.item():.4f} "
                           f"| retain: {retain_loss.item():.4f} | unlearn: {unlearn_loss.item():.4f}")

    if save:
        updated_model.save_pretrained(f"{args.output_file_path}model_caches/"
                              f"{args.model}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}_finetune.pth")
    
    import json
    with open(f'{args.output_file_path}/loss_{args.this_run_id}.json', 'w') as f:
        json.dump(all_loss, f)
    print('saved')

    return updated_model

def evaluate_clf(dataset, 
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'],
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "mllmu" in args.dataset:
            preds = data_process_clf_mllmu_batch(dataset, processor, model, args, modality='multi', data_type="train")
        elif "clear" in args.dataset:
            preds = data_process_clf_clear_batch(dataset, processor, model, args)
    elif data_modality == 'text':
        if "mllmu" in args.dataset:
            preds = data_process_clf_mllmu_batch(dataset, processor, model, args, modality='text', data_type="train")
        elif "clear" in args.dataset:
            assert False
    else:
        assert False

    model.cpu()
    preds, acc_by_llm = score_batch(preds, 'clf', args)
    model.to(args.device)

    msg = f"{dataset_split}set Finished.\tAccuracy by LLM: {acc_by_llm:.2%}"
    print(msg)
    # save the results
    write_logger(args, msg, args.output_file_path + "logs/")

    os.makedirs(args.output_file_path + f"logs/{args.this_run_id}", exist_ok=True)
    now = datetime.datetime.now().strftime('%d%H%M%S')
    pred_save_path = args.output_file_path + f"logs/{args.this_run_id}/clf_{dataset_split}set_{data_modality}_preds_{now}.json"
    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            # "acc": accuracy,
            "acc_by_llm": acc_by_llm,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 


def evaluate_gen(forgetset, 
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'], 
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "mllmu" in args.dataset:
            preds = data_process_gen_mllmu_batch(forgetset, processor, model, args, modality='multi', data_type="train")
        if "clear" in args.dataset:
            preds = data_process_gen_clear_batch(forgetset, processor, model, args, modality='multi')
    elif data_modality == 'text':
        if "mllmu" in args.dataset:
            preds = data_process_gen_mllmu_batch(forgetset, processor, model, args, modality='text', data_type="train")
        if "clear" in args.dataset:
            preds = data_process_gen_clear_batch(forgetset, processor, model, args, modality='text')
    else:
        assert False
    
    if len(preds) == 0:
        print(f"Warning: No samples found for {dataset_split} set with modality {data_modality}. Skipping evaluation.")
        return
    
    # calculate rouge and bleu
    from metrics.bleu.bleu import Bleu
    from metrics.rouge.rouge import Rouge
    bleu = Bleu()
    rouge = Rouge()
    try:
        bleu_scores = bleu.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    except ZeroDivisionError:
        bleu_scores = {'bleu': 0}
    rouge_scores = rouge.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    bleumean = bleu_scores['bleu']
    rouge1mean = rouge_scores['rouge1']
    rouge2mean = rouge_scores['rouge2']
    rougeLmean = rouge_scores['rougeL']
    rougeLsummean = rouge_scores['rougeLsum']

    model.cpu()

    preds, acc = score_batch(preds, 'gen', args)
    model.to(args.device)

    msg = f"{dataset_split}set Finished. acc: {acc:.2%}, Rouge1: {rouge1mean:.2%}, Rouge2: {rouge2mean:.2%}, RougeL: {rougeLmean:.2%}, RougeLsum: {rougeLsummean:.2%}, Bleu: {bleumean:.2%}"
    print(msg)
    # save the results
    write_logger(args, msg, args.output_file_path + "logs/")
    now = datetime.datetime.now().strftime('%d%H%M%S')
    
    os.makedirs(args.output_file_path + f"logs/{args.this_run_id}", exist_ok=True)
    pred_save_path = args.output_file_path + f"logs/{args.this_run_id}/gen_{dataset_split}set_{data_modality}_preds_{now}.json"
    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            "acc": acc,
            "bleu": bleumean,
            "rouge1": rouge1mean,
            "rouge2": rouge2mean,
            "rougeL": rougeLmean,
            "rougeLsum": rougeLsummean,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 


def score_batch(preds, task, args):
    batch_size = 16

    gc.collect()
    torch.cuda.empty_cache()
    score_llm = None

    # 检查是否使用远程 API
    use_remote = getattr(args, 'use_remote_scoring', False)
    remote_url = getattr(args, 'remote_scoring_url', None)
    remote_model = getattr(args, 'score_llm', 'Qwen2.5-VL-7B-Instruct')

    try:
        if use_remote and remote_url:
            # 使用远程 API 评分
            from score_by_llm import load_remote_llm, score_by_remote_llm_batch
            print(f'远程评分模式: {remote_url}')
            remote_client = load_remote_llm(
                base_url=remote_url,
                model_name=remote_model
            )
            preds = score_by_remote_llm_batch(remote_client, preds, batch_size, task)
        else:
            # 使用本地模型评分
            from score_by_llm import load_llm as local_load_llm, score_by_llm_batch as local_score_batch
            score_llm, tokenizer = local_load_llm(f"{args.llm_directory}{args.score_llm}", args.device)
            print(f'本地评分模式: {args.llm_directory}{args.score_llm}')
            preds = local_score_batch(score_llm, tokenizer, preds, batch_size, task, args.device)

        # 计算准确率
        valid_preds = [p for p in preds if p.get('correct') is not None]
        if valid_preds:
            acc = sum(map(lambda r: int(r['correct']), valid_preds)) / len(valid_preds)
        else:
            acc = 0.0

        return preds, acc
    except Exception as e:
        print(f"Error during scoring: {type(e)}  {e}")
        if score_llm is not None:
            del score_llm
        torch.cuda.empty_cache()
        gc.collect()
        return preds, -1
