import gc
import time
import torch
import numpy as np
import random
from causal_mip.project_paths import workspace_path, workspace_path_with_sep
from causal_mip.path_localization import export_candidate_paths_from_cached_scores

import argparse

weight_prune = False


# Configurations
seed = 42
epochs = 1
finetune_epochs = 1
batch_size = 6
learning_rate = 5e-4
noise_scale = 1000
image_resize = 224
# forget_ratio = 1
forget_ratio = 5
topk = 5
rmu_layer_id = 7 
rmu_steering_coeff = 1.0 
rmu_alpha = 0.5
rmu_beta = 1 - rmu_alpha
rmu_coeffs = 10.0

num_fisher_integrate_step = 4
ig_total_step = 4
train_flag = False
eval_flag = False
use_neuron_cache_flag = True
device = "cuda"
dataset = "clear"
unlearning = "our"


model = "Qwen2.5-VL-3B-Instruct"
base_path = workspace_path_with_sep("datasets")
llm_directory = workspace_path_with_sep("llms")
output_file_path = workspace_path_with_sep("outputs")
if "mllmu" in dataset:
    train_path = workspace_path("datasets", "MLLMU-Bench", "ft_Data", "train-00000-of-00001.parquet")
    fullset_path = workspace_path("datasets", "MLLMU-Bench", "Full_Set", "train-00000-of-00001.parquet")
    test_path = workspace_path("datasets", "MLLMU-Bench", "Test_Set", "test-00000-of-00001.parquet")
if "clear" in dataset:
    train_path = workspace_path_with_sep("datasets", "CLEAR", "full+tofu")
    fullset_path = workspace_path_with_sep("datasets", "CLEAR", "full")
    test_path = workspace_path("datasets", "CLEAR", "test.jsonl")
path_path = workspace_path_with_sep("influential_paths")
score_llm = "Qwen2-7B-Instruct"



# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--finetune_epochs", type=int, default=finetune_epochs)
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--learning_rate", type=float, default=learning_rate)
parser.add_argument("--train_flag", action='store_true', default=train_flag)
parser.add_argument("--eval_flag", action='store_true', default=eval_flag)
parser.add_argument("--use_neuron_cache_flag", action='store_true', default=use_neuron_cache_flag)
parser.add_argument("--unlearning", type=str, default=unlearning)
parser.add_argument("--noise_scale", type=float, default=noise_scale)
parser.add_argument("--device", type=str, default=device)
parser.add_argument("--dataset", type=str, default=dataset)
parser.add_argument("--image_resize", type=int, default=image_resize)
parser.add_argument("--topk", type=int, default=topk)
parser.add_argument("--ig_total_step", type=int, default=ig_total_step)
parser.add_argument("--num_fisher_integrate_step", type=int, default=num_fisher_integrate_step)
parser.add_argument("--forget_ratio", type=int, default=forget_ratio)
parser.add_argument("--seed", type=int, default=seed)
parser.add_argument("--model", type=str, default=model)
parser.add_argument("--llm_directory", type=str, default=llm_directory)
parser.add_argument("--base_path", type=str, default=base_path)
parser.add_argument("--path_path", type=str, default=path_path)
parser.add_argument("--train_path", type=str, default=train_path)
parser.add_argument("--fullset_path", type=str, default=fullset_path)
parser.add_argument("--test_path", type=str, default=test_path)
parser.add_argument("--output_file_path", type=str, default=output_file_path)
parser.add_argument("--score_llm", type=str, default=score_llm)
parser.add_argument("--rmu_layer_id", type=int, default=rmu_layer_id)
parser.add_argument("--rmu_steering_coeff", type=float, default=rmu_steering_coeff)
parser.add_argument("--rmu_alpha", type=float, default=rmu_alpha)
parser.add_argument("--rmu_beta", type=float, default=rmu_beta)
parser.add_argument("--rmu_coeffs", type=float, default=rmu_coeffs)
parser.add_argument("--use_masked_rmisu", action='store_true', default=False)
parser.add_argument(
    "--masked_rmisu_candidate_paths",
    type=str,
    default=workspace_path("outputs", "paths", "P_cand.jsonl"),
)
parser.add_argument(
    "--masked_rmisu_p_forget",
    type=str,
    default=workspace_path("outputs", "paths", "step6_v1", "P_forget.jsonl"),
)
parser.add_argument(
    "--masked_rmisu_p_shared",
    type=str,
    default=workspace_path("outputs", "paths", "step6_v1", "P_shared.jsonl"),
)
parser.add_argument("--masked_rmisu_p_probe", type=str, default=None)
parser.add_argument("--masked_rmisu_shared_alpha", type=float, default=1.0)
parser.add_argument("--masked_rmisu_probe_beta", type=float, default=0.05)
parser.add_argument("--masked_rmisu_probe_steering_coeff", type=float, default=1.0)
parser.add_argument("--masked_rmisu_probe_coeffs", type=float, default=1.0)
parser.add_argument(
    "--masked_rmisu_forget_objective",
    type=str,
    default="activation_random",
    choices=[
        "activation_random",
        "ce_ascent",
        "answer_ce_ascent",
        "name_ce_ascent",
        "activation_random_ce",
        "activation_random_answer_ce",
        "activation_random_name_ce",
    ],
)
parser.add_argument("--masked_rmisu_forget_ce_alpha", type=float, default=0.0)
parser.add_argument(
    "--masked_rmisu_target_ce_scope",
    type=str,
    default="all",
    choices=["all", "answer", "name"],
)
parser.add_argument(
    "--masked_rmisu_projector_edit_mode",
    type=str,
    default="qwen_merger_mlp",
    choices=["qwen_merger_mlp", "skip"],
)
parser.add_argument("--masked_rmisu_output", type=str, default=None)
parser.add_argument("--masked_rmisu_checkpoint_dir", type=str, default=None)
parser.add_argument("--masked_rmisu_max_steps", type=int, default=None)
parser.add_argument("--skip_post_unlearning_eval", action='store_true', default=False)

parser.add_argument("--ptm_ckpt_batch_size", type=int, default=16)
parser.add_argument("--weight_prune", type=bool, default=weight_prune)


parser.add_argument("--baseline_forget_training_epochs", type=int, default=1)
parser.add_argument("--baseline_npo_beta", type=float, default=0.4)
parser.add_argument("--baseline_npo_epoch", type=int, default=1)

# 远程评分配置
parser.add_argument("--use_remote_scoring", action='store_true', default=False,
                    help="是否使用远程 API 进行评分")
parser.add_argument("--remote_scoring_url", type=str, default="http://localhost:9192/v1",
                    help="远程评分 API 地址，如 http://your-server:9192/v1")

parser.add_argument("--this_run_id", type=str, default=time.strftime("%m%d_%H%M%S"))
parser.add_argument("--this_run_description", type=str, default="")
parser.add_argument("--export_candidate_paths", action='store_true', default=False)
parser.add_argument("--export_candidate_paths_only", action='store_true', default=False)
parser.add_argument("--candidate_num_paths", type=int, default=10)
parser.add_argument("--candidate_per_layer_topk", type=int, default=5)
parser.add_argument("--candidate_cross_modal_paths", type=int, default=20)
parser.add_argument(
    "--candidate_paths_output",
    type=str,
    default=workspace_path("outputs", "paths", "P_cand.jsonl"),
)

args = parser.parse_args()



def setup_seed(seed):
    from transformers import set_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(args.seed)

def main():
    if args.export_candidate_paths or args.export_candidate_paths_only:
        summary = export_candidate_paths_from_cached_scores(args)
        print(f"Exported candidate paths to {summary['output_path']}")
        print(
            f"text={summary['text_paths']}, vision={summary['vision_paths']}, "
            f"cross_modal={summary['cross_modal_paths']}, total={summary['all_paths']}"
        )
        if args.export_candidate_paths_only:
            return

    from load_model import load_base_model, load_model, load_peft_model
    from load_data import load_data_train,  load_data_retain, load_data_forget, load_data_finetune
    from train_eval import train, evaluate_clf, evaluate_gen
    from unlearning import ga_difference_training
    from ours import our
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import AutoProcessor

    setup_seed(args.seed)

    model_path = f"{args.llm_directory}{args.model}"
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    # forgetset, forget_loader, retainset, retain_loader = load_random_dataset(processor, args)
    forgetset_clf_eval, forgetset_gen_eval, forget_indices_eval = load_data_forget(processor, args)
    retainset_clf_eval, retainset_gen_eval, retain_indices_eval = load_data_retain(processor, args)

    forgetset_train, forgetloader, retainset_train, retainloader, forget_text_loader, retain_text_loader, forget_indices, retain_indices, collate = load_data_finetune(processor, forget_indices_eval, retain_indices_eval, args)
    if args.train_flag:
        model = load_model(args, visual_trainable=False)
        processor.tokenizer.padding_side = 'right' 
        trainset, train_loader = load_data_train(processor, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        model = train(model, train_loader, optimizer, args)
        processor.tokenizer.padding_side = 'left'  # Reset padding side after training

    batch_size = args.batch_size
    args.batch_size = args.ptm_ckpt_batch_size
    model = load_peft_model(args, trainable=True)
    args.batch_size = batch_size

    if args.eval_flag:
        print("Evaluating before forgetting")
        with torch.no_grad():
            evaluate_clf(forgetset_clf_eval, processor, 'forget', 'multi', args, model)
            evaluate_clf(retainset_clf_eval, processor, 'retain', 'multi', args, model)
            evaluate_gen(forgetset_gen_eval, processor, 'forget', 'multi', args, model)
            evaluate_gen(retainset_gen_eval, processor, 'retain', 'multi', args, model)
            if 'mllmu' in args.dataset: # clear doesn't got text clf task
                evaluate_clf(forgetset_clf_eval, processor, 'forget', 'text', args, model)
                evaluate_clf(retainset_clf_eval, processor, 'retain', 'text', args, model)
            evaluate_gen(forgetset_gen_eval, processor, 'forget', 'text', args, model)
            evaluate_gen(retainset_gen_eval, processor, 'retain', 'text', args, model)
    if args.unlearning is None or args.unlearning == '':
        return

    if args.unlearning == "ga":
        model = ga_difference_training(forgetloader, retainloader, model, args)
    elif args.unlearning == "kl":
        from unlearning import kl_min
        vanilla_model = load_base_model(args).eval()
        model = kl_min(forgetloader, retainloader, model, vanilla_model, args)
    elif args.unlearning == "npo":
        from unlearning import npo
        ep = args.epochs
        model.cpu()
        args.epochs = args.baseline_npo_epoch
        batch_size = args.batch_size
        args.batch_size = 4
        model_finetune_on_retainset = load_model(args)
        model_finetune_on_retainset = train(model_finetune_on_retainset, 
                                            retainloader, 
                                            torch.optim.Adam(model_finetune_on_retainset.parameters(), lr=args.learning_rate), 
                                            args,
                                            save=True,
                                            save_identifier=f"trainOnRetainSet{args.forget_ratio}",
                                            skip_train_if_exists=True)
        model.to(args.device)
        args.batch_size = batch_size
        args.epochs = ep
        gc.collect()
        torch.cuda.empty_cache()
        model = npo(forgetloader, retainloader, model, model_finetune_on_retainset, args)
    elif args.unlearning == "our":
        model = model.merge_and_unload()
        model.requires_grad_(True)
        sampler = WeightedRandomSampler(weights=[1.0] * len(forget_indices), num_samples=len(retain_indices), replacement=True)
        sampled_forget_loader = DataLoader(forgetset_train, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
        model = our(model, forgetloader, forget_text_loader, forget_indices, retainloader, retain_text_loader, retain_indices, sampled_forget_loader, args)
    elif args.unlearning == "manu":
        model = model.merge_and_unload()
        from manu import MANUPruner
        pruner = MANUPruner(model, args.device)
        forget_score = pruner.compute_importance(model, forgetloader)
        gc.collect()
        torch.cuda.empty_cache()
        retain_score = pruner.compute_importance(model, retainloader)
        gc.collect()
        torch.cuda.empty_cache()
        model, cancel = pruner.prune(model, forget_score, retain_score)

    else:
        assert False, f"Unknown unlearning method: {args.unlearning}"

    if args.skip_post_unlearning_eval:
        if args.use_masked_rmisu and args.masked_rmisu_checkpoint_dir:
            processor.save_pretrained(args.masked_rmisu_checkpoint_dir)
        print("Skipping post-unlearning evaluation (--skip_post_unlearning_eval).")
        return

    with torch.no_grad():
        with torch.no_grad():
            evaluate_clf(forgetset_clf_eval, processor, 'forget', 'multi', args, model)
            evaluate_clf(retainset_clf_eval, processor, 'retain', 'multi', args, model)
            evaluate_gen(forgetset_gen_eval, processor, 'forget', 'multi', args, model)
            evaluate_gen(retainset_gen_eval, processor, 'retain', 'multi', args, model)
            if 'mllmu' in args.dataset: # clear doesn't got text clf task
                evaluate_clf(forgetset_clf_eval, processor, 'forget', 'text', args, model)
                evaluate_clf(retainset_clf_eval, processor, 'retain', 'text', args, model)
            evaluate_gen(forgetset_gen_eval, processor, 'forget', 'text', args, model)
            evaluate_gen(retainset_gen_eval, processor, 'retain', 'text', args, model)


if __name__ == "__main__":
    main()
