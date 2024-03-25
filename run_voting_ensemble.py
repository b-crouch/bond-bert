import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from tqdm import tqdm
import evaluate
from dataset_utils import *
from model_utils import *
from ensemble_utils import *
from collections import deque
import pickle
import argparse
import sys
import os
from transformers import logging
import warnings

logging.set_verbosity_error()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Ensemble of BERTs for token classification')
parser.add_argument('--train_filepath', type=str, help="Filepath to JSON containing training data")
parser.add_argument('--val_filepath', type=str, help="Filepath to JSON containing validation data")
parser.add_argument('--save_filepath', type=str, help="Filepath to save results")
parser.add_argument('--save_dir', type=str, default=".", help="Directory to save results")
parser.add_argument('--id2label', default="id2label.pkl", type=str, help="Filepath to id2label relation")
parser.add_argument('--label2id', default="label2id.pkl", type=str, help="Filepath to label2id relation")
parser.add_argument('--models', nargs="+", type=str, help="Models to use as sublearners in ensemble")
parser.add_argument('--batch_size', default=16, type=int, help="Batch size for DataLoaders")
parser.add_argument('--mode', type=str, default="all", choices=["majority_vote", "boosted", "joint_density", "all"], help="Ensemble mode: majority voting, joint density maximization, boosted")
args = parser.parse_args(sys.argv[1:])

train_filepath, val_filepath, save_filepath = args.train_filepath, args.val_filepath, args.save_filepath
models_to_use = args.models
batch_size = args.batch_size
mode = args.mode
save_dir = args.save_dir
valid_models = ["matbert", "biobert", "scibert", "chembert", "batterybert"]

assert all([model in valid_models for model in models_to_use]), "Unsupported model. Allowed sublearners are 'matbert', 'biobert', 'scibert', 'chembert' and 'batterybert'"

print(f"Running ensemble with sublearners {models_to_use}")

assert batch_size > 0, "Can't have empty batch!"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device {device}")

with open(args.label2id, "rb") as handle:
    label2id = pickle.load(handle)

with open(args.id2label, "rb") as handle:
    id2label = pickle.load(handle)

num_models = len(models_to_use)
models = {}

for sublearner in models_to_use:
    if sublearner == "matbert":
        base_model = "matbert-base-cased"
        tuned_model = "matbert_tuned/checkpoint_epoch_2.pt"
    elif sublearner == "biobert":
        base_model = "dmis-lab/biobert-base-cased-v1.2"
        tuned_model = "biobert_tuned/checkpoint_epoch_2.pt"
    elif sublearner == "chembert":
        base_model = "recobo/chemical-bert-uncased"
        tuned_model = "chembert_tuned/checkpoint_epoch_2.pt"
    elif sublearner == "scibert":
        base_model = "allenai/scibert_scivocab_cased"
        tuned_model = "scibert_tuned/checkpoint_epoch_2.pt"
    elif sublearner == "batterybert":
        base_model = "batterydata/batterybert-cased"
        tuned_model = "batterybert_tuned/checkpoint_epoch_2.pt"
        
    tokenizer = BertTokenizerFast.from_pretrained(base_model, do_lower_case=False)
    model = BertForTokenClassification.from_pretrained(base_model, id2label=id2label, label2id=label2id, output_hidden_states=True)
    model.load_state_dict(torch.load(tuned_model, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    
    dataset = tokenize_dataset(train_filepath, val_filepath, tokenizer)
    train_dataloader, val_dataloader = prepare_dataloader(dataset.rename_column("ner_tags", "labels"), tokenizer, batch_size, shuffle=False)
    model_attributes = {"name":sublearner, "model":model, "tokenizer":tokenizer, "train_dataloader":iter(train_dataloader), "val_dataloader":iter(val_dataloader)}
    
    if mode == "boosted" or mode == "all":
        model_attributes["error"] = evaluate.load("seqeval", zero_division=0)
    models[sublearner] = model_attributes

if mode == "boosted" or mode == "all":
    print("Begin boosting")
    dataloader_length = len(train_dataloader)
    with tqdm(total=dataloader_length+1) as boost_progress_bar:
        for batch_idx in range(dataloader_length):
            aligned = align_all(models, is_train=True, device=device)
            while aligned:
                sublearner = aligned.pop()
                preds = torch.argmax(torch.softmax(sublearner["logits"], dim=-1), dim=-1)
                true, predicted = postprocess(preds, sublearner["labels"], id2label)
                models[sublearner["name"]]["error"].add_batch(predictions=predicted, references=true)
            boost_progress_bar.update(1)
        for m in models:
            sublearner = models[m]
            results = sublearner["error"].compute()
            sublearner["alpha"] = get_alpha_by_label(results, id2label)
        boost_progress_bar.update(1)


print("Begin evaluation")

if mode == "majority_vote" or mode == "all":
    majority_vote_metric = evaluate.load("seqeval", zero_division=1)  
if mode == "joint_density" or mode == "all":
    joint_metric = evaluate.load("seqeval", zero_division=1)  
if mode == "boosted" or mode == "all":
    boosted_metric = evaluate.load("seqeval", zero_division=1)  
    
with tqdm(total=len(val_dataloader)) as progress_bar:
    for batch_idx in range(len(val_dataloader)):
        aligned = align_all(models, is_train=False, device=device)
        all_predictions, all_logits, all_alphas = [], [], []
        while aligned:
            sublearner = aligned.pop()
            all_logits.append(sublearner["logits"])
            all_predictions.append(torch.argmax(torch.softmax(sublearner["logits"], dim=-1), dim=-1))
            all_alphas.append(models[sublearner["name"]]["alpha"])
        
        aligned_labels = sublearner["labels"]
        aligned_logits_shape = sublearner["logits"].shape
        
        if mode == "majority_vote" or mode == "all":
            ensemble_votes = torch.mode(torch.stack(all_predictions, dim=-1), dim=-1).values
            majority_vote_true, majority_vote_predicted = postprocess(ensemble_votes, aligned_labels, id2label)
            majority_vote_metric.add_batch(predictions=majority_vote_predicted, references=majority_vote_true)
        
        if mode == "joint_density" or mode == "all":
            joint_prob = 1
            for logit in all_logits:
                joint_prob *= torch.softmax(logit, dim=-1)
            joint_argmax = torch.argmax(joint_prob, dim=-1)
            joint_true, joint_preds = postprocess(joint_argmax, aligned_labels, id2label)
            joint_metric.add_batch(predictions=joint_preds, references=joint_true)

        if mode == "boosted" or mode == "all":
            predictions_by_model = torch.stack(all_predictions, dim=-1)
            alpha_by_model = torch.stack(all_alphas, dim=-1)

            boosted_scores = get_boosted_scores(predictions_by_model, alpha_by_model, id2label, aligned_logits_shape)
            boosted_argmax = torch.argmax(boosted_scores, dim=-1)
                        
            boosted_true, boosted_preds = postprocess(boosted_argmax, aligned_labels, id2label)
            boosted_metric.add_batch(predictions=boosted_preds, references=boosted_true)
        
        progress_bar.update(1)

if save_dir != "." and not os.path.exists(save_dir):
    os.makedirs(save_dir)

if mode == "majority_vote":
    results = majority_vote_metric.compute()
    print(results)
    with open(f'{save_dir}/{save_filepath}.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    f = open(f"{save_dir}/{save_filepath}_f1.txt", "w")
    f.write(str(results["overall_f1"]))
    f.close()
elif mode == "joint_density":
    results = joint_metric.compute()
    print(results)
    with open(f'{save_dir}/{save_filepath}.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    f = open(f"{save_dir}/{save_filepath}_f1.txt", "w")
    f.write(str(results["overall_f1"]))
    f.close()
elif mode == "boosted":
    results = boosted_metric.compute()
    print(results)
    with open(f'{save_dir}/{save_filepath}.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    f = open(f"{save_dir}/{save_filepath}_f1.txt", "w")
    f.write(str(results["overall_f1"]))
    f.close()
else:
    majority_vote_results = majority_vote_metric.compute()
    print("Majority vote results:")
    print(majority_vote_results)
    with open(f'{save_dir}/{save_filepath}_majority_vote.pkl', 'wb') as handle:
        pickle.dump(majority_vote_results, handle)
    f = open(f"{save_dir}/{save_filepath}_majority_vote_f1.txt", "w")
    f.write(str(majority_vote_results["overall_f1"]))
    f.close()

    joint_results = joint_metric.compute()
    print("Likelihood maximization results:")
    print(joint_results)
    with open(f'{save_dir}/{save_filepath}_joint_density.pkl', 'wb') as handle:
        pickle.dump(joint_results, handle)
    f = open(f"{save_dir}/{save_filepath}_joint_density_f1.txt", "w")
    f.write(str(joint_results["overall_f1"]))
    f.close()

    boosted_results = boosted_metric.compute()
    print("Boosted results:")
    print(boosted_results)
    with open(f'{save_dir}/{save_filepath}_boosted.pkl', 'wb') as handle:
        pickle.dump(boosted_results, handle)
    f = open(f"{save_dir}/{save_filepath}_boosted_f1.txt", "w")
    f.write(str(boosted_results["overall_f1"]))
    f.close()
    
    