import numpy as np
import torch
from dataset_utils import *
from model_utils import *
import spacy_alignments as tokenizations
from collections import deque
import re
import math

def get_tokens(tokenizer, batch):
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    input_ids = batch["input_ids"]
    if torch.is_tensor(input_ids):
        input_ids = input_ids.clone().cpu()
    return np.vectorize(id_to_token.get)(input_ids.numpy())

def align_outputs(tokens_1, tokens_2, logits_1, logits_2, labels_1, labels_2): 
    aligned_logits_1 = torch.zeros(logits_1.shape)
    aligned_logits_2 = torch.zeros(logits_2.shape)
    aligned_tokens_1 = np.empty(tokens_1.shape, dtype="object")
    aligned_tokens_2 = np.empty(tokens_2.shape, dtype="object")
    aligned_labels_1 = np.empty(labels_1.shape, dtype="object")
    aligned_labels_2 = np.empty(labels_2.shape, dtype="object")

    # First, align 1 with 2
    for ex in range(tokens_1.shape[0]):
        alignment, _ = tokenizations.get_alignments(tokens_1[ex], tokens_2[ex])
        insert_idx = 0
        for i, token_idx_list in enumerate(alignment):
            if insert_idx >= aligned_tokens_1.shape[1]:
                break
            elif tokens_1[ex][i] == "[PAD]":
                aligned_tokens_1[ex, insert_idx] = "[PAD]"
                aligned_labels_1[ex, insert_idx] = -100
                aligned_logits_1[ex, insert_idx] = -100*torch.ones(logits_1.shape[2])
                insert_idx += 1
            elif len(token_idx_list) == 1:
                aligned_tokens_1[ex, insert_idx] = tokens_2[ex, token_idx_list[0]]
                aligned_labels_1[ex, insert_idx] = labels_1[ex, i].item() if torch.is_tensor(labels_1[ex, i]) else labels_1[ex, i]
                aligned_logits_1[ex, insert_idx, :] = logits_1[ex, i, :]
                insert_idx += 1
            elif len(token_idx_list) > 1:
                aligned_tokens_1[ex, insert_idx:insert_idx+len(token_idx_list)] = tokens_2[ex, token_idx_list]
                aligned_labels_1[ex, insert_idx:insert_idx+len(token_idx_list)] = labels_1[ex, i].item() if torch.is_tensor(labels_1[ex, i]) else labels_1[ex, i]
                aligned_logits_1[ex, insert_idx:insert_idx+len(token_idx_list)] = logits_1[ex, i, :]
                insert_idx += len(token_idx_list)
        while insert_idx < aligned_tokens_1.shape[1]:
            aligned_tokens_1[ex, insert_idx] = "[PAD]"
            aligned_labels_1[ex, insert_idx] = -100
            aligned_logits_1[ex, insert_idx, :] = -100*torch.ones(logits_1.shape[2])
            insert_idx += 1
            
    # Now, go the other way
    for ex in range(tokens_2.shape[0]):
        alignment, _ = tokenizations.get_alignments(tokens_2[ex], aligned_tokens_1[ex])
        insert_idx = 0
        for i, token_idx_list in enumerate(alignment):
            if insert_idx >= aligned_tokens_1.shape[1]:
                break
            elif tokens_2[ex][i] == "[PAD]":
                aligned_tokens_2[ex, insert_idx] = "[PAD]"
                aligned_labels_2[ex, insert_idx] = -100
                aligned_logits_2[ex, insert_idx] = -100*torch.ones(logits_2.shape[2])
                insert_idx += 1
            elif len(token_idx_list) == 1:
                aligned_tokens_2[ex, insert_idx] = aligned_tokens_1[ex, token_idx_list[0]]
                aligned_labels_2[ex, insert_idx] = labels_2[ex, i].item() if torch.is_tensor(labels_2[ex, i]) else labels_2[ex, i]
                aligned_logits_2[ex, insert_idx, :] = logits_2[ex, i, :]
                insert_idx += 1
            elif len(token_idx_list) > 1:
                aligned_tokens_2[ex, insert_idx:insert_idx+len(token_idx_list)] = aligned_tokens_1[ex, token_idx_list]
                aligned_labels_2[ex, insert_idx:insert_idx+len(token_idx_list)] = labels_2[ex, i].item() if torch.is_tensor(labels_2[ex, i]) else labels_2[ex, i]
                aligned_logits_2[ex, insert_idx:insert_idx+len(token_idx_list)] = logits_2[ex, i, :]
                insert_idx += len(token_idx_list)
        while insert_idx < aligned_tokens_2.shape[1]:
            aligned_tokens_2[ex, insert_idx] = "[PAD]"
            aligned_labels_2[ex, insert_idx] = -100
            aligned_logits_2[ex, insert_idx, :] = -100*torch.ones(logits_2.shape[2])
            insert_idx += 1
                 
    return aligned_tokens_1, aligned_tokens_2, aligned_logits_1, aligned_logits_2, aligned_labels_1, aligned_labels_2

def align_outputs_fc(tokens_1, tokens_2, logits_1, logits_2, labels_1, labels_2, hidden_1, hidden_2, device="cuda"): 
    aligned_logits_1 = torch.zeros(logits_1.shape)
    aligned_logits_2 = torch.zeros(logits_2.shape)
    aligned_tokens_1 = np.empty(tokens_1.shape, dtype="object")
    aligned_tokens_2 = np.empty(tokens_2.shape, dtype="object")
    aligned_labels_1 = np.empty(labels_1.shape, dtype="object")
    aligned_labels_2 = np.empty(labels_2.shape, dtype="object")
    aligned_hidden_1 = torch.zeros(hidden_1.shape)
    aligned_hidden_2 = torch.zeros(hidden_2.shape)

    # First, align 1 with 2
    for ex in range(tokens_1.shape[0]):
        alignment, _ = tokenizations.get_alignments(tokens_1[ex], tokens_2[ex])
        insert_idx = 0
        for i, token_idx_list in enumerate(alignment):
            if insert_idx >= aligned_tokens_1.shape[1]:
                break
            elif tokens_1[ex][i] == "[PAD]":
                aligned_tokens_1[ex, insert_idx] = "[PAD]"
                aligned_labels_1[ex, insert_idx] = -100
                aligned_logits_1[ex, insert_idx, :] = -100*torch.ones(logits_1.shape[2])
                aligned_hidden_1[ex, insert_idx, :] = hidden_1[ex, -1, :] #last embedding should be a pad token
                insert_idx += 1
            elif len(token_idx_list) == 1:
                aligned_tokens_1[ex, insert_idx] = tokens_2[ex, token_idx_list[0]]
                aligned_labels_1[ex, insert_idx] = labels_1[ex, i].item() if torch.is_tensor(labels_1[ex, i]) else labels_1[ex, i]
                aligned_logits_1[ex, insert_idx, :] = logits_1[ex, i, :]
                aligned_hidden_1[ex, insert_idx, :] = hidden_1[ex, i, :]
                insert_idx += 1
            elif len(token_idx_list) > 1:
                aligned_tokens_1[ex, insert_idx:insert_idx+len(token_idx_list)] = tokens_2[ex, token_idx_list]
                aligned_labels_1[ex, insert_idx:insert_idx+len(token_idx_list)] = labels_1[ex, i].item() if torch.is_tensor(labels_1[ex, i]) else labels_1[ex, i]
                aligned_logits_1[ex, insert_idx:insert_idx+len(token_idx_list)] = logits_1[ex, i, :]
                aligned_hidden_1[ex, insert_idx:insert_idx+len(token_idx_list)] = hidden_1[ex, i, :]
                insert_idx += len(token_idx_list)
        while insert_idx < aligned_tokens_1.shape[1]:
            aligned_tokens_1[ex, insert_idx] = "[PAD]"
            aligned_labels_1[ex, insert_idx] = -100
            aligned_logits_1[ex, insert_idx, :] = -100*torch.ones(logits_1.shape[2])
            aligned_hidden_1[ex, insert_idx, :] = hidden_1[ex, -1, :]
            insert_idx += 1
            
    # Now, go the other way
    for ex in range(tokens_2.shape[0]):
        alignment, _ = tokenizations.get_alignments(tokens_2[ex], aligned_tokens_1[ex])
        insert_idx = 0
        for i, token_idx_list in enumerate(alignment):
            if insert_idx >= aligned_tokens_1.shape[1]:
                break
            elif tokens_2[ex][i] == "[PAD]":
                aligned_tokens_2[ex, insert_idx] = "[PAD]"
                aligned_labels_2[ex, insert_idx] = -100
                aligned_logits_2[ex, insert_idx] = -100*torch.ones(logits_2.shape[2])
                aligned_hidden_2[ex, insert_idx, :] = hidden_2[ex, -1, :]
                insert_idx += 1
            elif len(token_idx_list) == 1:
                aligned_tokens_2[ex, insert_idx] = aligned_tokens_1[ex, token_idx_list[0]]
                aligned_labels_2[ex, insert_idx] = labels_2[ex, i].item() if torch.is_tensor(labels_2[ex, i]) else labels_2[ex, i]
                aligned_logits_2[ex, insert_idx, :] = logits_2[ex, i, :]
                aligned_hidden_2[ex, insert_idx, :] = hidden_2[ex, i, :]
                insert_idx += 1
            elif len(token_idx_list) > 1:
                aligned_tokens_2[ex, insert_idx:insert_idx+len(token_idx_list)] = aligned_tokens_1[ex, token_idx_list]
                aligned_labels_2[ex, insert_idx:insert_idx+len(token_idx_list)] = labels_2[ex, i].item() if torch.is_tensor(labels_2[ex, i]) else labels_2[ex, i]
                aligned_logits_2[ex, insert_idx:insert_idx+len(token_idx_list)] = logits_2[ex, i, :]
                aligned_hidden_2[ex, insert_idx:insert_idx+len(token_idx_list)] = hidden_2[ex, i, :]
                insert_idx += len(token_idx_list)
        while insert_idx < aligned_tokens_2.shape[1]:
            aligned_tokens_2[ex, insert_idx] = "[PAD]"
            aligned_labels_2[ex, insert_idx] = -100
            aligned_logits_2[ex, insert_idx, :] = -100*torch.ones(logits_2.shape[2])
            aligned_hidden_2[ex, insert_idx, :] = hidden_2[ex, -1, :]
            insert_idx += 1
                 
    return aligned_tokens_1, aligned_tokens_2, aligned_logits_1, aligned_logits_2, aligned_labels_1, aligned_labels_2, aligned_hidden_1.to(device), aligned_hidden_2.to(device)


def get_weighted_prob(logits, alpha_dict, id2label, ner_tags):
    weighting = []
    for idx in id2label:
        if id2label[idx] == "O":
            weighting.append(1) # Arbitrary weighting as seqeval discards O in calculation
        else:
            for tag in ner_tags:
                if re.match(fr".*{tag}.*", id2label[idx]):
                    weighting.append(alpha_dict[tag])
                    break
    return torch.tensor(weighting)*torch.softmax(logits, dim=-1)

def get_alpha_by_label(results, id2label):
    alphas = torch.zeros(len(id2label))
    epsilon = 1e-5
    for i, idx in enumerate(id2label):
        if id2label[idx] == "O":
            alphas[i] = np.log((1-0.5+epsilon)/(0.5+epsilon)) + np.log(len(id2label)-1)
        tag = re.search(r".*-([A-Z]+)", id2label[idx]).group(1) if re.match(r".*-([A-Z]+)", id2label[idx]) else None
        if tag:
            error = 1 - results[tag]["precision"] if tag in results else 0.5
            alphas[i] = np.log((1-error+epsilon)/(error+epsilon)) + np.log(len(id2label)-1)
    return alphas

def get_boosted_scores(predictions_by_model, alpha_by_model, id2label, logits_shape):
    scores = torch.zeros(logits_shape)
    for i, idx in enumerate(id2label):
        model_alphas = alpha_by_model[i, :]
        masked_predictions = predictions_by_model == idx
        model_alphas = model_alphas.unsqueeze(0).unsqueeze(1).repeat(\
                                                    masked_predictions.shape[0], masked_predictions.shape[1], 1)
        scores[:, :, i] = torch.sum(model_alphas*masked_predictions, dim=-1)
    return scores

def align_all(models, is_train, device="cuda", mode="voting"):
    dataloader_type = "train_dataloader" if is_train else "val_dataloader"
    to_align = deque([])
    for m in models:
        sublearner = models[m]
        batch = next(sublearner[dataloader_type]).to(device)
        tokens = get_tokens(sublearner["tokenizer"], batch)
        outputs = sublearner["model"].to(device)(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        hidden = torch.stack(outputs.hidden_states[-4:]).permute(1, 2, 3, 0).to(device)
        if mode == "voting":
            sublearner_alignment = {"name":sublearner["name"], "tokens":tokens, "logits":logits, "labels":labels}
        elif mode == "fc":
            sublearner_alignment = {"name":sublearner["name"], "tokens":tokens, "logits":logits, "labels":labels, "hidden":hidden}
        to_align.append(sublearner_alignment)
    n_repeat_alignments = 1 + math.floor(len(models)/2)
    completed_alignments = 0
    aligned = deque([])
    ref_alignment = to_align.popleft()
    aligned.append(ref_alignment)
    while to_align:
        model_1 = to_align.popleft()
        model_2 = aligned[-1]
        if mode == "fc":
            model_1["tokens"], model_2["tokens"], model_1["logits"], model_2["logits"], model_1["labels"], model_2["labels"], model_1["hidden"], model_2["hidden"] \
                    = align_outputs_fc(model_1["tokens"], model_2["tokens"], model_1["logits"], model_2["logits"], \
                                                                            model_1["labels"], model_2["labels"], model_1["hidden"], model_2["hidden"], device=device)
        elif mode == "voting":
            model_1["tokens"], model_2["tokens"], model_1["logits"], model_2["logits"], model_1["labels"], model_2["labels"] \
                    = align_outputs(model_1["tokens"], model_2["tokens"], model_1["logits"], model_2["logits"], \
                                                                            model_1["labels"], model_2["labels"])
        if completed_alignments < n_repeat_alignments:
            to_align.append(aligned.pop())
        aligned.append(model_1) 
        completed_alignments += 1
    return aligned