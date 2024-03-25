import torch
from torch import nn
import numpy as np
import evaluate
from transformers import BertTokenizerFast, BertForTokenClassification
import pickle
from dataset_utils import *
from ensemble_utils import *
from multilearner_class import *
from tqdm import tqdm
from transformers import logging
from transformers import get_scheduler
import warnings
import time
import argparse 
import sys
import os
import csv
import re

logging.set_verbosity_error()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run stacked ensemble')
parser.add_argument('--predict', action="store_true", help="Run model in training or prediction mode")
parser.add_argument('--train_filepath', type=str, required=False, help="Filepath to JSON containing training data")
parser.add_argument('--val_filepath', type=str, required=False, help="Filepath to JSON containing validation data")
parser.add_argument('--state_dict', type=str, required=False, help="Filepath to state dict for metalearner network parameters")
parser.add_argument('--save_dir', type=str, help="Directory to save results")
parser.add_argument('--save_filepath', type=str, help="Filepath to save results")
parser.add_argument('--id2label', default="id2label.pkl", type=str, help="Filepath to id2label relation")
parser.add_argument('--label2id', default="label2id.pkl", type=str, help="Filepath to label2id relation")
parser.add_argument('--batch_size', default=8, type=int, help="Batch size for DataLoaders")
parser.add_argument("--arch", type=str, choices=["fc", "bilstm"], help="Architecture to build on BERT embeddings")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=4.5e-5)
parser.add_argument("--decay", type=float, default=0.05)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--dropout", type=float, default=0.05)
args = parser.parse_args(sys.argv[1:])

predict = args.predict
train_filepath, val_filepath = args.train_filepath, args.val_filepath
save_dir, save_filepath = args.save_dir, args.save_filepath

if not predict:
    assert train_filepath, "Must provide data for training mode"
else:
    state_dict = args.state_dict
    assert val_filepath, "Must provide test data for evaluation mode"
    assert state_dict, "Must provide saved model parameters for evaluation mode"
    

lr, decay, warmup, dropout_rate = args.lr, args.decay, args.warmup, args.dropout
batch_size, num_epochs = args.batch_size, args.n_epochs
architecture = args.arch

with open(args.label2id, "rb") as handle:
    label2id = pickle.load(handle)

with open(args.id2label, "rb") as handle:
    id2label = pickle.load(handle)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device {device}")
print("Begin loading sublearners")
matbert_tokenizer = BertTokenizerFast.from_pretrained("matbert-base-cased", do_lower_case=False)
matbert_model = BertForTokenClassification.from_pretrained("matbert-base-cased", id2label=id2label, label2id=label2id, output_hidden_states=True)
matbert_model.load_state_dict(torch.load("matbert_tuned/checkpoint_epoch_2.pt", map_location=device)["model_state_dict"])
matbert_model.eval()

biobert_tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-base-cased-v1.2", do_lower_case=False)
biobert_model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2", id2label=id2label, label2id=label2id, output_hidden_states=True)
biobert_model.load_state_dict(torch.load("biobert_tuned/checkpoint_epoch_2.pt", map_location=device)["model_state_dict"])
biobert_model.eval()

scibert_tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_cased", do_lower_case=False)
scibert_model = BertForTokenClassification.from_pretrained("allenai/scibert_scivocab_cased", id2label=id2label, label2id=label2id, output_hidden_states=True)
scibert_model.load_state_dict(torch.load("scibert_tuned/checkpoint_epoch_2.pt", map_location=device)["model_state_dict"])
scibert_model.eval()

biobert_dataset = tokenize_dataset(train_filepath, val_filepath, biobert_tokenizer)
matbert_dataset = tokenize_dataset(train_filepath, val_filepath, matbert_tokenizer)
scibert_dataset = tokenize_dataset(train_filepath, val_filepath, scibert_tokenizer)

if not predict:
    biobert_train_dataloader, biobert_val_dataloader = prepare_dataloader(biobert_dataset.rename_column("ner_tags", "labels"), biobert_tokenizer, batch_size, shuffle=False)
    matbert_train_dataloader, matbert_val_dataloader = prepare_dataloader(matbert_dataset.rename_column("ner_tags", "labels"), matbert_tokenizer, batch_size, shuffle=False)
    scibert_train_dataloader, scibert_val_dataloader = prepare_dataloader(scibert_dataset.rename_column("ner_tags", "labels"), scibert_tokenizer, batch_size, shuffle=False)
else:
    biobert_test_dataloader = prepare_dataloader(biobert_dataset.rename_column("ner_tags", "labels"), biobert_tokenizer, batch_size, shuffle=False, predict=True)
    matbert_test_dataloader = prepare_dataloader(matbert_dataset.rename_column("ner_tags", "labels"), matbert_tokenizer, batch_size, shuffle=False, predict=True)
    scibert_test_dataloader = prepare_dataloader(scibert_dataset.rename_column("ner_tags", "labels"), scibert_tokenizer, batch_size, shuffle=False, predict=True)

biobert_model.to(device)
matbert_model.to(device)
scibert_model.to(device)

if architecture == "fc":
    ensemble = StackedEnsembleFC(dropout_rate=dropout_rate).to(device)
elif architecture == "bilstm":
    ensemble = StackedEnsembleBiLSTM(dropout_rate=dropout_rate).to(device)

if device.type == "cpu" or device.type == "mps":
        label_type = torch.LongTensor
else:
    label_type = torch.cuda.LongTensor

metric = evaluate.load("seqeval", zero_division=1)
start = time.time()

if not predict: # Run in training mode
    num_iter = num_epochs * len(biobert_train_dataloader)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    adam = torch.optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=decay)

    lr_scheduler = get_scheduler(
                "linear",
                optimizer=adam,
                num_warmup_steps=warmup,
                num_training_steps=num_iter,
            )

    best_val_f1 = 0
    iter_since_improvement = 0

    with tqdm(total=num_iter) as progress_bar:
        for epoch in range(num_epochs):
            biobert_train_dataloader, biobert_val_dataloader = prepare_dataloader(biobert_dataset.rename_column("ner_tags", "labels"), biobert_tokenizer, batch_size, shuffle=False)
            matbert_train_dataloader, matbert_val_dataloader = prepare_dataloader(matbert_dataset.rename_column("ner_tags", "labels"), matbert_tokenizer, batch_size, shuffle=False)
            scibert_train_dataloader, scibert_val_dataloader = prepare_dataloader(scibert_dataset.rename_column("ner_tags", "labels"), scibert_tokenizer, batch_size, shuffle=False)
            models = {}
            models["matbert"] = {"name":"matbert", "model":matbert_model, "tokenizer":matbert_tokenizer, "train_dataloader":iter(matbert_train_dataloader), "val_dataloader":iter(matbert_val_dataloader)}
            models["biobert"] = {"name":"biobert", "model":biobert_model, "tokenizer":biobert_tokenizer, "train_dataloader":iter(biobert_train_dataloader), "val_dataloader":iter(biobert_val_dataloader)}
            models["scibert"] = {"name":"scibert", "model":scibert_model, "tokenizer":scibert_tokenizer, "train_dataloader":iter(scibert_train_dataloader), "val_dataloader":iter(scibert_val_dataloader)}
            num_examples = batch_size * len(biobert_train_dataloader)
            num_val_examples = batch_size * len(biobert_val_dataloader)

            train_loss = 0
            print("-"*50)
            for batch_idx in range(len(biobert_train_dataloader)):
                ensemble.train()
                adam.zero_grad()
                aligned = align_all(models, is_train=True, device=device, mode="fc")
                hidden_layers = []
                while aligned:
                    sublearner = aligned.pop()
                    hidden = sublearner["hidden"]
                    hidden_layers.append(hidden.reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2]*hidden.shape[3]))
                    labels = torch.Tensor(sublearner["labels"].astype(int)).type(label_type).to(device)
                    del sublearner, hidden
                
                all_hidden = torch.stack(hidden_layers).permute(1, 2, 3, 0).to(device)
                outputs = ensemble(all_hidden)
                
                loss = loss_func(torch.permute(outputs, (0, 2, 1)), labels)
                
                loss.backward()
                adam.step()
                lr_scheduler.step()
                train_loss += loss.item()
                progress_bar.update(1)
                del all_hidden, outputs, aligned, hidden_layers
            
            del biobert_train_dataloader, matbert_train_dataloader, scibert_train_dataloader
            torch.cuda.empty_cache()

            with torch.no_grad():
                val_loss = 0
                for batch_idx in range(len(biobert_val_dataloader)):
                    aligned = align_all(models, is_train=False, device=device, mode="fc")
                    hidden_layers = []
                    while aligned:
                        sublearner = aligned.pop()
                        hidden = sublearner["hidden"]
                        hidden_layers.append(hidden.reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2]*hidden.shape[3]))
                        labels = torch.Tensor(sublearner["labels"].astype(int)).type(label_type).to(device)
                        del sublearner, hidden
                    
                    all_hidden = torch.stack(hidden_layers).permute(1, 2, 3, 0).to(device)
                    outputs = ensemble(all_hidden)
                    
                    loss = loss_func(torch.permute(outputs, (0, 2, 1)), labels)
                    val_loss += loss
                    predictions = outputs.argmax(dim=-1)
                    true_tags, predicted_tags = postprocess(predictions, labels, id2label)
                    metric.add_batch(predictions=predicted_tags, references=true_tags)
                    del all_hidden, outputs, aligned, hidden_layers
            
            print(f"Epoch {epoch+1} ({np.round((time.time()-start)/60, 2)} minutes elapsed)")
            print(f"Training loss {train_loss/num_examples}")
            print(f"Validation loss {val_loss/num_val_examples}")
            results = metric.compute()
            print(f"Validation F1 {results['overall_f1']}")
            
            if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            with open(f"{save_dir}/{save_filepath}_results", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([train_loss/num_examples, val_loss.item()/num_val_examples, results['overall_f1']])

            if results['overall_f1'] > best_val_f1:
                print("Best model! Saving checkpoint")
                best_val_f1 = results['overall_f1']
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': ensemble.state_dict(),
                            'optimizer_state_dict': adam.state_dict(),
                            'val_f1': results['overall_f1'],
                            },
                            f"{save_dir}/checkpoint_epoch_{epoch}.pt"
                        )
                f = open(f"{save_dir}/{save_filepath}_f1.txt", "w")
                f.write(str(best_val_f1))
                f.close()
            else:
                iter_since_improvement += 1
                if iter_since_improvement >= num_epochs/3:
                    print("Model not improving; terminating early")
                    del models, biobert_val_dataloader, matbert_val_dataloader, scibert_val_dataloader
                    torch.cuda.empty_cache()
                    break
                
            del models, biobert_val_dataloader, matbert_val_dataloader, scibert_val_dataloader
            torch.cuda.empty_cache()

else: # Run in prediction mode
    ensemble.load_state_dict(torch.load(state_dict, map_location=device)["model_state_dict"])
    tags = []
    with torch.no_grad():
        models = {}
        models["matbert"] = {"name":"matbert", "model":matbert_model, "tokenizer":matbert_tokenizer, "val_dataloader":iter(matbert_test_dataloader)}
        models["biobert"] = {"name":"biobert", "model":biobert_model, "tokenizer":biobert_tokenizer, "val_dataloader":iter(biobert_test_dataloader)}
        models["scibert"] = {"name":"scibert", "model":scibert_model, "tokenizer":scibert_tokenizer, "val_dataloader":iter(scibert_test_dataloader)}
        num_examples = batch_size * len(biobert_test_dataloader)
        with tqdm(total=num_examples) as progress_bar:
            for batch_idx in range(len(biobert_test_dataloader)):
                aligned = align_all(models, is_train=False, device=device, mode="fc")
                hidden_layers = []
                while aligned:
                    sublearner = aligned.pop()
                    hidden = sublearner["hidden"]
                    hidden_layers.append(hidden.reshape(hidden.shape[0], hidden.shape[1], hidden.shape[2]*hidden.shape[3]))
                    labels = torch.Tensor(sublearner["labels"].astype(int)).type(label_type).to(device)
                    del sublearner, hidden
                
                all_hidden = torch.stack(hidden_layers).permute(1, 2, 3, 0).to(device)
                outputs = ensemble(all_hidden)
                predictions = outputs.argmax(dim=-1)
                true_tags, predicted_tags = postprocess(predictions, labels, id2label)
                tags.extend(true_tags)
                metric.add_batch(predictions=predicted_tags, references=true_tags)
                progress_bar.update(1)
                del all_hidden, outputs, aligned, hidden_layers
                torch.cuda.empty_cache()
        
        results = metric.compute()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(results)
        with open(f'{save_dir}/{save_filepath}.pkl', 'wb') as handle:
            pickle.dump(results, handle)
        with open(f'{save_dir}/{save_filepath}_tags.pkl', 'wb') as handle:
            pickle.dump(tags, handle)