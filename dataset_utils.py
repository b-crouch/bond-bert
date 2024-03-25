import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader


def clean_matscholar(filepath):
     df = pd.read_json(filepath, orient='split')
     ner = df["labels"].apply(lambda x: [i for i in x]).to_frame().rename(columns={"labels":"tokens"})
     ner["ner_names"] = df["labels"].apply(lambda x: [x[i] for i in x])
     label2id = {name:i for i, name in enumerate(ner["ner_names"].explode().unique())}
     id2label = {label2id[key]:key for key in label2id}
     ner["ner_tags"] = ner["ner_names"].apply(lambda x: [label2id[name] for name in x])
     return ner, label2id, id2label

def align_tokens_tags(tags, word_ids):
    aligned_tags = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned_tags.append(-100)
        elif word_id != current_word:
            aligned_tags.append(tags[word_id])
        else:
            tag = tags[word_id]
            # Odd tags are B- (start of word)
            # If previous tag of the same word was B-, any subsequent tags should be I-
            if tag%2 == 1:
                tag += 1
            aligned_tags.append(tag)
    return aligned_tags

def tokenize_and_align(data, tokenizer):
    tokenized_inputs = tokenizer(data["tokens"], is_split_into_words=True, padding="max_length", max_length=256)
    all_tags = data["ner_tags"]
    aligned_tags = []
    for i, tags in enumerate(all_tags):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_tags.append(align_tokens_tags(tags, word_ids))
    tokenized_inputs["ner_tags"] = aligned_tags
    return tokenized_inputs

def tokenize_dataset(train_filepath, test_filepath, tokenizer):
    if not train_filepath:
        data = load_dataset("json", data_files={"test":test_filepath})
    else:
        data = load_dataset("json", data_files={"train":train_filepath, "test":test_filepath})
    aligned_data = data.map(tokenize_and_align, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=data["test"].column_names)
    return aligned_data

def prepare_dataloader(dataset, tokenizer, batch_size, shuffle=True, predict=False):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
    test_dataloader = DataLoader(dataset["test"], shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)
    if not predict:
        train_dataloader = DataLoader(dataset["train"], shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)
        return train_dataloader, test_dataloader
    return test_dataloader

def init_matbert(label2id, id2label):
    matbert_tokenizer = BertTokenizerFast.from_pretrained("matbert-base-cased", do_lower_case=False)
    matbert_model = BertForTokenClassification.from_pretrained("matbert-base-cased", id2label=id2label, label2id=label2id)
    return matbert_tokenizer, matbert_model