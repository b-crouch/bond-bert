import torch

def postprocess(predictions, labels, id2label):
    if torch.is_tensor(predictions):
        predictions = predictions.clone().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.clone().cpu().numpy()
    label_tags = [[id2label[l] for l in label if l != -100] for label in labels]
    prediction_tags = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    return label_tags, prediction_tags