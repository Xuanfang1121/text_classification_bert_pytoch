# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 23:18
# @Author  : zxf
import os
import json

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import BertTokenizer
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

from config import config


# tokenizer
if config.model_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model)


def save_dict(dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict, ensure_ascii=False, indent=2))


def read_dict(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_data(data_file):
    """
    :param data_file: sting
    :return: list
    数据每一行 label \t text
    """
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split()   # '\t'
            if len(line) == 2:
                data.append([line[0], line[1]])
    return data


# read 预测data
def load_infer_data(data_file):
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def get_label_dict(data):
    label_dict = {}
    for iterm in data:
        if iterm[0] not in label_dict:
            label_dict[iterm[0]] = len(label_dict)
    return label_dict


def data_encoder(data, label2id):
    """
    :param data: [(label, text)]
    :param label2id: dict
    :return: list
    根据label字典进行encoder
    """
    result = [(label2id[iterm[0]], iterm[1]) for iterm in data]
    return result


# def collate_fn(batch_data):
#     text = [item[1] for item in batch_data]
#     class_labels = torch.LongTensor([item[0] for item in batch_data])
#     return {'text': text, 'class_labels': class_labels}


def collate_fn(batch_data):
    text = [item[1] for item in batch_data]
    encoder = tokenizer.batch_encode_plus(text, padding=True,
                                          truncation=True,
                                          max_length=config.max_length,
                                          return_tensors='pt')
    labels = torch.LongTensor([item[0] for item in batch_data])
    return {"input_ids": encoder["input_ids"],
            "token_type_ids": encoder["token_type_ids"],
            "attention_mask": encoder["attention_mask"],
            "class_labels": labels}


# def get_model_evaluation(data_iter, model, label2id, type, logger):
#     """
#        新增模型评价指标
#     """
#     model.eval()
#     id2label = {value: key for key, value in label2id.items()}
#     y_pred = []
#     y_true = []
#     with torch.no_grad():
#         for batch_iterm in tqdm(data_iter, total=len(data_iter)):
#             logits = model(batch_iterm['text'], type)
#             pred = torch.softmax(logits, dim=1)
#             pred = torch.argmax(pred, 1).tolist()
#             true_label = batch_iterm['class_labels'].tolist()
#             assert len(pred) == len(true_label), print("预测结果和真实batch的数量不一致")
#             y_pred.extend(pred)
#             y_true.extend(true_label)
#     # 计算模型的评价指标结果
#     acc = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     pred_label_list = unique_labels(y_true, y_pred).tolist()
#     target_names = [id2label[iterm] for iterm in pred_label_list]
#     logger.info(classification_report(y_true, y_pred,
#                                       target_names=target_names))
#     logger.info("model test data result \n")
#     logger.info("accuracy: {} precision: {} recall: {} f1: {}".format(acc,
#                                                                       precision,
#                                                                       recall,
#                                                                       f1))
#     return acc, f1


def get_model_evaluation(data_iter, model, label2id, device, logger):
    """
       新增模型评价指标
    """
    model.eval()
    id2label = {value: key for key, value in label2id.items()}
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_iterm in tqdm(data_iter, total=len(data_iter)):
            input_ids = batch_iterm["input_ids"].to(device)
            token_type_ids = batch_iterm["token_type_ids"].to(device)
            attention_mask = batch_iterm["attention_mask"].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            pred = torch.softmax(logits, dim=1)
            pred = torch.argmax(pred, 1).tolist()
            true_label = batch_iterm['class_labels'].tolist()
            assert len(pred) == len(true_label), print("预测结果和真实batch的数量不一致")
            y_pred.extend(pred)
            y_true.extend(true_label)
    # 计算模型的评价指标结果
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    pred_label_list = unique_labels(y_true, y_pred).tolist()
    target_names = [id2label[iterm] for iterm in pred_label_list]
    logger.info(classification_report(y_true, y_pred,
                                      target_names=target_names))
    logger.info("model test data result \n")
    logger.info("accuracy: {} precision: {} recall: {} f1: {}".format(acc,
                                                                      precision,
                                                                      recall,
                                                                      f1))
    return acc, f1
# def model_train_step(model, batch_data, device, type):
#     batch_text = batch_data["text"]
#     batch_label = batch_data['class_labels']
#     logits = model(batch_text, type)
#     loss = model.get_loss(logits, batch_label.to(device))
#     return logits, loss


# 模型训练
def model_train_step(model, batch_data, device):
    input_ids = batch_data["input_ids"].to(device)
    token_type_ids = batch_data["token_type_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    batch_label = batch_data['class_labels'].to(device)
    logits = model(input_ids, token_type_ids, attention_mask)
    loss = model.get_loss(logits, batch_label)
    return logits, loss


# save model predict result
def save_model_infer_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for iterm in data:
            f.write(iterm[0] + "\t" + iterm[1] + "\n")
    print("model predict result save finish")


def predict_collate_fn(batch_data):
    text = [item[0] for item in batch_data]
    return {'text': text}