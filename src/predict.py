# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 22:27
# @Author  : zxf
import os
import json
import traceback

import torch
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from config import config
from utils.util import read_dict
from common.common import logger
from utils.util import load_infer_data
from utils.util import predict_collate_fn
from utils.util import save_model_infer_data
from models.TextClassificationModel import TextClassiModel

"""
    这里的predict 分为加载原生模型和加载onnx模型做模型推理
"""


def predict():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = "cpu" if config.gpu_ids == "-1" else "cuda"
    # read data
    data = load_infer_data(config.test_data_file)
    logger.info("predict data size: {}".format(len(data)))
    # read label2id
    label2id = read_dict(config.label2id_path)
    id2label = {value: key for key, value in label2id.items()}
    test_dataloader = DataLoader(data, batch_size=config.batch_size,
                                 shuffle=False, collate_fn=predict_collate_fn)

    # model
    model = TextClassiModel(config, len(label2id)).to(device)
    model.load_state_dict(torch.load(os.path.join(config.model_path, config.model_name),
                                     map_location=lambda storage, loc: storage))
    # model.to(device)
    model.eval()
    y_pred = []
    for i, batch_data in enumerate(test_dataloader):
        batch_text = batch_data["text"]
        logits = model(batch_text)
        pred = torch.softmax(logits, dim=1)
        pred = torch.argmax(pred, 1).tolist()
        pred = [id2label[label] for label in pred]
        y_pred.extend(pred)

    result = []
    for i in range(min(len(data), len(y_pred))):
        result.append([y_pred[i], data[i]])

    save_model_infer_data(result, config.predict_file)


def model_onnx_predict():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = "cuda" if config.gpu_ids != "-1" else "cpu"
    # load inder data
    data = load_infer_data(config.test_data_file)
    logger.info("predict data size: {}".format(len(data)))
    # read label2id
    label2id = read_dict(config.label2id_path)
    logger.info("label2id: {}".format(label2id))
    id2label = {value: key for key, value in label2id.items()}
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model)
    logger.info("tokenizer finish")
    # 加载onnx模型
    sess = onnxruntime.InferenceSession(config.onnx_model_path)
    logger.info("load onnx finish")
    # ort_inputs = {"input": np.array(["荆楚网消息(楚天金报)记者王际凯报道:去年5月8日,"
    #                         "震惊象棋界的浙江队选手“买棋丑闻”,让广大棋迷领略到象棋的“另类魔力”。"
    #                         "一转眼,这一事件已经过去快一年的时间。"]),
    #               "type": config.type
    #               }
    data = ["昨天的中乙揭幕战可谓高朋满座,但最引人注目的恐怕要数客队上海东亚的总教练徐根宝了。"
            "2001年,从申花退位的徐根宝归隐江湖",
            "中新网2月21日电最新一期香港《亚洲周刊》载文说,台湾军方高层进行建军以来最大幅度的调动,包括陆、海、空三军“总司令”、"
            "“国防部”副部长等高层换人,以清理“国防部”前“部长”汤曜明所留下来的“汤家军”。"]
    y_pred = []
    for text in data:
        inputs = tokenizer.batch_encode_plus([text], padding=True,
                                             truncation=True,
                                             max_length=config.max_length)
        # print(inputs)
        ort_inputs = {"input_ids": np.array(inputs["input_ids"], dtype=np.int64),
                      "input_type_ids": np.array(inputs["token_type_ids"], dtype=np.int64),
                      "attention_mask": np.array(inputs["attention_mask"], dtype=np.int64)}

        logits = sess.run(['logits'], ort_inputs)[0]
        # 方式一： 标准计算
        pred = torch.softmax(torch.from_numpy(logits), dim=1)
        pred = torch.argmax(pred, 1).tolist()
        print("y_pred: ", pred)
        # 方式二：
        # pred = np.argmax(logits, 1).tolist()
        # print("y_pred: ", y_pred)
        pred = [id2label[label] for label in pred]
        print("pred: ", pred)
        y_pred.extend(pred)

    assert len(data) == len(y_pred), f'data size: {len(data)} y_pred size: {len(y_pred)}'
    result = []
    for i in range(len(data)):
        result.append([y_pred[i], data[i]])

    save_model_infer_data(result, config.predict_file)
    logger.info("model infer finish")


if __name__ == "__main__":
    # predict()
    model_onnx_predict()