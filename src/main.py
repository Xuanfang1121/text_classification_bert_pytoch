# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 23:52
# @Author  : zxf
import os
import json

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from config import config
from utils.util import load_data
from utils.util import save_dict
from common.common import logger
from utils.util import collate_fn
from utils.util import data_encoder
from utils.util import get_label_dict
from utils.util import model_train_step
from utils.util import get_model_evaluation
from models.TextClassificationModel import TextClassiModel


"""
     训练模型
"""


def train():
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    device = "cpu" if config.gpu_ids == '-1' else "cuda"
    # read data
    train_data = load_data(config.train_data_file)
    logger.info("train data size: {}".format(len(train_data)))
    dev_data = load_data(config.test_data_file)
    logger.info("dev data size:{}".format(len(dev_data)))
    # get label2id
    label2id = get_label_dict(train_data)
    print("label2id", label2id)
    # save dict
    save_dict(label2id, config.label2id_path)
    # data_encoder
    train_data = data_encoder(train_data, label2id)
    dev_data = data_encoder(dev_data, label2id)
    logger.info("train data and dev data encoder finish")
    # dataloader
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    # model
    model = TextClassiModel(config, len(label2id)).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    best_f1 = 0
    for epoch in range(config.epochs):
        train_loss_list = []
        train_acc_list = []
        for i, batch_data in enumerate(train_dataloader):
            batch_label = batch_data['class_labels'].tolist()
            batch_logits, batch_loss = model_train_step(model, batch_data, device)
            train_loss_list.append(batch_loss.tolist())
            # 计算 train acc
            pred = torch.softmax(batch_logits, dim=1)
            pred = torch.argmax(pred, 1).tolist()
            batch_acc = accuracy_score(batch_label, pred)
            train_acc_list.append(batch_acc)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (i + 1) % config.pre_step_print == 0:
                logger.info("epoch: {}/{} step: {}/{} train loss: {} train acc: {}".format(
                    epoch + 1, config.epochs, i + 1, len(train_dataloader),
                    np.mean(train_loss_list), np.mean(train_acc_list)
                ))

        # 模型进行验证
        acc, f1 = get_model_evaluation(dev_dataloader, model, label2id, device, logger)
        logger.info("epoch: {}/{} dev data accuracy: {} f1 score: {}".format(epoch + 1,
                                                                             config.epochs, acc,
                                                                             f1))
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.model_path,
                                                        config.model_name))


if __name__ == "__main__":
    train()
