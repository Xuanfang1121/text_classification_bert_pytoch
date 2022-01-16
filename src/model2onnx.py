# -*- coding: utf-8 -*-
# @Time: 2021/12/23 11:00
# @Author: zxf
import os

import torch

from config import config
from common.common import logger
from utils.util import read_dict
from models.TextClassificationModel import TextClassiModel


def model2onnx():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = "cpu"
    label2id = read_dict(config.label2id_path)
    logger.info("label2id: {}".format(label2id))
    # model
    model = TextClassiModel(config, len(label2id)).to(device)
    model.load_state_dict(torch.load(os.path.join(config.model_path, config.model_name),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    input_ids = torch.zeros([1, config.max_length], dtype=torch.long)
    input_type_ids = torch.zeros([1, config.max_length], dtype=torch.long)
    attention_mask = torch.zeros([1, config.max_length], dtype=torch.long)
    torch.onnx.export(model, (input_ids, input_type_ids, attention_mask),
                      config.onnx_model_path,
                      verbose=True, opset_version=10,
                      input_names=["input_ids", "input_type_ids", "attention_mask"],
                      output_names=["logits"],
                      dynamic_axes={'input_ids': [0, 1],
                                    "input_type_ids": [0, 1],
                                    "attention_mask": [0, 1],
                                    'logits': [0]
                                    }
                      )


if __name__ == "__main__":
    model2onnx()