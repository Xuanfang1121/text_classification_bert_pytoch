# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 18:27
# @Author  : zxf
import json
import requests
import traceback

import numpy as np
import onnxruntime
from flask import Flask
from flask import jsonify
from flask import request
from transformers import AutoTokenizer
from transformers import BertTokenizer

from config import config
from utils.util import read_dict
from common.common import logger


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# tokenizer
if config.model_type == "bert":
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model)

# load label
label2id = read_dict(config.label2id_path)
logger.info("label2id: {}".format(label2id))
id2label = {value: key for key, value in label2id.items()}

# load model
sess = onnxruntime.InferenceSession(config.onnx_model_path)
logger.info("load onnx model finish")


@app.route('/text_classifier', methods=['POST'])
def text_classifier_infer():
    try:
        data_params = json.loads(request.get_data(), encoding="utf-8")
        text = data_params["context"]
        if len(text) > 0:
            # text list
            inputs = tokenizer.batch_encode_plus(text, padding=True,
                                                 truncation=True,
                                                 max_length=config.max_length)
            ort_inputs = {"input_ids": np.array(inputs["input_ids"], dtype=np.int64),
                          "input_type_ids": np.array(inputs["token_type_ids"], dtype=np.int64),
                          "attention_mask": np.array(inputs["attention_mask"], dtype=np.int64)}
            logits = sess.run(['logits'], ort_inputs)[0]
            pred = np.argmax(logits, 1).tolist()
            y_pred = [id2label[label] for label in pred]

            return jsonify({"code": 200,
                            "context": text,
                            "label": y_pred,
                            "message": "success"})
        else:
            return jsonify({"code": 200,
                            "context": text,
                            "label": [],
                            "message": "输入空数据"})
    except Exception as e:
        logger.info(e)
        return jsonify({"code": 400,
                        "message": traceback.format_exc()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)