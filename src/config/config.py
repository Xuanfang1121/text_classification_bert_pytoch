# -*- coding: utf-8 -*-
# @Time: 2021/8/17 10:48
# @Author: zxf


# pretrain model
# pretrain_model = "./pretrain_model/bert-base-chinese/"
pretrain_model = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/"
# pretrain_model = "/opt/nlp/pretrain_model/chinese-roberta-wwm-ext"
model_type = "bert"

# file
train_data_file = "./data/sogou/train_demo.data"
dev_data_file = "./data/sogou/test_demo.data"
test_data_file = "./data/sogou/test_demo.data"
pred_result_file = "./output/roberta_pred_sougou_result.json"
error_result_file = "./output/roberta_pred_sogou_error.json"

# data
# train_data_file = "./baidu_event/train.data"
# dev_data_file = "./baidu_event/dev.data"
# test_data_file = "./baidu_event/dev.data"
# pred_result_file = "./output/roberta_pred_baidu_result.json"
# error_result_file = "./output/roberta_pred_baidu_error.json"
# model params
batch_size = 4
lr = 1e-5
epochs = 1
max_length = 256
dropout = 0.3
gpu_ids = "-1"
pre_step_print = 100
type = "pool"  # 输出选择，包括cls，last_avg，last_2_avg，first_last，mean

# cls_label2id_path = "./output/cls_label2id_sogou.dict"
# cls_model_name = 'elec_docu_bert_model_roberta_sogou--{f1_score:.4f}'
# # # onnx model
# cls_onnx_model_path = "./output/elec_docu_roberta_sogou.onnx"
model_path = "./output/"
label2id_path = "./output/cls_label2id.dict"
model_name = 'sugo_model.pt'
predict_file = "./output/predict_result.txt"
# # onnx model
onnx_model_path = "./output/text_classification.onnx"

