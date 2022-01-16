# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 22:02
# @Author  : zxf
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer


"""
    这里的文本分类分为
    1. 取pool output
    2. 取cls
    3. 取任意一层等
"""


class TextClassiModel(nn.Module):
    def __init__(self, args, tag_nums):
        super(TextClassiModel, self).__init__()
        self.pretrain_model = args.pretrain_model
        self.max_length = args.max_length
        self.tag_nums = tag_nums
        self.lr = args.lr
        self.dropout = torch.nn.Dropout(args.dropout)
        self.model_type = args.model_type
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu" if args.gpu_ids == '-1' else "cuda"
        self.type = args.type
        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_model)
            self.model = BertModel.from_pretrained(self.pretrain_model)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_model)
            self.model = BertModel.from_pretrained(self.pretrain_model)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.tag_nums)

    # def forward(self, text, type):
    #     # 这里传入的是batch data,直接转为tensor
    #     batch_data = self.tokenizer(text,
    #                                 padding=True,
    #                                 truncation=True,
    #                                 max_length=self.max_length,
    #                                 return_tensors="pt").to(self.device)
    #     outputs = self.model(**batch_data,
    #                          output_hidden_states=True,
    #                          output_attentions=False
    #                          )
    #     if type == "pool":
    #         output = outputs['pooler_output']
    #         output = self.dropout(output)
    #     elif type == "cls":
    #         output = outputs["last_hidden_state"][:, 0]
    #     elif type == "last_avg":
    #         output = outputs["last_hidden_state"].mean(dim=1)
    #     elif type == "last_2_avg":
    #         # output = outputs["hidden_states"][:-2].mean(dim=1)
    #         output = (outputs["hidden_states"][-1] + outputs["hidden_states"][-2]).mean(dim=1)
    #     elif type == "first_last":
    #         output = (outputs["hidden_states"][1] + outputs["hidden_states"][-1]).mean(dim=1)
    #     elif type == "mean":
    #         # output = outputs["last_hidden_state"].mean(dim=1)
    #         attention_mask = batch_data['attention_mask'].unsqueeze(-1)
    #         output = torch.sum(outputs["last_hidden_state"] * attention_mask,
    #         dim=1) / torch.sum(attention_mask, dim=1)
    #     logits = self.linear(output)
    #     return logits

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 这里输入的是tensor
        outputs = self.model(input_ids, token_type_ids, attention_mask,
                             output_hidden_states=True,
                             output_attentions=False
                             )
        if self.type == "pool":
            output = outputs['pooler_output']
            output = self.dropout(output)
        elif self.type == "cls":
            output = outputs["last_hidden_state"][:, 0]
        elif self.type == "last_avg":
            output = outputs["last_hidden_state"].mean(dim=1)
        elif self.type == "last_2_avg":
            # output = outputs["hidden_states"][:-2].mean(dim=1)
            output = (outputs["hidden_states"][-1] + outputs["hidden_states"][-2]).mean(dim=1)
        elif self.type == "first_last":
            output = (outputs["hidden_states"][1] + outputs["hidden_states"][-1]).mean(dim=1)
        elif self.type == "mean":
            # output = outputs["last_hidden_state"].mean(dim=1)
            attention_mask = attention_mask.unsqueeze(-1)
            output = torch.sum(outputs["last_hidden_state"] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        logits = self.linear(output)
        return logits

    def get_loss(self, logits, y_true):
        loss = torch.nn.functional.cross_entropy(logits, y_true)
        return loss
