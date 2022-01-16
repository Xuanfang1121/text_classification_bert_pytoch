### 文本分类
本项目的代码是基于pytorch实现的文本分类，是一个baseline模型，
模型包括了对bert编码得到的embedding层做了不同的向量，包括常见的
pooler，cls，最有一层的hidden states等
#### requirements
```
tqdm==4.62.0
numpy==1.19.2
torch==1.8.1
transformers==4.10.1
scikit_learn==1.0.1
```
#### 代码结构
```
   src
    |__common
    |__config  配置文件等
    |__data 数据
    |__log
    |__models 模型代码
    |__utils  工具类函数
    |__main.py  主代码
    |__predict.py 预测
    |__model2onnx.py 模型转onnx
    |__app.py flask 模型推理
    |__ output 输出目录
   
```
执行main.py 直接运行

#### 代码测试
在sogou文本分类数据集上进行测试模型的效果，数据的label分别为
体育，教育，健康，汽车，军事。模型效果如下：
```
                  precision    recall  f1-score   support
          体育       1.00      0.99      1.00       167
          军事       0.99      0.97      0.98       166
          教育       0.95      0.97      0.96       181
          健康       0.97      0.98      0.97       171
          汽车       1.00      0.85      0.92        13

    accuracy                           0.98       698
   macro avg       0.98      0.95      0.97       698
weighted avg       0.98      0.98      0.98       698

```
#### 模型推理接口格式
```
curl --location --request POST 'http://127.0.0.1:5000/text_classifier' \
--header 'Content-Type: application/json' \
--data-raw '{"context": ["昨天的中乙揭幕战可谓高朋满座,但最引人注目的恐怕要数客队上海东亚的总教练徐根宝了。2001年,从申花退位的徐根宝归隐江湖"]}'
```