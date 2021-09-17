# EventExtraction（适用于事件抽取任务）

# 简介

主要参考苏剑林的方法：https://github.com/bojone/lic2020_baselines/blob/master/ee.py改动实现的pytorch版本  
在原有代码上修改了评价方案，使得评价指标更合理。另外，该方案实质是采用序列标注方式的联合抽取模型，该方案不支持event trigger的检测和抽取。

# 安装

## 环境依赖

1. python >= 3.6
2. seqeval==1.2.2
3. transformers==4.10.0
4. pydantic==1.8.2
5. accelerate==0.4.0
6. tensorflow==2.6.0
7. torch==1.9.0

# 训练数据格式

数据格式参考百度千言数据集DuEE1.0，数据格式如下（每行一个json。若无event_type，需标注event_type为OHTER；若无arguments， 可标注arguments为[]）

```
{"text": "盛世投资:积极践行ESG投资和绿色投资", "id": "1031-0", "event_list": [{"event_type": "BUSEXP", "arguments": [{"argument": "盛世投资", "role": "ORG", "argument_start_index": 0}, {"argument": "ESG投资和绿色投资", "role": "CHA", "argument_start_index": 9}]}]}
{"text": "人人网向开心汽车投资600万美元 后者股价5个多月下跌超67%", "id": "4724-0", "event_list": [{"event_type": "OTHER", "arguments": [{"argument": "人人网", "role": "ORG", "argument_start_index": 0}]}]}
```

除了train.json, dev.json, test.json之外还需要event_schema.json，格式如下:

```
{"event_type": "STRCOO", "role_list": [{"role": "ORG", "name": "公司名称"}, {"role": "CHA", "name": "业务方向"}], "class": "战略合作"}
```

# 使用方法

## 训练

最重要的是配置config字典，只需要改变config字典中的参数即可实现从头开始训练和恢复训练。

```
from EventExtraction import EventExtractor, DataAndTrainArguments

config = {
    'task_name': 'ee',
    'data_dir': '../data/normal_data/news2',
    'model_type': 'bert',
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',
    'model_sate_dict_path': '../data/output/bert/best_model',   # 保存的checkpoint文件地址用于继续训练
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录
    'cache_dir': '',   # 指定下载的预训练模型保存地址
    'evaluate_during_training': True,
    'do_eval_per_epoch': True,
    'use_lstm': True,
    'from_scratch': True,
    'from_last_checkpoint': False,
    'early_stop': False,
    'overwrite_output_dir': True,
    'overwrite_cache': True,
    'no_cuda': False,
    'fp16': True,
    'train_max_seq_length': 128,
    'eval_max_seq_length': 128,
    'per_gpu_train_batch_size': 16,
    'per_gpu_eval_batch_size': 16,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-05,
    'crf_learning_rate': 5e-05,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-08,
    'warmup_proportion': 0.1,
    'num_train_epochs': 50.0,
    'max_steps': -1,
    'tolerance': 5,   # 指定early stop容忍的epoch数量
    'logging_steps': 500,
    'save_steps': 500,
    'scheduler_type': 'linear',   # ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]
    'cuda_number': '0',   # '0,1,2,3'
    'seed': 2333,
    'local_rank': -1,
    'dropout_rate': 0.3
}

args = DataAndTrainArguments(**config)
extractor = EventExtractor(args)

# training from scratch, set config['from_scratch'] = True
extractor.train_and_valid()

# continue train from 'model_sate_dict_path', set config['from_scratch'] = False
extractor.train_and_valid()

# continue train from last checkpoint file, set config['from_scratch'] = False, config['from_last_checkpoint']=True.
# And should rise the 'num_train_epochs'
extractor.train_and_valid()
```

## 验证模型指标

模型的验证方法有4种使用方式，与训练过程一样，也需要配置config字典。

```
from EventExtraction import EventExtractor, DataAndTrainArguments

config = {
    'task_name': 'ee',
    'data_dir': '../data/normal_data/news2',
    'model_type': 'bert',
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录
    'do_lower_case': False,
    'use_lstm': False,
    'no_cuda': False,
    'eval_max_seq_length': 128,
    'per_gpu_eval_batch_size': 8,
    'cuda_number': '0',   # '0,1,2,3'
    'local_rank': -1,
}

args = DataAndTrainArguments(**config)
extractor = EventExtractor(args)

# evaluate all checkpoints file for the dev datasets
extractor.evaluate(eval_all_checkpoints=True)

# only evaluate best model for the dev datasets
extractor.evaluate()

# evaluate all checkpoints file for the test datasets, and the test datasets sample must labeled
extractor.evaluate(data_type='test', eval_all_checkpoints=True)

# only evaluate best model for the test datasets, and the test datasets sample must labeled
extractor.evaluate(data_type='test')
```

## 预测

```
from EventExtraction import EventExtractor, DataAndTrainArguments

config = {
    'task_name': 'ee',
    'data_dir': '../data/normal_data/news2',
    'model_type': 'bert',
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录
    'do_lower_case': True,
    'use_lstm': False,
    'no_cuda': False,
    'eval_max_seq_length': 128,
}

args = DataAndTrainArguments(**config)
extractor = EventExtractor(args, state='pred')

# data_type: 只能是'test'，或者None。若为test则表示在测试数据集上预测
# input_texts: 若不为空，则表示是预测新的数据
# pred_output_dir: 若不为空，则表示将预测结果写入指定位置保存，可以是目录，也可以是文件

# 表示在测试数据集上预测, 不保存预测结果
for res in extractor.predict(data_type='test'):
    print(res)

# 表示在测试数据集上预测, 保存预测结果
for res in extractor.predict(data_type='test', pred_output_dir="../data/output/bert"):
    print(res)

# 表示预测raw text, raw text可以是str, List[str]
texts = "博盛医疗完成Pre-A轮融资澳银资本重点参与"
texts = ["博盛医疗完成Pre-A轮融资澳银资本重点参与",
         "百炼智能完成A轮一亿元融资，由今日头条领投"]
for res in extractor.predict(input_texts=texts):
    print(res)
```
返回格式：
```
{
    "text":"",
    "event_list": [
        {
            "event_type":"",
            'event_type_name':
            "arguments": [
                {
                    "role":"",
                    "role_name":"",
                    "argument": ""
                }
            ]
        }
    ]
 }
```
