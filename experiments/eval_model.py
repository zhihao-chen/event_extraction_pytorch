# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: eval_model
    Author: czh
    Create Date: 2021/9/16
--------------------------------------
    Change Activity: 
======================================
"""
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
    'cuda_number': '0',  # '0,1,2,3'
    'local_rank': -1,
}

args = DataAndTrainArguments(**config)
extractor = EventExtractor(args)

# evaluate all checkpoints file for the dev datasets
# extractor.evaluate(eval_all_checkpoints=True)

# only evaluate best model for the dev datasets
# extractor.evaluate()

# evaluate all checkpoints file for the test datasets, and the test datasets sample must labeled
# extractor.evaluate(data_type='test', eval_all_checkpoints=True)

# only evaluate best model for the test datasets, and the test datasets sample must labeled
extractor.evaluate(data_type='test')
