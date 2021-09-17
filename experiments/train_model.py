# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: train_model
    Author: czh
    Create Date: 2021/9/15
--------------------------------------
    Change Activity: 
======================================
"""
import sys
sys.path.append('/data/chenzhihao/EventExtraction')
from EventExtraction import EventExtractor, DataAndTrainArguments

config = {
    'task_name': 'ee',
    'data_dir': '../data/normal_data/news2',
    'model_type': 'bert',
    'model_name_or_path': 'hfl/chinese-roberta-wwm-ext',
    'model_sate_dict_path': '../data/output/bert/best_model',  # 保存的checkpoint文件地址用于继续训练
    'output_dir': '../data/output/',  # 模型训练中保存的中间结果，模型，日志等文件的主目录
    'cache_dir': '',  # 指定下载的预训练模型保存地址
    'evaluate_during_training': True,
    'do_eval_per_epoch': True,
    'use_lstm': False,
    'from_scratch': True,
    'from_last_checkpoint': False,
    'early_stop': False,
    'overwrite_output_dir': True,
    'overwrite_cache': True,
    'no_cuda': False,
    'fp16': True,
    'train_max_seq_length': 128,
    'eval_max_seq_length': 128,
    'per_gpu_train_batch_size': 32,
    'per_gpu_eval_batch_size': 32,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-05,
    'crf_learning_rate': 5e-05,
    'weight_decay': 0.01,
    'adam_epsilon': 1e-08,
    'warmup_proportion': 0.1,
    'num_train_epochs': 30.0,
    'max_steps': -1,
    'tolerance': 5,  # 指定early stop容忍的epoch数量
    'logging_steps': 500,
    'save_steps': 500,
    'scheduler_type': 'cosine',
    # ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]
    'cuda_number': '0',  # '0,1,2,3'
    'seed': 2333,
    'local_rank': -1,
    'dropout_rate': 0.3
}

args = DataAndTrainArguments(**config)  # noqa
extractor = EventExtractor(args)

# training from scratch, set config['from_scratch'] = True
extractor.train_and_valid()

# continue train from 'model_sate_dict_path', set config['from_scratch'] = False
# extractor.train_and_valid()

# continue train from last checkpoint file, set config['from_scratch'] = False, config['from_last_checkpoint']=True.
# And should rise the 'num_train_epochs'
# extractor.train_and_valid()
