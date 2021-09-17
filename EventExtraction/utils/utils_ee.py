# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: utils_ee
    Author: czh
    Create Date: 2021/9/6
--------------------------------------
    Change Activity: 
======================================
"""
# 事件抽取util
import codecs
import json
from pathlib import Path
from typing import List, Dict, Union

from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


class ChineseTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
        super(ChineseTokenizer, self).__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = vocab_file

    def tokenize(self, text, **kwargs) -> List[str]:
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens


class DataProcessor(object):
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # DuEE1.0 苏剑林格式
    @classmethod
    def read_event_schema(cls, file_name_or_path: Union[str, Path] = None, alist: List = None):
        id2label, label2id = {}, {}
        event_type_dict = {}
        n = 0
        datas = []
        if file_name_or_path:
            with codecs.open(file_name_or_path, encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = json.loads(line)
                    datas.append(line)
        elif alist:
            datas = alist
        else:
            raise ValueError
        for line in datas:
            event_type = line['event_type']
            if event_type not in event_type_dict:
                event_type_dict[event_type] = {}
            event_type_dict[event_type]["name"] = line['class']
            for role in line['role_list']:
                event_type_dict[event_type][role["role"]] = role["name"]
                key = (event_type, role['role'])
                id2label[n] = key
                label2id[key] = n
                n += 1
        num_labels = len(id2label) * 2 + 1
        return id2label, label2id, num_labels, event_type_dict

    # DuEE1.0 苏剑林格式
    @classmethod
    def read_json(cls, file_name_or_path):
        lines = []
        with codecs.open(file_name_or_path, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                arguments = {}
                event_list = line.get("event_list", None)
                if event_list:
                    for event in line['event_list']:
                        for argument in event['arguments']:
                            key = argument['argument']
                            value = (event['event_type'], argument['role'])
                            arguments[key] = value
                # (text, {argument: (event_type, role)})
                lines.append((line['text'], arguments))
        return lines


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
        id2label:
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
        id2label:
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def extract_entities(seq, id2label, markup='bios'):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def get_argument_for_seq(pred_labels, id2label: Dict, suffix=None):
    """
    针对序列标注方式，提取argument, 参考苏剑林的代码https://github.com/bojone/lic2020_baselines/blob/master/ee.py
    :param pred_labels:
    :param id2label:
    :param suffix: 'BIOS', 'BIEOS', 'BIO'
    :return: [[label, start_id, end_id]]
    """
    label_entities = []
    if suffix in ["BIOS", 'BIO']:
        label_entities = extract_entities(pred_labels, id2label, suffix)
    elif suffix == "BIEOS":
        label_entities = get_entities(pred_labels)
    else:
        arguments = []
        starting = False
        for i, label in enumerate(pred_labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    arguments.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    arguments[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        for w, label in arguments:
            start = w[0]
            end = w[-1]
            label_entities.append([label, start, end])
    return label_entities
