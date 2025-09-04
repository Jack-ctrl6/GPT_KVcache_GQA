import torch
from torch.utils.data import Dataset
import tiktoken
import json
import numpy as np
from utils.config import GPTConfig
from typing import List, Tuple

torch.manual_seed(1024)

class nanoDataset(Dataset):
    def __init__(self, path):
        self.char_data = self._read_dataset(path)

    def _read_dataset(self, path):
        char_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())  # 使用 json.loads 解析单行
                char_data.extend(data.split('\n'))  # 打印解析后的数据
        return char_data

    def __len__(self):
        return len(self.char_data)

    def __getitem__(self, idx):
        chunk = self.char_data[idx]
        return chunk


class nanoCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def _convert_one_piece_tokens(self, piece):  ## 处理成tensor格式

        input_ids = torch.cat([torch.tensor(self.tokenizer.encode(piece)), torch.tensor([self.tokenizer.eot_token])])
        return input_ids

    def __call__(self, batch: List[str]):
        input_ids_list = []
        max_input_len_list = []
        final_labels_list = []
        for piece in batch:
            inputs_ids = self._convert_one_piece_tokens(piece)
            input_ids_list.append(inputs_ids)
            max_input_len_list.append(len(inputs_ids))

        max_length = max(max_input_len_list) + 2
        final_inputs_list = []
        for index, input_ids in enumerate(input_ids_list):
            pad_len = max_length - max_input_len_list[index]
            pad_input_ids = torch.concat([input_ids, torch.full([pad_len], self.config.pad_token_id)], axis=0)
            pad_labels = torch.concat([input_ids, torch.full([pad_len], self.config.IGNORE_INDEX)], axis=0)
            final_inputs_list.append(pad_input_ids[:-1])
            final_labels_list.append(pad_labels[1:])
        final_input_ids = torch.stack(final_inputs_list, axis=0)
        final_labels = torch.stack(final_labels_list, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.config.pad_token_id] = 0

        return final_input_ids, attention_mask, final_labels

