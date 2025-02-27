import torch
from torch.utils.data import Dataset
import tiktoken
import json
import numpy as np
from utils.config import GPTConfig


torch.manual_seed(1024)

# 写一个 dataset，为了 Dataloader 准备
class MyDataset(Dataset):
    def __init__(self, path, config):
        # 数据 mobvoi_seq_monkey_general_open_corpus.jsonl 中，
        # 读取前 1000 行
        self.enc = tiktoken.get_encoding("gpt2")
        self.max_length = config.max_length

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]  # 数字代表是 50256
        self.encoded_data = []
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析每一行的 JSON 对象
                data = json.loads(line.strip())  # 使用 json.loads 解析单行
                raw_data.append(data)  # 打印解析后的数据

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.max_length):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i + self.max_length + 1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.max_length + 1:
                chunk = chunk + [self.eos_token] * (self.max_length + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)


def Return_Question_tensor(question,config):   ## 对question进行序列化 ，如果输入过长可以增加max_length

    enc = tiktoken.get_encoding("gpt2")
    encoded_text = enc.encode(question)
    full_encoded = encoded_text + [config.eos_token]
    encoded_data = []
    assert len(full_encoded) < config.max_length   ## embedding 之后的长度不能大于max_length
    for i in range(0, 1, 512):
        # 多取一个 Token 作为目标
        chunk = full_encoded[i:i + config.max_length + 1]
        # 如果长度不够，用 eos_token 填充
        if len(chunk) < config.max_length + 1:
            chunk = chunk + [config.eos_token] * (config.max_length  - len(chunk))
        encoded_data.append(chunk)
    return torch.tensor(encoded_data,dtype=torch.long)


if __name__ == '__main__':
    s= Return_Question_tensor("你好",config=GPTConfig())
    print(np.array(s).shape)
    print(s)
