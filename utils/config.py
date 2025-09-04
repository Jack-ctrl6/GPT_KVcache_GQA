import torch


class GPTConfig:
    def __init__(self):
        self.IGNORE_INDEX = -100
        self.pad_token_id = 50257
        self.max_length = 384      ## 输入最大长度，也是输入＋输出最大长度
        self.batch_size = 8       ## 批
        self.n_layers = 12         ## block 层数
        self.n_heads = 8          ## attention 头数 ，必须能被dims整除
        self.dims = 896            ## 隐藏层维度
        self.dropout = 0.1         ## 随机丢弃概率
        self.vocab_size = 50258    ## 词表大小2
        self.n_groups = 2          ## gqa中的参数 ，必须能被n_heads整除
        self.eos_token = 50256     ## 结束符
        self.max_new_tokens = 4096  ## 生成最大长度
        self.setting = True        ## kv cache 推理加速，True 代表开启，FALSE 代表禁止启用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
