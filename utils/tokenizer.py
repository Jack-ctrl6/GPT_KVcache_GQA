from typing import List
import tiktoken
import torch

class Tokenizer:
    def __init__(self,config,tokenzier_name='gpt2'):
        '''
        :param tokenzier_name: only support tiktoken methods
        '''
        self.tokenizer = tiktoken.get_encoding(tokenzier_name)
        self.config = config
        self.pad_token_id = config.pad_token_id

    def encode(self,text:str)->torch.Tensor:
        input_ids = torch.cat([
            torch.tensor(self.tokenizer.encode(text)), torch.tensor([self.tokenizer.eot_token])
        ],axis = 0).long()
        return input_ids.unsqueeze(0)

    def batch_encode(self,text_list:List[str])->torch.Tensor:
        '''
        notice: model do not have batch_generate_ids,so if you want to use batch_generate_ids,
        you can use tokenzier's batch_encode and batch_decode function,but you should use a loop model.generate
        :param text_list:
        :return:
        '''
        batch_input_ids = self.tokenizer.encode_batch(text_list)
        inputs_ids_length = [len(i) for i in batch_input_ids]
        pad_batch_input_ids = []
        max_length = min(max(inputs_ids_length),self.config.max_length) + 1
        for index,each_input_ids in enumerate(batch_input_ids):
            ## 关键是inputs_ids_length是超过max_length该怎么做，也就是config maxlengh小该怎么做
            if inputs_ids_length[index] > max_length:
                pad_batch_input_ids.append(torch.tensor(each_input_ids[:max_length - 1] + [self.tokenizer.eot_token]))
            else:
                pad_batch_input_ids.append(torch.tensor(each_input_ids + [self.tokenizer.eot_token] +
                                           [self.pad_token_id] * (max_length - inputs_ids_length[index] - 1)))
        # print(pad_batch_input_ids)
        pad_batch_input_ids = torch.stack((pad_batch_input_ids),axis=0).long()

        return pad_batch_input_ids


    def decode(self,outputs_ids:torch.Tensor)->str:
        outputs = outputs_ids.to('cpu').detach().numpy()[0]
        return self.tokenizer.decode(outputs)

    def batch_decode(self,outputs_ids:torch.Tensor)->List[str]:  ## exclude padding
        outputs = [i[0].to('cpu').detach().numpy() for i in outputs_ids]
        return self.tokenizer.decode_batch(outputs)


if __name__ == '__main__':
    from utils.config import GPTConfig
    config = GPTConfig()
    tk = Tokenizer(config)
    print(tk.batch_decode([torch.tensor([[28839,   101,   162,   253,    98, 13783,   226,   164,   247,   248,
          28156,   222,   161,    95,   252,   161,   222,   120,   163,   101,
            236, 10310,   241, 18796,   101, 20998,   239,   163,    98,   101,
            162,    94,   230, 20015,   114, 40792,   171,   120,   234, 30585,
            116, 30585,   116,   162,   114,   231, 20998,   232, 32573,   249,
            165,    94,   117, 45911,   247,   162,   232,   113,   163,   101,
            236,   165,    95,   251,   161,   240,   234,   163,   101,   236,
            162,   105,   122,   162,   235,   253, 13783,   109, 21410,   164,
            106,    97, 22522,   248,   161,   240,   234, 13783,   226, 49426,
            228, 16764, 50256,   163,   120,   244,   163,   254,   242,   163,
            102,   114, 43095, 37345,   243,   162,    94,   230,   160,   122,
            233, 36685,   224,    25,   198,   164,   229,   103, 10310,   119,
            165,    95,   246, 43095, 37345,   243,   162,   224,   231,   171,
            120,   234,   165,   250,   222, 17358,   223, 37345,   101, 35707,
            237, 21410,   165,    94,   117, 45911,   247,   162,   232,   113,
            163,   101,   236,   165,    95,   251,   171,   120,   234, 20015,
             98,   160,   122,   233, 36685,   224,    25,   198,    16,    13,
            198, 10310,   118, 29826,    97,   171,   120,   234, 10310,   118,
            165,    94,   117, 33566,   106, 35707,   237, 12859,   228,   163,
            101,   236,   162,   105,   122,   162,   235,   253, 13783,   109,
          46479,   251,   165,   247,   102,   171,   120,   234, 18796,   109,
          12859,   236,   163,   118,    99, 10310,   118,   163,   119,   241,
            161,   240,   234,   163,   101,   236,   162,   105,   122,   162,
            235,   253, 13783,   109, 21410, 22522,   252,   165,   247,   227,
          46349,   227, 37863,   113,   171,   120,   234,   164,   106,    97,
          22522,   248, 42468, 10310,   118,   165,    95,   251,   163,   120,
            118, 13783,   109,   164,   106,    94,   163,   106,   245,   163,
            101,   236, 21410, 34932,   229, 46268,   161,   252,   233,   171,
            120,   234,   165,    95,   226,   163,   224,   117,   162,   107,
            237, 10310,   103, 17312,   230,   161,    95,   252, 27950,   254,
          50256]], device='cuda:0'),
 torch.tensor([[28839,   101,   164,   106,    94,   163,   106,   245,   163,   101,
            236,   162,   105,   122,   162,   235,   253, 13783,   109, 33768,
            114,   171,   120,   234, 17358,   223, 38834, 17358,   223, 49546,
          32573,   249,   165,    94,   117, 45911,   247,   162,   232,   113,
            163,   101,   236,   165,    95,   251, 44293,   227,   162,   233,
            105, 28839,   101, 37863,   227,   171,   120,   253, 50256, 50257,
          50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
          50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
          50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257,
          50257, 50257, 50257, 16764,   198, 22522,   252, 31965,   102,   163,
            241,   114, 20998,   232,   165,    94,   117, 33566,   106,   171,
            120,   234,   162,   232,   113,   162,   232,   222, 17312,   107,
            163,   101,   236,   163,   236,   229, 42468, 31660,   163,   100,
            235, 22522,    97,   163,   254,   242,   163,   102,   114,   163,
            111,   119, 26344,   245, 23513, 45635, 18796,   101, 20015,    98,
          20998,   232,   165, 45865,   163,   255,   231,   163,   119,   241,
            162,   252,   226, 37605,   241,     7, 28938,   230, 43718,   229,
          49035,   228,   171,   120,   234,    46,   306,    12,    22,   767,
            767,   767, 50256]], device='cuda:0')]))