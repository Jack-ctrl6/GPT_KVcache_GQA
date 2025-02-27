from utils.dataset import Return_Question_tensor
from utils.model import GPT
from utils.config import GPTConfig
import torch
import tiktoken

enc = tiktoken.get_encoding("gpt2")
config = GPTConfig()

model = GPT(config)  # 你的模型定义
model.to(config.device)

# 加载 checkpoint
checkpoint_path = 'checkpoints/model_epoch_199.pt'  # 替换为你要加载的 checkpoint 路径
checkpoint = torch.load(checkpoint_path)

# 恢复模型的状态
model.load_state_dict(checkpoint['model_state_dict'])

# 将模型设置为评估模式（如果你只是进行推理）
model.eval()

x = Return_Question_tensor("你好啊",config).to("cuda")
# x = torch.tensor([[50256, 50256, 50256]]).to('cuda')
# x1 = torch.randint(0,50257,(1,512)).to('cuda')
# print(x.shape)
# print(x1.shape)
out = model.generate(x)
output = enc.decode(out[:,config.max_length:].data.to("cpu").numpy()[0])
print("output:",output)

