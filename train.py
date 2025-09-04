import tiktoken

from utils.dataset import nanoDataset, nanoCollator
from utils.model import GPT
import torch
from torch.utils.data import DataLoader
from utils.config import GPTConfig

config = GPTConfig()
train_dataset = nanoDataset(r'./data/mobvoi_seq_monkey_general_open_corpus_1000.json')
tokenizer = tiktoken.get_encoding('gpt2')
nanocollator = nanoCollator(tokenizer, config)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=nanocollator)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, collate_fn=nanocollator)

model = GPT(config)
device = config.device
model = model.to(device)

# 打印模型一共有多少参数

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# 设置 cosine 学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)


# 训练循环
def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, labels,) in enumerate(train_loader):
        # 将数据移到设备上
        input_ids, attention_mask, labels = input_ids.to(device).long(), attention_mask.to(device).long(), labels.to(
            device).long()

        # 前向传播
        logits, loss = model(input_ids, targets=labels, padding_attention_mask=attention_mask)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 调整学习率
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss


def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (input_ids, attention_mask, labels,) in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device).long(), attention_mask.to(
                device).long(), labels.to(device).long()

            logits, loss = model(input_ids, targets=labels, padding_attention_mask=attention_mask)
            val_loss += loss.item()
    return val_loss


for epoch in range(200):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
    val_loss = eval(model, val_loader, device)
    print(
        f'Epoch: {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    # 保存模型
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }
    # 保存每个epoch的模型
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
