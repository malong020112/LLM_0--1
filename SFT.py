import os 
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from model import Llama
from Config import LLMConfig
from dataset import SFTDataset


# 余弦退火调整学习率
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数，使用交叉熵损失

    start_time = time.time()  # 记录训练开始时间
    

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        

        #更细优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 在上下文管理器中进行前向传播和反向传播
        with ctx:
            res = model(X) #前向传播得到结果
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps # 梯度累积

        # 反向传播
        scaler.scale(loss).backward()
        # 每累积一定步数后进行优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 解除梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪

            scaler.step(optimizer) # 执行优化步骤
            scaler.update() # 更新缩放器
            # scheduler.step()
            optimizer.zero_grad(set_to_none=True) # 清空梯度

        # 每隔一定步数记录一次日志
        if step % args.log_step == 0:
            spend_time = time.time() - start_time  # 计算当前的时间
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,  # 当前epoch
                    args.epochs,  # 总的epoch数
                    step,  # 当前步数
                    iter_per_epoch,  # 每个epoch的迭代次数
                    loss.item() * args.accumulation_steps,  # 损失值，乘以累积步数
                    optimizer.param_groups[-1]['lr'],  # 当前学习率
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 估计剩余时间
                    )
            )
            if (wandb is not None):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                "lr": optimizer.param_groups[-1]['lr'],
                "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60}
                )

          # 每隔一定步数保存模型
        if (step + 1) % args.save_step == 0:
            model.eval()  # 设置模型为评估模式
            ckp = f'{args.save_dir}/SFT.pth'  # 保存模型的路径

            state_dict = model.state_dict()

            torch.save(state_dict, ckp)  # 保存模型
            model.train()  # 恢复为训练模式


def init_model(llm_config):
    """
    初始化模型和分词器
    """
    # 加载预训练模型
    model = Llama(llm_config).to(args.device)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    # 加载预训练模型
    ckp = './results/pretrain.pth'
    # 加载预训练模型的参数
    state_dict = torch.load(ckp, map_location = args.device)
    # 载入模型
    model.load_state_dict(state_dict, strict = False)
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万') 
    model = model.to(args.device)
    return model, tokenizer

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # 添加各类参数，用于配置训练过程
    parser.add_argument("--save_dir", type=str, default="results")  # 保存结果的目录
    parser.add_argument("--epochs", type=int, default=1)  # 训练的轮数
    parser.add_argument("--batch_size", type=int, default=80)  # 每批次的样本数量  24G显存可用80
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # 学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 设备类型，支持cuda或cpu
    parser.add_argument("--use_wandb", type=bool,default=True)  # 是否使用wandb进行日志记录
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型，默认使用bfloat16
    parser.add_argument("--wandb_project", type=str, default="Llama-SFT")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载时的工作线程数
    parser.add_argument("--accumulation_steps", type=int, default=2)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 学习率预热的迭代次数
    parser.add_argument("--log_step", type=int, default=10)  # 每多少步记录一次日志
    parser.add_argument("--save_step", type=int, default=1000)  # 每多少步保存一次模型
    parser.add_argument('--max_seq_len', default=512, type=int)  # 输入的最大序列长度
    parser.add_argument("--data_path", type=str, default="sft.jsonl")  # 训练数据的路径
    
    # 解析命令行参数
    args = parser.parse_args()

    # 配置语言模型的参数
    lm_config = LLMConfig(max_seq_len=args.max_seq_len)

    # 创建保存目录
    args.save_dir = os.path.join(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # 每次迭代处理的token数量
    tokens_per_iter = args.batch_size * lm_config.max_seq_len

    # 随机种子
    torch.manual_seed(10246)

    # 判断设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"Llama-pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型选择自动混合精度（AMP）或不使用AMP
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    if args.use_wandb:
        import wandb

        wandb.init(project = args.wandb_project, name = args.wandb_run_name)
    else:
        wandb = None
    
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)

    # 加载数据集
    train_dataset = SFTDataset(args.data_path, tokenizer, max_length = args.max_seq_len)
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        pin_memory = True, #是否将数据复制到CUDA内存
        drop_last = False,
        shuffle = False,
        num_workers = args.num_workers, #数据加载时的工作线程数
    )

    # 梯度缩放器 用于混合精度训练
    # scaler = torch.cuda.amp.GradScaler(enabled = (args.dtype in ['float16', 'bfloat16']))
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)

    # 每个epoch的迭代次数
    iter_per_epoch = len(train_loader)

    # 训练过程
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)


