import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):
        """
        预训练数据集
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """
        从文件逐行读取数据 并解析为JSON对象
        """
        samples = []
        with open(path, 'r', encoding = 'utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # 根据索引获取单个样本数据
        sample = self.samples[index]
        # 构建输入文本  加上起始符和结束符
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        # 用tokenizer将文本编码为token id 固定最大长度 进行padding和截断
        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        # 获取输入token IDs
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        
        # X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length  
        # 加载数据
        self.samples = self.load_data(jsonl_path)
        # 开始和结束的token ID
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
     
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding = 'utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 使用分词器提供的模板方法将对话转换为模型输入
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False
        )
    
    def _generate_loss_mask(self, input_ids):
        # 生成loss掩码 将response外的token标记为0
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 检查当前位置是否匹配开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 记录开始位置，排除 bos 部分
                start = i + len(self.bos_id)
                end = start
                # 从开始位置向后查找结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将开始标记之后到结束标记位置之间的 token 标记为 1（参与损失计算）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 更新索引：跳过整个对话部分（包括结束标记）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


    def __getitem__(self, index):
        # 根据索引获取单个样本
        sample = self.samples[index]
        prompt = self._create_chat_prompt(sample['conversations'])
        # 将对话进行编码并限制最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 若不足最大长度则填充
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 掩码
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype = torch.long)
        Y = torch.tensor(input_ids[1:], dtype = torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype = torch.long)
        return X, Y, loss_mask
