import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512,
                 image_special_token='@' * 196):

        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]
        
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_tensor = torch.stack([MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes])
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_tensor



class GeoQADataset(Dataset):
    """
    GeoQA 数据集，支持 SFT 和 GRPO 两种模式。

    数据来源: scripts/build_sft_data.py 生成的 Parquet 文件
      - sft_train.parquet / sft_test.parquet: conversations + image_bytes
      - rl_base.parquet:                      conversations + image_bytes + answer

    mode='sft':
        返回 (X, Y, loss_mask, pixel_values)
        - X: 输入 token ids [seq_len]
        - Y: 目标 token ids [seq_len]（assistant 回复部分）
        - loss_mask: 1 表示该位置参与损失计算，0 表示忽略
        - pixel_values: [1, C, H, W]

    mode='grpo':
        返回 {'prompt_ids', 'pixel_values', 'answer'}
        - prompt_ids: 只含 user 问题的 token ids（变长）
        - pixel_values: [1, C, H, W]
        - answer: 正确答案字母 A/B/C/D

    image_token_str:
        conversations 里的 <image> 占位符会被替换为这个字符串再做 tokenize。
        InternVL3-8B 默认值由训练脚本传入：
            '<img>' + '<IMG_CONTEXT>' * num_image_token + '</img>'
        若不传则保留 <image> 原样（用于测试/其他模型）。
    """

    # InternVL3-8B 的 assistant 回复边界 token
    # apply_chat_template 生成的格式：
    #   <|im_start|>assistant\n{content}<|im_end|>\n
    ASSISTANT_START = '<|im_start|>assistant\n'
    TURN_END        = '<|im_end|>\n'

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        image_processor,          # InternVL 的 CLIPImageProcessor / InternVLProcessor
        mode: str = 'sft',        # 'sft' | 'grpo'
        max_length: int = 1024,
        image_token_str: str = '<image>',  # 替换 <image> 占位符的实际 token 串
    ):
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.max_length = max_length
        self.image_token_str = image_token_str

        # 预计算边界 token ids（用于定位 assistant 回复区间）
        self._assistant_start_ids = tokenizer(
            self.ASSISTANT_START, add_special_tokens=False
        ).input_ids
        self._turn_end_ids = tokenizer(
            self.TURN_END, add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.table)

    def _load_image(self, index: int):
        """从 Parquet 读取图像并预处理为 pixel_values tensor"""
        image_bytes = self.table['image_bytes'][index].as_py()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # image_processor 返回 dict，取 pixel_values [1, C, H, W]
        pixel_values = self.image_processor(
            images=image, return_tensors='pt'
        )['pixel_values']  # [1, C, H, W]
        return pixel_values

    def _build_loss_mask(self, input_ids: list) -> list:
        """
        生成 loss_mask：只对 assistant 回复部分置 1。

        InternVL3 chat template 格式：
          <|im_start|>system\\n{sys}<|im_end|>\\n
          <|im_start|>user\\n{question}<|im_end|>\\n
          <|im_start|>assistant\\n{answer}<|im_end|>\\n
                                 ^^^^^^^^^^^^^^^^^ 只有这段参与损失
        """
        mask = [0] * len(input_ids)
        start_ids = self._assistant_start_ids
        end_ids   = self._turn_end_ids
        n = len(input_ids)
        i = 0
        while i < n:
            # 找 ASSISTANT_START
            if input_ids[i: i + len(start_ids)] == start_ids:
                content_start = i + len(start_ids)
                j = content_start
                # 找对应的 TURN_END
                while j < n:
                    if input_ids[j: j + len(end_ids)] == end_ids:
                        break
                    j += 1
                content_end = min(j + len(end_ids), n)
                for k in range(content_start, content_end):
                    mask[k] = 1
                i = content_end
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        pixel_values  = self._load_image(index)

        if self.mode == 'sft':
            return self._get_sft_item(conversations, pixel_values)
        elif self.mode == 'grpo':
            return self._get_grpo_item(conversations, pixel_values, index)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

    def _get_sft_item(self, conversations, pixel_values):
        """SFT 模式：返回 (X, Y, loss_mask, pixel_values)"""
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        # 将 <image> 占位符替换为模型实际使用的 image token 串
        # InternVL3: '<img><IMG_CONTEXT>×256</img>'
        prompt = prompt.replace('<image>', self.image_token_str)
        input_ids = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids[:self.max_length]

        # 右填充到 max_length
        pad_len   = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        loss_mask = self._build_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:],  dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.float)

        return X, Y, mask, pixel_values

    def _get_grpo_item(self, conversations, pixel_values, index: int):
        """GRPO 模式：返回 dict{prompt_ids, pixel_values, answer}"""
        # conversations 只含 user turn（来自 rl_base.parquet）
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,  # 加上 <|im_start|>assistant\n，让模型续写
        )
        prompt = prompt.replace('<image>', self.image_token_str)
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids[:self.max_length]

        answer = self.table['answer'][index].as_py()

        return {
            'prompt_ids':   torch.tensor(prompt_ids, dtype=torch.long),
            'pixel_values': pixel_values,
            'answer':       answer,
        }


# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        t = pq.read_table(path); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            ax[i].imshow(Image.open(io.BytesIO(t['image_bytes'][i].as_py()))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{len(t)}条')
