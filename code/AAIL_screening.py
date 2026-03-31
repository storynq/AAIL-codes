import pandas as pd
import torch
from total_models import ModifiedEsmForAntibodyScreen_V0
from datasets import (
    Dataset,
)
from transformers import (
    Trainer,
    TrainingArguments,
)

from datasets import DatasetDict
from transformers import (
    EsmTokenizer,
    EsmModel,
    BertTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
    EsmConfig,
    BertConfig,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import random
import os
import ipdb
from sklearn import metrics
trust_remote_code=True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def compute_metrics(p):
    
    predictions, labels = p

    outputs = torch.tensor(predictions)  
    labels = torch.tensor(labels)

    # 获取预测的类别ID（硬预测）
    pred_ids = torch.argmax(outputs, dim=-1)  # 形状与 labels 一致

    # 计算 softmax 概率
    probs = torch.softmax(outputs, dim=-1)  # 形状与 outputs 一致，如 (样本数, 2)
    # 提取正类别（索引1）的概率
    positive_probs = probs[..., 1]  # 形状比 outputs 少一维，例如 (样本数,) 或 (样本数, 序列长度)

    # 计算基础指标
    acc = metrics.accuracy_score(labels, pred_ids)
    f1 = metrics.f1_score(labels, pred_ids, average='binary', zero_division=0)
    precision = metrics.precision_score(labels, pred_ids, average='binary', zero_division=0)
    recall = metrics.recall_score(labels, pred_ids, average='binary', zero_division=0)

    # 计算 ROC-AUC 和 PR-AUC（需要正类概率）
    roc_auc = 0.0
    pr_auc = 0.0
    # 确保数据中至少包含两个类别，否则 AUC 计算会报错
    if len(np.unique(labels)) == 2:
        try:
            roc_auc = metrics.roc_auc_score(labels, positive_probs)
        except Exception:
            roc_auc = 0.0
        try:
            pr_auc = metrics.average_precision_score(labels, positive_probs)
        except Exception:
            pr_auc = 0.0
    else:
        # 如果验证集中只有一个类别，AUC 无法定义，可记录为0或跳过
        print("警告: 验证集中只出现了一个类别，无法计算 AUC 指标。")

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def set_seed(seed: int = 42):
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def preprocess(batch):
    sequences = []
    
    for h, l in zip(batch['VH'], batch['VL']):
        # 处理None值和NaN值
        heavy_part = '' if (h is None or pd.isna(h)) else str(h)
        light_part = '' if (l is None or pd.isna(l)) else str(l)
        
        if heavy_part and light_part:
            seq = '[HEAVY]' + heavy_part + '[LIGHT]' + light_part
        elif heavy_part:  
            seq = '[HEAVY]' + heavy_part
        elif light_part:  
            seq = '[LIGHT]' + light_part
        else: 
            seq = ''
        sequences.append(seq)
        
    tokenizer = EsmTokenizer.from_pretrained('/home/hnq/test_code/Model_ESM/Tokenizer')
    special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
    tokenizer.add_special_tokens(special_tokens)

    t_inputs = tokenizer(sequences, max_length=292, padding="max_length", truncation=True)                   

    antigen_tokenizer = BertTokenizer.from_pretrained('/home/hnq/test_code/Model_Prot')
    antigen_seq = [' '.join(seq) for seq in batch['Antigen']]
    antigen_inputs = antigen_tokenizer(antigen_seq,max_length=1536, padding="max_length", truncation=True)   

    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['antigen_input_ids'] = antigen_inputs.input_ids
    batch['antigen_attention_mask'] = antigen_inputs.attention_mask
    batch['labels'] = batch['label']

    return batch


np.random.seed(42)

# 数据文件路径
datasets = {
    'train': {
        'positive': '/data/hnq/Screening_dataset/dataset2/train.xlsx',
        'negative': '/data/hnq/Screening_dataset/dataset2/negative_train.xlsx'
    },
    'validation': {
        'positive': '/data/hnq/Screening_dataset/dataset2/validation.xlsx', 
        'negative': '/data/hnq/Screening_dataset/dataset2/negative_validation.xlsx'
    },
    'test': {
        'positive': '/data/hnq/Screening_dataset/dataset2/test.xlsx',
        'negative': '/data/hnq/Screening_dataset/dataset2/negative_test.xlsx'
    }
}

# 读取并合并每个数据集的正负样本
merged_datasets = {}
for split_name, files in datasets.items():
    # 读取正样本
    pos_df = pd.read_excel(files['positive'])
    pos_df['label'] = 1  # 正样本标签
    
    # 读取负样本  
    neg_df = pd.read_excel(files['negative'])
    neg_df['label'] = 0  # 负样本标签
    
    # 合并并打乱
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    merged_datasets[split_name] = combined_df
    
    print(f"{split_name}: 正样本 {len(pos_df)}, 负样本 {len(neg_df)}, 总计 {len(combined_df)}")

# 创建Dataset对象
dataset_dict = DatasetDict({
    split_name: Dataset.from_pandas(df) 
    for split_name, df in merged_datasets.items()
})


# 应用预处理函数（假设preprocess函数已定义）
dataset_tokenized = dataset_dict.map(
    preprocess, 
    batched=True,
    batch_size=1000,
    remove_columns=merged_datasets['train'].columns.tolist()  # 移除原始列
)

print(f"\n最终数据集结构:")
print(dataset_tokenized)
batch_size = 16
RUN_ID = "811_33_newlr"
SEED = 0

def custom_optimizer(model):
    # 分离 BERT 参数和自定义分类层参数
    bert_params = []
    cls_params = []
    lora_params = []
    diffusion_params = []
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            bert_params.append(param)
        elif name.startswith('antigen'):
            lora_params.append(param)
        elif any(keyword in name for keyword in [
            'feature_extracion','ag_feature_extraction', 'step_embedding',
            'rep0_mlp', 'fusion_mlp', 'correction_mlp', 'conv'
        ]):
            diffusion_params.append(param)
        else:
            cls_params.append(param)
    
    # 为不同参数组设置不同学习率
    print(f"Antibody参数: {len(bert_params)} 组")
    print(f"Antigen参数: {len(lora_params)} 组") 
    print(f"扩散模块参数: {len(diffusion_params)} 组")
    print(f"分类层参数: {len(cls_params)} 组")

    optimizer = AdamW([
        {'params': bert_params, 'lr': 2e-5},  # BERT 层使用较小的学习率
        {'params': lora_params, 'lr': 2e-5},
        {'params': diffusion_params, 'lr': 5e-5},
        {'params': cls_params, 'lr': 1e-4}   # 自定义分类层使用较大的学习率
    ],
    betas=(0.9, 0.98),
    eps=1e-6)
    return optimizer

args = TrainingArguments(
    f"/data/hnq/model_save/{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=50,
    warmup_ratio=0.1, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    weight_decay=0.01,
    lr_scheduler_type='cosine_with_restarts',
    metric_for_best_model='pr_auc', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 创建优化器
        self.optimizer = custom_optimizer(self.model)
        # 创建学习率调度器
        self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        
tokenizer = EsmTokenizer.from_pretrained('/home/hnq/test_code/Model_ESM/Tokenizer')
special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
tokenizer.add_special_tokens(special_tokens)

model = EsmModel.from_pretrained('/home/hnq/PairedESM/Code/ESM_pretrain/checkpoint-500000', add_pooling_layer=False)
state_dict = model.state_dict()   ## 197 = 5 + 16*12
new_state_dict = {f'bert.{key}': value for key, value in state_dict.items()}  ##所用参数前面多了一个bert 是我在模型里命名导致的 所以给它加上去
config = EsmConfig.from_pretrained('/home/hnq/PairedESM/Code/ESM_pretrain/checkpoint-500000')

#提取word_embedding层
embedding_layer = model.embeddings.word_embeddings

for param in embedding_layer.parameters():
    param.requires_grad = False

# 抗原state_dict导入
antigen_tokenizer = BertTokenizer.from_pretrained('/home/hnq/test_code/Model_Prot')
antigen_model = BertModel.from_pretrained('/home/hnq/test_code/Model_Prot')
antigen_state_dict = antigen_model.state_dict()
antigen_new_state = {f'antigen.{key}': value for key, value in antigen_state_dict.items()} 
antigen_config = BertConfig.from_pretrained('/home/hnq/test_code/Model_Prot')

embedding_layer2 = antigen_model.embeddings.word_embeddings

for param in embedding_layer2.parameters():
    param.requires_grad = False


model2 = ModifiedEsmForAntibodyScreen_V0(config = config, config2 = antigen_config, word_embedding = embedding_layer, word_embedding2=embedding_layer2, num_labels=2, ag_step=3, ab_step=3)

model2.load_state_dict(new_state_dict, strict = False)
model2.load_state_dict(antigen_new_state, strict = False)

# for name, param in model2.named_parameters():               # Freeze antigen layers
#     if name.startswith('antigen'):
#         param.requires_grad = False

total_num = sum(p.numel() for p in model2.parameters())
trainable_num = sum(p.numel() for p in model2.parameters() if p.requires_grad)

print('model:', model2)
print('total_num:', total_num)
print('trainable_num:', trainable_num)

trainer = CustomTrainer(
    model = model2,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset_tokenized['train'],
    eval_dataset=dataset_tokenized['validation'], 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
)

# checkpoint_path = "/data/hnq/model_save/811dataset_0/checkpoint-17728"
trainer.train()

pred = trainer.predict(
    dataset_tokenized['test']
)

print(pred.metrics)