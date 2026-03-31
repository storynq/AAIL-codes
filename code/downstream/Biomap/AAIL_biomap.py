import pandas as pd
import torch
from AAIL_models import ModifiedEsmForSequenceClassification_Biomap
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
from torch.optim import AdamW
from sklearn import metrics
import numpy as np
import random
import os
import ipdb
from scipy.stats import pearsonr, spearmanr
trust_remote_code=True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def compute_metrics(p):
    predictions, labels = p   #pred:(eval_num, max_length,vocab_size) --> (100,170,30)  labels(eval_num, max_length) --> (100,170)
    labels = torch.tensor(labels)
    outputs = torch.tensor(predictions)
    outputs = torch.squeeze(outputs)
    rmse = np.sqrt(metrics.mean_squared_error(outputs, labels))
    r2 = metrics.r2_score(labels, outputs)   #r2 should be (y_true, y_pred)
    mae = metrics.mean_absolute_error(outputs, labels)
    rp = pearsonr(labels, outputs)  #rp有两个值 一个值一个p_value
    spearman = spearmanr(labels, outputs)


    return {
        "RMSE": rmse,
        "r2": r2,
        "MAE": mae,
        "rp_s": rp[0],
        "rp_p": rp[1],
        "sp_s": spearman[0],
        "sp_p": spearman[1]
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


def preprocess(batch): #这段在弄的事情是： seq在tokenizer之后长度会变，会根据最长的那个后面变多一定数量的token(就是padding) 前面也会加上起始的token 而label(1,0)的数量恒等于AA的数量 是不会变的 因此要给非token的label加上-100的标签
    sequence = ['[HEAVY]' + h + '[LIGHT]' + l for h, l in zip(batch['antibody_seq_a'], batch['antibody_seq_b'])]
    tokenizer = EsmTokenizer.from_pretrained('/home/hnq/test_code/Model_ESM/Tokenizer')
    special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
    tokenizer.add_special_tokens(special_tokens)

    t_inputs = tokenizer(sequence, max_length=510, padding="max_length", truncation=True)                   #biomap的抗体序列也在200+到400+接近500的情况 大部分都是400+的

    antigen_tokenizer = BertTokenizer.from_pretrained('/home/hnq/test_code/Model_Prot')
    antigen_seq = [' '.join(seq) for seq in batch['antigen_seq']]
    antigen_inputs = antigen_tokenizer(antigen_seq,max_length=512, padding="max_length", truncation=True)   #这里抗原序列长度分布很不好 最短的可以只有9个氨基酸 最长的有1267 256大概能覆盖2/3的抗原 意味着会有1/3的抗原有信息丢失

    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask
    batch['antigen_input_ids'] = antigen_inputs.input_ids
    batch['antigen_attention_mask'] = antigen_inputs.attention_mask
    batch['labels'] = batch['delta_g']

    return batch


np.random.seed(21)

data_file = pd.read_csv('/home/hnq/test_code/biomap.csv')
datasets = Dataset.from_pandas(data_file[['antibody_seq_a','antibody_seq_b','antigen_seq', 'delta_g']])

n_samples = len(datasets)
idx_full = np.arange(n_samples)
np.random.shuffle(idx_full)
validation_split = 0.2  
test_split = 0.2        
len_valid = int(n_samples * validation_split)
len_test = int(n_samples * test_split)

valid_idx = idx_full[0:len_valid]
test_idx = idx_full[len_valid: (len_valid + len_test)]
train_idx = np.delete(idx_full, np.arange(0, len_valid + len_test))

train_dataset = datasets.select(train_idx.tolist())
valid_dataset = datasets.select(valid_idx.tolist())
test_dataset = datasets.select(test_idx.tolist())

dataset_final = DatasetDict({
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
})
print(train_idx)
print(valid_idx)
print(test_idx)
dataset_tokenized = dataset_final.map(
    preprocess, 
    batched=True,
    batch_size=200,
    remove_columns=['antibody_seq_a','antibody_seq_b','antigen_seq', 'delta_g']
)

batch_size = 32
RUN_ID = "Biomap_AttnESM_seed21"
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
    print(f"BERT参数: {len(bert_params)} 组")
    print(f"LoRA参数: {len(lora_params)} 组") 
    print(f"扩散模块参数: {len(diffusion_params)} 组")
    print(f"分类层参数: {len(cls_params)} 组")

    optimizer = AdamW([
        {'params': bert_params, 'lr': 4e-5},  # BERT 层使用较小的学习率
        {'params': lora_params, 'lr': 4e-5},
        {'params': diffusion_params, 'lr': 1e-4},
        {'params': cls_params, 'lr': 2e-4}   # 自定义分类层使用较大的学习率
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
    num_train_epochs=300,
    warmup_ratio=0.1, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    weight_decay=0.01,
    lr_scheduler_type='cosine_with_restarts',
    metric_for_best_model='rp_s', # name of the metric here should correspond to metrics defined in compute_metrics
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


model2 = ModifiedEsmForSequenceClassification_Biomap(config = config, config2 = antigen_config, word_embedding = embedding_layer, word_embedding2=embedding_layer2, num_labels=1, ag_step=0, ab_step=1)

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

trainer.train()

pred = trainer.predict(
    dataset_tokenized['test']
)

y_pred = np.concatenate(pred[0])
y_true = pred[1]

test_df = pd.DataFrame({'antibody_a': dataset_final['test']['antibody_seq_a'] ,
                        'antibody_b': dataset_final['test']['antibody_seq_b'],
                        'antigen': dataset_final['test']['antigen_seq'],
                        'y_true': list(y_true.flatten()),
                        'y_pred': list(y_pred.flatten())})
test_df.to_csv('./PairedESM_seed21_results.csv', index=False)
print(pred.metrics)