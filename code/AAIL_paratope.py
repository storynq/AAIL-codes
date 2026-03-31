import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from datasets import Dataset
import torch
from collections import defaultdict
import re
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
from total_models import  ModifiedEsmForTokenClassification_V2
from torch.optim import AdamW
from sklearn import metrics
import random
import os
import ipdb
from scipy.stats import pearsonr, spearmanr

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def compute_metrics(p, save_sample_metrics=False, output_csv="sample_metrics.csv", identifiers = None):
    predictions, labels = p   
    probs = torch.softmax(torch.from_numpy(predictions), dim=2).numpy()
    
    antibody_labels = labels[0]
    antigen_labels = labels[1]
    antibody_pred = probs[:, :256, 1]
    antigen_pred = probs[:, 256:, 1]
    
    all_sample_metrics = []
    
    antibody_metrics = {'auc': [], 'pr-auc': [], 'precision': [], 'recall': [], 'f1': [], 'pos_rate': []}
    antigen_metrics = {'auc': [], 'pr-auc': [], 'precision': [], 'recall': [], 'f1': [], 'pos_rate': []}
    

    for i in range(antibody_pred.shape[0]):
        sample_metrics = {'sample_id': i}

        seq_probs = antibody_pred[i]
        seq_labels = antibody_labels[i]
        valid_mask = (seq_labels != -100)
        valid_probs = seq_probs[valid_mask]
        valid_labels = seq_labels[valid_mask].astype(int)
        
        if len(valid_labels) > 0:
            pos_count = np.sum(valid_labels == 1)
            pos_rate = pos_count / len(valid_labels)
            antibody_metrics['pos_rate'].append(pos_rate)
            sample_metrics['ab_pos_rate'] = pos_rate
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(valid_labels, valid_probs)
                antibody_metrics['auc'].append(roc_auc)
                sample_metrics['ab_auc'] = roc_auc
            except ValueError:
                sample_metrics['ab_auc'] = np.nan
            
            # PR AUC
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(valid_labels, valid_probs)
                pr_auc = auc(recall_curve, precision_curve)
                antibody_metrics['pr-auc'].append(pr_auc)
                sample_metrics['ab_pr_auc'] = pr_auc
            except ValueError:
                sample_metrics['ab_pr_auc'] = np.nan
            

            binary_preds = (valid_probs >= 0.5).astype(int)
            precision_val = precision_score(valid_labels, binary_preds, zero_division=0)
            recall_val = recall_score(valid_labels, binary_preds, zero_division=0)
            f1_val = f1_score(valid_labels, binary_preds, zero_division=0)
            
            antibody_metrics['precision'].append(precision_val)
            antibody_metrics['recall'].append(recall_val)
            antibody_metrics['f1'].append(f1_val)
            
            sample_metrics['ab_precision'] = precision_val
            sample_metrics['ab_recall'] = recall_val
            sample_metrics['ab_f1'] = f1_val
            sample_metrics['ab_pred_pos_rate'] = np.mean(binary_preds) 
        else:

            for key in ['ab_pos_rate', 'ab_auc', 'ab_pr_auc', 'ab_precision', 'ab_recall', 'ab_f1', 'ab_pred_pos_rate']:
                sample_metrics[key] = np.nan
        

        seq_probs = antigen_pred[i]
        seq_labels = antigen_labels[i]
        valid_mask = (seq_labels != -100)
        valid_probs = seq_probs[valid_mask]
        valid_labels = seq_labels[valid_mask].astype(int)
        
        if len(valid_labels) > 0:
            pos_count = np.sum(valid_labels == 1)
            pos_rate = pos_count / len(valid_labels)
            antigen_metrics['pos_rate'].append(pos_rate)
            sample_metrics['ag_pos_rate'] = pos_rate
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(valid_labels, valid_probs)
                antigen_metrics['auc'].append(roc_auc)
                sample_metrics['ag_auc'] = roc_auc
            except ValueError:
                sample_metrics['ag_auc'] = np.nan
            
            # PR AUC
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(valid_labels, valid_probs)
                pr_auc = auc(recall_curve, precision_curve)
                antigen_metrics['pr-auc'].append(pr_auc)
                sample_metrics['ag_pr_auc'] = pr_auc
            except ValueError:
                sample_metrics['ag_pr_auc'] = np.nan
            
            binary_preds = (valid_probs >= 0.5).astype(int)
            precision_val = precision_score(valid_labels, binary_preds, zero_division=0)
            recall_val = recall_score(valid_labels, binary_preds, zero_division=0)
            f1_val = f1_score(valid_labels, binary_preds, zero_division=0)
            
            antigen_metrics['precision'].append(precision_val)
            antigen_metrics['recall'].append(recall_val)
            antigen_metrics['f1'].append(f1_val)
            
            sample_metrics['ag_precision'] = precision_val
            sample_metrics['ag_recall'] = recall_val
            sample_metrics['ag_f1'] = f1_val
            sample_metrics['ag_pred_pos_rate'] = np.mean(binary_preds) 
        else:

            for key in ['ag_pos_rate', 'ag_auc', 'ag_pr_auc', 'ag_precision', 'ag_recall', 'ag_f1', 'ag_pred_pos_rate']:
                sample_metrics[key] = np.nan
        
        sample_metrics['best_metric'] = sample_metrics.get('ab_f1', 0) + 2 * sample_metrics.get('ag_f1', 0)
        
        all_sample_metrics.append(sample_metrics)
    

    if save_sample_metrics:
        df = pd.DataFrame(all_sample_metrics)
        
        if identifiers is not None:
            df.insert(0, "pdb_id", identifiers)

        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row['sample_id'] = 'MEAN'
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        df.to_csv(output_csv, index=False)
    
    def safe_mean(values, default=0):
        return np.nanmean(values) if values else default
    
    ab_auc_avg = safe_mean(antibody_metrics['auc'], np.nan)
    ab_pr_auc_avg = safe_mean(antibody_metrics['pr-auc'])
    ab_pre_avg = safe_mean(antibody_metrics['precision'])
    ab_recall_avg = safe_mean(antibody_metrics['recall'])
    ab_f1_avg = safe_mean(antibody_metrics['f1'])
    ab_pos_rate_avg = safe_mean(antibody_metrics['pos_rate'])
    
    ag_auc_avg = safe_mean(antigen_metrics['auc'], np.nan)
    ag_pr_auc_avg = safe_mean(antigen_metrics['pr-auc'])
    ag_pre_avg = safe_mean(antigen_metrics['precision'])
    ag_recall_avg = safe_mean(antigen_metrics['recall'])
    ag_f1_avg = safe_mean(antigen_metrics['f1'])
    ag_pos_rate_avg = safe_mean(antigen_metrics['pos_rate'])
    
    best_metric = ab_f1_avg + 2 * ag_f1_avg 
    
    return {
        'AUC(antibody)': ab_auc_avg,
        'AUC(antigen)': ag_auc_avg,
        'PR_AUC(antibody)': ab_pr_auc_avg,
        'PR_AUC(antigen)': ag_pr_auc_avg,
        'pre(antibody)': ab_pre_avg,
        'pre(antigen)': ag_pre_avg,
        'recall(antibody)': ab_recall_avg,
        'recall(antigen)': ag_recall_avg,
        'f1(antibody)': ab_f1_avg,
        'f1(antigen)': ag_f1_avg,
        'Pos_rate(antibody)': ab_pos_rate_avg,
        'Pos_rate(antigen)': ag_pos_rate_avg,
        'best_metric': best_metric
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

def prepare_dataset(samples):
    sequence = ['[HEAVY]' + h + '[LIGHT]' + l for h, l in zip(samples['antibody_h'], samples['antibody_l'])]
    antibody_mask = ['0' + h + '0' + l for h,l in zip(samples['H_paratope'], samples['L_paratope'])]

    tokenizer = EsmTokenizer.from_pretrained('/home/hnq/test_code/Model_ESM/Tokenizer')
    special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
    tokenizer.add_special_tokens(special_tokens)
    t_inputs = tokenizer(sequence , max_length=256, padding="max_length", truncation=True)   

    antigen_tokenizer = BertTokenizer.from_pretrained('/home/hnq/test_code/Model_Prot')
    antigen_seq = [' '.join(seq) for seq in samples['antigen_seq']]
    antigen_inputs = antigen_tokenizer(antigen_seq,max_length=512, padding="max_length", truncation=True) 

    antigen_labels = []
    antibody_labels = []
    for index, labels in enumerate(antibody_mask):

        antibody_input_length = len(t_inputs.input_ids[index])
        antigen_input_length = len(antigen_inputs.input_ids[index])

        paratope_label_length = len(antibody_mask[index])
        epitope_label_length = len(samples['antigen_epitope'][index])

        paratope_padding_num = max(1, antibody_input_length-paratope_label_length-1)
        epitope_padding_num = max(1, antigen_input_length-epitope_label_length-1)

        paratope_labels = list(map(int, labels))
        epitope_labels = list(map(int, samples['antigen_epitope'][index]))

        if len(paratope_labels) > 256 -2 :
            paratope_labels = paratope_labels[:254]
        if len(epitope_labels) > 512 -2:
            epitope_labels = epitope_labels[:510]

        paratope_labels_padded = [-100] + paratope_labels + [-100] * paratope_padding_num
        epitope_labels_padded = [-100] + epitope_labels + [-100] * epitope_padding_num

        assert len(paratope_labels_padded) == antibody_input_length, \
        f"Antibody don't align, {len(labels)}"

        assert len(epitope_labels_padded) == antigen_input_length, \
        f"Antigen don't align, {len(labels)}"

        antigen_labels.append(epitope_labels_padded)
        antibody_labels.append(paratope_labels_padded)

    samples['input_ids'] = t_inputs.input_ids
    samples['attention_mask'] = t_inputs.attention_mask
    samples['antigen_input_ids'] = antigen_inputs.input_ids
    samples['antigen_attention_mask'] = antigen_inputs.attention_mask
    samples['labels'] = antibody_labels
    samples['antigen_labels'] = antigen_labels

    return samples

train_file = pd.read_csv('/data/hnq/epi_train.csv')
train_data = Dataset.from_pandas(train_file[['pdb_id','antigen_seq','antibody_h','antibody_l','antigen_epitope','H_paratope','L_paratope']])

n_samples = len(train_data)
idx_full = np.arange(n_samples)

seed = 42
np.random.seed(seed)
np.random.shuffle(idx_full)
validation_split = 0.25         
len_valid = int(n_samples * validation_split)

valid_idx = idx_full[0:len_valid]
train_idx = np.delete(idx_full, np.arange(0, len_valid))

train_dataset = train_data.select(train_idx.tolist())
valid_dataset = train_data.select(valid_idx.tolist())

test_file = pd.read_csv('/data/hnq/epi_test.csv')
test_dataset = Dataset.from_pandas(test_file[['pdb_id','antigen_seq','antibody_h','antibody_l','antigen_epitope','H_paratope','L_paratope']])

train_tokenized = train_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=200,
    remove_columns=['antigen_seq', 'antibody_h', 'antibody_l', 'antigen_epitope', 'H_paratope', 'L_paratope']
)
valid_tokenized = valid_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=200,
    remove_columns=['antigen_seq', 'antibody_h', 'antibody_l', 'antigen_epitope', 'H_paratope', 'L_paratope']
)
test_tokenized = test_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=200,
    remove_columns=['antigen_seq', 'antibody_h', 'antibody_l', 'antigen_epitope', 'H_paratope', 'L_paratope']
)

batch_size = 32
RUN_ID = "Epi&Para131_Bench_AttnESM33"
SEED = 0

def custom_optimizer(model):
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
            'ab_feature_extraction', 'ag_feature_extraction', 'step_embedding',
            'rep0_mlp', 'fusion_mlp', 'correction_mlp', 'conv'
        ]):
            diffusion_params.append(param)
        else:
            cls_params.append(param)

    optimizer = AdamW([
        {'params': bert_params, 'lr': 4e-5},  
        {'params': lora_params, 'lr': 4e-5},
        {'params': diffusion_params, 'lr': 2e-3},
        {'params': cls_params, 'lr': 2e-3}   
    ],
    betas=(0.9, 0.98),
    eps=1e-6)
    return optimizer

args = TrainingArguments(
    f"/data/hnq/model_save/{RUN_ID}_{SEED}", 
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1200,
    warmup_ratio=0.1, # 0, 0.05, 0.1 .... 
    load_best_model_at_end=True,
    weight_decay=0.01,
    lr_scheduler_type='cosine_with_restarts',
    metric_for_best_model='best_metric', 
    logging_strategy='epoch',
    seed=SEED
)

set_seed(SEED)

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = custom_optimizer(self.model)
        self.lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

tokenizer = EsmTokenizer.from_pretrained('/home/hnq/test_code/Model_ESM/Tokenizer')
special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
tokenizer.add_special_tokens(special_tokens)

model = EsmModel.from_pretrained('/home/hnq/PairedESM/Code/ESM_pretrain/checkpoint-500000')
state_dict = model.state_dict()   ## 197 = 5 + 16*12
new_state_dict = {f'bert.{key}': value for key, value in state_dict.items()}  
config = EsmConfig.from_pretrained('/home/hnq/PairedESM/Code/ESM_pretrain/checkpoint-500000')


embedding_layer = model.embeddings.word_embeddings
for param in embedding_layer.parameters():
    param.requires_grad = False


antigen_tokenizer = BertTokenizer.from_pretrained('/home/hnq/test_code/Model_Prot')
antigen_model = BertModel.from_pretrained('/home/hnq/test_code/Model_Prot')
antigen_state_dict = antigen_model.state_dict()
antigen_new_state = {f'antigen.{key}': value for key, value in antigen_state_dict.items()} 
antigen_config = BertConfig.from_pretrained('/home/hnq/test_code/Model_Prot')
embedding_layer2 = antigen_model.embeddings.word_embeddings

for param in embedding_layer2.parameters():
    param.requires_grad = False

model2 = ModifiedEsmForTokenClassification_V2(config = config, config2 = antigen_config, word_embedding = embedding_layer, word_embedding2=embedding_layer2, num_labels=2, ab_step=2, ag_step=1)

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
    train_dataset=train_tokenized ,
    eval_dataset=valid_tokenized , 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
)

trainer.train()

pred = trainer.predict(test_tokenized)
test_pdb_ids = test_tokenized['pdb_id']

metrics = compute_metrics(
    (pred.predictions, pred.label_ids), 
    save_sample_metrics=True, 
    output_csv="test33_metrics.csv",
    identifiers=test_pdb_ids
)

print(metrics)
