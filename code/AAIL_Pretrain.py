from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EsmForMaskedLM,
    EsmConfig,
    EsmTokenizer,
    AutoTokenizer
)
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import os
import pandas as pd
import ipdb

trust_remote_code=True

def preprocess_and_split_data(paired_csv_path, vh_csv_path, vl_csv_path, 
                                      test_size=0.1, seed=42,
                                      paired_factor=1.0, vh_factor=0.3, vl_factor=0.3):
    
    train_datasets = []
    test_datasets = []
    
    if os.path.exists(paired_csv_path):

        paired_df = pd.read_csv(paired_csv_path)
        if paired_factor < 1.0:
            paired_df = paired_df.sample(frac=paired_factor, random_state=seed)

        paired_data = process_paired_data(paired_df)
        paired_dataset = Dataset.from_pandas(pd.DataFrame(paired_data))
        
        paired_split = paired_dataset.train_test_split(test_size=test_size, seed=seed)
        train_datasets.append(paired_split['train'])
        test_datasets.append(paired_split['test'])

    if os.path.exists(vh_csv_path):

        vh_df = pd.read_csv(vh_csv_path)
        if vh_factor < 1.0:
            vh_df = vh_df.sample(frac=vh_factor, random_state=seed)

        vh_data = process_vh_data(vh_df)
        vh_dataset = Dataset.from_pandas(pd.DataFrame(vh_data))
        
        vh_split = vh_dataset.train_test_split(test_size=test_size, seed=seed)
        train_datasets.append(vh_split['train'])
        test_datasets.append(vh_split['test'])


    if os.path.exists(vl_csv_path):

        vl_df = pd.read_csv(vl_csv_path)
        if vl_factor < 1.0:
            vl_df = vl_df.sample(frac=vl_factor, random_state=seed)

        vl_data = process_vl_data(vl_df)
        vl_dataset = Dataset.from_pandas(pd.DataFrame(vl_data))
        
        vl_split = vl_dataset.train_test_split(test_size=test_size, seed=seed)
        train_datasets.append(vl_split['train'])
        test_datasets.append(vl_split['test'])
    
    if train_datasets and test_datasets:
        combined_train = concatenate_datasets(train_datasets)
        combined_test = concatenate_datasets(test_datasets)
        
        train_types = {}
        test_types = {}
        
        for item in combined_train:
            data_type = item['data_type']
            train_types[data_type] = train_types.get(data_type, 0) + 1
            
        for item in combined_test:
            data_type = item['data_type']
            test_types[data_type] = test_types.get(data_type, 0) + 1
        
        return {
            'train': combined_train,
            'test': combined_test
        }
    else:
        raise ValueError("NO DATA")

def process_paired_data(df):
    data = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('sequence_alignment_aa_heavy')) or pd.isna(row.get('sequence_alignment_aa_light')):
            continue
            
        combined_sequence = f"[HEAVY]{row['sequence_alignment_aa_heavy']}[LIGHT]{row['sequence_alignment_aa_light']}"
        

        region_labels = create_paired_region_labels(row)
        data.append({
            'combined_sequence': combined_sequence,
            'region_labels': region_labels,
            'data_type': 'paired'
        })
    
    return data

def process_vh_data(df):
    data = []
    for _, row in df.iterrows():
        if pd.isna(row.get('sequence_alignment_aa')):
            continue
            
        combined_sequence = f"[HEAVY]{row['sequence_alignment_aa']}"
        region_labels = create_single_region_labels(row)
        data.append({
            'combined_sequence': combined_sequence,
            'region_labels': region_labels,
            'data_type': 'vh'
        })
    return data

def process_vl_data(df):

    data = []
    for _, row in df.iterrows():
        if pd.isna(row.get('sequence_alignment_aa')):
            continue
            
        combined_sequence = f"[LIGHT]{row['sequence_alignment_aa']}"
        region_labels = create_single_region_labels(row)
        data.append({
            'combined_sequence': combined_sequence,
            'region_labels': region_labels,
            'data_type': 'vl'
        })
    return data

def create_paired_region_labels(row):
    region_labels = []

    region_labels.append('special')
    heavy_regions = []
    if not pd.isna(row.get('fwr1_aa_heavy')):
        heavy_regions.extend(['fwr1'] * len(row['fwr1_aa_heavy']))
    if not pd.isna(row.get('cdr1_aa_heavy')):
        heavy_regions.extend(['cdr1'] * len(row['cdr1_aa_heavy']))
    if not pd.isna(row.get('fwr2_aa_heavy')):
        heavy_regions.extend(['fwr2'] * len(row['fwr2_aa_heavy']))
    if not pd.isna(row.get('cdr2_aa_heavy')):
        heavy_regions.extend(['cdr2'] * len(row['cdr2_aa_heavy']))
    if not pd.isna(row.get('fwr3_aa_heavy')):
        heavy_regions.extend(['fwr3'] * len(row['fwr3_aa_heavy']))
    if not pd.isna(row.get('cdr3_aa_heavy')):
        heavy_regions.extend(['cdr3'] * len(row['cdr3_aa_heavy']))
    if not pd.isna(row.get('fwr4_aa_heavy')):
        heavy_regions.extend(['fwr4'] * len(row['fwr4_aa_heavy']))
    region_labels.extend(heavy_regions)
    region_labels.append('special')
    

    light_regions = []
    if not pd.isna(row.get('fwr1_aa_light')):
        light_regions.extend(['fwr1'] * len(row['fwr1_aa_light']))
    if not pd.isna(row.get('cdr1_aa_light')):
        light_regions.extend(['cdr1'] * len(row['cdr1_aa_light']))
    if not pd.isna(row.get('fwr2_aa_light')):
        light_regions.extend(['fwr2'] * len(row['fwr2_aa_light']))
    if not pd.isna(row.get('cdr2_aa_light')):
        light_regions.extend(['cdr2'] * len(row['cdr2_aa_light']))
    if not pd.isna(row.get('fwr3_aa_light')):
        light_regions.extend(['fwr3'] * len(row['fwr3_aa_light']))
    if not pd.isna(row.get('cdr3_aa_light')):
        light_regions.extend(['cdr3'] * len(row['cdr3_aa_light']))
    if not pd.isna(row.get('fwr4_aa_light')):
        light_regions.extend(['fwr4'] * len(row['fwr4_aa_light']))
    region_labels.extend(light_regions)
    return region_labels

def create_single_region_labels(row):
    region_labels = []

    region_labels.append('special')
    
    heavy_regions = []
    if not pd.isna(row.get('fwr1_aa')):
        heavy_regions.extend(['fwr1'] * len(row['fwr1_aa']))
    if not pd.isna(row.get('cdr1_aa')):
        heavy_regions.extend(['cdr1'] * len(row['cdr1_aa']))
    if not pd.isna(row.get('fwr2_aa')):
        heavy_regions.extend(['fwr2'] * len(row['fwr2_aa']))
    if not pd.isna(row.get('cdr2_aa')):
        heavy_regions.extend(['cdr2'] * len(row['cdr2_aa']))
    if not pd.isna(row.get('fwr3_aa')):
        heavy_regions.extend(['fwr3'] * len(row['fwr3_aa']))
    if not pd.isna(row.get('cdr3_aa')):
        heavy_regions.extend(['cdr3'] * len(row['cdr3_aa']))
    if not pd.isna(row.get('fwr4_aa')):
        heavy_regions.extend(['fwr4'] * len(row['fwr4_aa']))
    
    region_labels.extend(heavy_regions)
    
    return region_labels


def tokenize_function(examples):

    tokenized = tokenizer(
        examples["combined_sequence"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )
    
    token_region_labels = []
    
    for i in range(len(tokenized["input_ids"])):

        input_ids = tokenized["input_ids"][i]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        special_tokens_mask = tokenized["special_tokens_mask"][i]

        sequence_regions = []
        current_pos = 0
        
        for j, (token, is_special) in enumerate(zip(tokens, special_tokens_mask)):
            if is_special == 1:
                sequence_regions.append("special")
            else:
                if token in ["[HEAVY]", "[LIGHT]"]:
                    sequence_regions.append("special")
                    current_pos += 1
                else:
                    if current_pos < len(examples["region_labels"][i]):
                        sequence_regions.append(examples["region_labels"][i][current_pos])
                        current_pos += 1
                    else:
                        sequence_regions.append("other")
        
        token_region_labels.append(sequence_regions)
    
    tokenized["region_labels"] = token_region_labels
    return tokenized


class RegionAwareAntibodyCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        
        self.region_probabilities = {
            'fwr1': 0.1,
            'cdr1': 0.2,
            'fwr2': 0.1,
            'cdr2': 0.3,
            'fwr3': 0.15,
            'cdr3': 0.4,
            'fwr4': 0.1,
            'special': 0.0,  
            'other': 0.1 
        }
        
        self.heavy_id = self.tokenizer.convert_tokens_to_ids("[HEAVY]")
        self.light_id = self.tokenizer.convert_tokens_to_ids("[LIGHT]")
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id
        
    
    def __call__(self, examples):
        batch = self._build_batch(examples)
        
        if self.mlm:
            inputs = batch["input_ids"].clone()
            labels = batch["input_ids"].clone()
            

            region_labels = batch["region_labels"]
            
            special_tokens_mask = torch.isin(
                inputs, 
                torch.tensor([self.heavy_id, self.light_id, self.pad_id, self.cls_id, self.eos_id]) 
            )
            
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            
            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    region = region_labels[i][j]
                    probability_matrix[i, j] = self.region_probabilities.get(region, self.region_probabilities['other'])
                    if special_tokens_mask[i, j]:
                        probability_matrix[i, j] = 0.0

            masked_indices = torch.bernoulli(probability_matrix).bool()
            masked_indices[special_tokens_mask] = False
            labels[~masked_indices] = -100
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.mask_id
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
            batch["input_ids"] = inputs
            batch["labels"] = labels

        del batch["region_labels"]

        return batch
    
    def _build_batch(self, examples):
        batch_size = len(examples)
        seq_length = len(examples[0]["input_ids"]) 
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        region_labels = []
        for example in examples:
            if "region_labels" in example:
                if len(example["region_labels"]) == seq_length:
                    region_labels.append(example["region_labels"])
                elif len(example["region_labels"]) < seq_length:
                    region_labels.append(example["region_labels"] + ["other"] * (seq_length - len(example["region_labels"])))
                else:
                    region_labels.append(example["region_labels"][:seq_length])
            else:
                region_labels.append(["other"] * seq_length)
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "region_labels": region_labels 
        }
        
        return batch
        

paired_csv_path = '/data/hnq/paired_final.csv'
vh_csv_path = '/data/hnq/VH_final.csv'  
vl_csv_path = '/data/hnq/VL_final.csv'  


dataset_dict = preprocess_and_split_data(
    paired_csv_path=paired_csv_path,
    vh_csv_path=vh_csv_path,
    vl_csv_path=vl_csv_path,
    test_size=0.1,
    seed=42
)


tokenizer = EsmTokenizer.from_pretrained('/data/hnq/Tokenizer')
special_tokens = {"additional_special_tokens": ["[HEAVY]", "[LIGHT]"]}
tokenizer.add_special_tokens(special_tokens)

tokenized_train = dataset_dict['train'].map(tokenize_function, batched=True)
tokenized_test = dataset_dict['test'].map(tokenize_function, batched=True)

tokenized_datasets = {
    'train': tokenized_train,
    'test': tokenized_test
}

esm_config = {
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 35,
    "max_len": 512,
    "max_position_embeddings": 512,   
    "batch_size": 200,  
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
    "pad_token_id": 1
}

model_config = EsmConfig(
    vocab_size=esm_config.get("vocab_size"),
    hidden_size=esm_config.get("hidden_size"),
    max_position_embeddings=esm_config.get("max_position_embeddings"),
    num_hidden_layers=esm_config.get("num_hidden_layers", 12),
    num_attention_heads=esm_config.get("num_attention_heads", 12),
    pad_token_id= esm_config.get("pad_token_id", 1),
)

model = EsmForMaskedLM(model_config)
print(model)

total_num = sum(p.numel() for p in model.parameters())  # 12 + 12 80M // 16 + 16 114M // 24 + 24 171M
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  

print('total_para:', total_num)
print('train_para:', trainable_num)

args = TrainingArguments(
    output_dir="ESM_pretrain",
    overwrite_output_dir=True,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    max_steps=600000,  
    save_steps=50000,
    logging_steps=10000,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=1,
    fp16=True,
    evaluation_strategy="steps",
    seed=42,
    remove_unused_columns=False,
    dataloader_num_workers=8,
)

collator = RegionAwareAntibodyCollator(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 
)
checkpoint_path = '/home/hnq/PairedESM/Code/ESM_pretrain/checkpoint-500000'

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train(resume_from_checkpoint=checkpoint_path)

save_dir = '/home/hnq/PairedESM/Code/model_save'
trainer.save_model(save_dir)
