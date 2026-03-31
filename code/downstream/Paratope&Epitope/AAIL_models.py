from transformers import (
    EsmModel,
    EsmPreTrainedModel,
    BertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import ipdb

class ModifiedEsmForTokenClassification_V2(EsmPreTrainedModel):
    def __init__(self, config, config2, word_embedding, word_embedding2, num_labels, ab_step, ag_step):
        super().__init__(config)
        self.ab_step = ab_step
        self.ag_step = ag_step

        self.num_labels = num_labels

        self.bert = EsmModel(config)
        
        self.antigen = BertModel(config2)

        self.ab_feature_extraction = ab_representation_feature_extraction(feature_dim=768, total_step=ab_step)
        self.ag_feature_extraction = ag_representation_feature_extraction(feature_dim=1024, total_step=ag_step)

        self.ab_cls = Atten_tokenclassficationHead(word_embedding)

        self.ag_cls = Atten_tokenclassficationHead2(word_embedding2)

        self.classifiers = nn.Sequential(
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(1024,self.num_labels)

        )

    def focal_loss(self, inputs, targets, pos_weight):

        mask = (targets != -100)
        inputs = inputs[mask]
        targets = targets[mask]

        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(inputs, targets)

        probs = F.softmax(inputs, dim=1)
        p_t = probs[torch.arange(len(targets)), targets]
        focal_factor = (1 - p_t) ** 2

        class_weights = torch.ones_like(targets, dtype = torch.float)
        class_weights[targets == 1] = pos_weight

        loss = focal_factor * ce_loss * class_weights

        return loss.mean()


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            antigen_labels = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            infill_mask= None,
            antigen_input_ids=None,
            antigen_attention_mask=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        antigen_rep = self.antigen(
            input_ids = antigen_input_ids,
            attention_mask=antigen_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[2]
        # antibody_pooler = outputs[1] # [batch, 768]
        representations = torch.stack(hidden_states, dim=2) # [batch, 510, 17, 768]

        new_rep = (torch.zeros(17).softmax(0).unsqueeze(0).cuda() @ representations.cuda()).squeeze(2) # [batch, 510, 768]
        rep0 = outputs[0]

        all_reps, extraction_output = self.ab_feature_extraction(rep0, new_rep)

        ag_hidden = antigen_rep[2]
        ag_representations = torch.stack(ag_hidden, dim=2)
        ag_rep = (torch.zeros(31).softmax(0).unsqueeze(0).cuda() @ ag_representations.cuda()).squeeze(2) # [batch, 512, 1024]

        ag_rep0 = antigen_rep[0]
        all_reps2, ag_extraction_output = self.ag_feature_extraction(ag_rep0, ag_rep)


        antibody_input = extraction_output
        antigen_input = ag_extraction_output
        antibody_output, antibody_attnetion = self.ab_cls(antibody_input,  input_ids) # [batch, antibody_length, 1024]
        antigen_output, antigen_attention = self.ag_cls(antigen_input, antigen_input_ids) # [batch, antigen_length, 1024]

        final_input = torch.cat([antibody_output, antigen_output], dim=1) # [batch, antibody_length+antigen_length, 1024]
        logits = self.classifiers(final_input) # [batch, antibody_length+antigen_length, 2]

        # nomarl CE loss
        # loss = None
        # loss_fct = nn.CrossEntropyLoss()
        # labels = torch.cat([labels, antigen_labels], dim=1)
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        antibody_logits = logits[:, :256, :]
        antigen_logits = logits[:, 256:, :]

        ab_loss =  self.focal_loss(
            antibody_logits.reshape(-1,2),
            labels.view(-1),
            pos_weight=5
        )

        ag_loss = self.focal_loss(
            antigen_logits.reshape(-1,2),
            antigen_labels.view(-1),
            pos_weight=10
        )

        total_loss = ab_loss + 2*ag_loss   #这里可以加点参数


        return SequenceClassifierOutput(
            loss = total_loss,
            logits = logits,
            hidden_states= None,
            attentions= None,
        )


class Atten_tokenclassficationHead(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()

        self.Linear = nn.Linear(768,512)
        self.word_linear = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.attention1 = nn.Sequential(
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Dropout(0.1)
        )

        self.attention2 = nn.Sequential(
            nn.Linear(512,256),
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )

        self.Linear2 = nn.Linear(256,1)

        self.word_embedding = word_embedding


    def forward(self, input,  input_ids):   # input [batch, 152, 768]

        input1 = self.Linear(input)  # input1 [batch, 152, 512]
        a = self.attention1(input1)  # a [batch, 152, 256]
        b = self.attention2(input1)  # b [batch, 152, 256]
        attention = a.mul(b)
        attention = self.Linear2(attention) # [batch, 152, 1]

        attention = F.softmax(attention, dim=1) 

        aa_embeds = self.word_embedding(input_ids)
        aa_embeds = self.word_linear(aa_embeds)
        aa_embeds = aa_embeds * attention # [batch, 152, 512]
    
        antibody_feature = input1 * attention # [batch, 152, 512]

        antibody_input = torch.cat([antibody_feature.squeeze(1), aa_embeds.squeeze(1)], dim=2)  # [batch, 152, 1024]

        return antibody_input, attention

class Atten_tokenclassficationHead2(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()

        self.Linear = nn.Linear(1024,512)
        self.word_linear = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.attention1 = nn.Sequential(
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Dropout(0.1)
        )

        self.attention2 = nn.Sequential(
            nn.Linear(512,256),
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )

        self.Linear2 = nn.Linear(256,1)

        self.word_embedding = word_embedding


    def forward(self, input, input_ids):   # input [batch, 512, 1024];  input_ids [batch, 512]

        input1 = self.Linear(input)  # input1 [batch, 512, 512]
        a = self.attention1(input1)  # a [batch, 512, 256]
        b = self.attention2(input1)  # b [batch, 512, 256]
        attention = a.mul(b)
        attention = self.Linear2(attention)

        attention = F.softmax(attention, dim=1) 

        aa_embeds = self.word_embedding(input_ids) # [batch, 512, 1024]
        aa_embeds = self.word_linear(aa_embeds)
        aa_embeds = aa_embeds * attention 
    
        antigen_feature = input1 * attention 

        antigen_input = torch.cat([antigen_feature.squeeze(1), aa_embeds.squeeze(1)], dim=2)  # [batch, 512, 1024]

        return antigen_input, attention
    
class ab_representation_feature_extraction(nn.Module):
    def __init__(self,feature_dim =768, total_step=10):
        super().__init__()

        self.feature_dim = feature_dim
        self.total_step = total_step
        
        if total_step > 0:
            self.step_embedding = nn.Embedding(total_step, feature_dim)
        else:
            self.register_parameter('step_embedding', None)

        self.rep0_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )

        self.conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=7, stride=1, padding=3)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )

        self.correction_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, rep0, new_rep):

        batch_size, seq_len, _ = rep0.shape
        device = rep0.device

        all_representations = []
        current_rep = rep0
        all_representations.append(current_rep)

        if self.total_step > 0:
            step_indices = torch.arange(self.total_step, dtype=torch.long, device=device)
            step_embed = self.step_embedding(step_indices) # [5,768]

            for step in range(self.total_step):
                rep_processed = self.rep0_mlp(current_rep.detach())
                step_embed_expanded = step_embed[step].view(1,1,-1).expand(batch_size, seq_len, -1)
                rep_processed = rep_processed + step_embed_expanded

                rep_cov = self.conv(rep_processed.transpose(1,2))
                rep_processed = rep_cov.transpose(1,2)

                fused_rep = torch.cat([rep_processed, new_rep], dim=-1)
                fused_rep = self.fusion_mlp(fused_rep)

                correction = self.correction_mlp(fused_rep)
                current_rep = current_rep - correction

                all_representations.append(current_rep)

        return all_representations, all_representations[-1]
    
class ag_representation_feature_extraction(nn.Module):
    def __init__(self,feature_dim =1024, total_step=1):
        super().__init__()

        self.feature_dim = feature_dim
        self.total_step = total_step
        
        if total_step > 0:
            self.step_embedding = nn.Embedding(total_step, feature_dim)
        else:
            self.register_parameter('step_embedding', None)

        self.rep0_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )

        self.conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=7, stride=1, padding=3)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )

        self.correction_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, rep0, new_rep):

        batch_size, seq_len, _ = rep0.shape
        device = rep0.device

        all_representations = []
        current_rep = rep0

        all_representations.append(current_rep)

        if self.total_step > 0:
            step_indices = torch.arange(self.total_step, dtype=torch.long, device=device)
            step_embed = self.step_embedding(step_indices) # [5,768]

            for step in range(self.total_step):
                rep_processed = self.rep0_mlp(current_rep.detach())
                step_embed_expanded = step_embed[step].view(1,1,-1).expand(batch_size, seq_len, -1)
                rep_processed = rep_processed + step_embed_expanded

                rep_cov = self.conv(rep_processed.transpose(1,2))
                rep_processed = rep_cov.transpose(1,2)

                fused_rep = torch.cat([rep_processed, new_rep], dim=-1)
                fused_rep = self.fusion_mlp(fused_rep)

                correction = self.correction_mlp(fused_rep)
                current_rep = current_rep - correction

                all_representations.append(current_rep)

        return all_representations, all_representations[-1]

