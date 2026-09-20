import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# 新增：TinyBERT分类头
class TinyBertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # 取[CLS] token
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=3.0):
        super().__init__()
        self.alpha = alpha  
        self.temp = temperature  
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce_loss(student_logits, labels)

        soft_teacher = F.softmax(teacher_logits / self.temp, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temp, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temp ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# 修改：原StudentModel改为GraphCodeBertStudentModel
class DistilRobertaStudentModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(DistilRobertaStudentModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)

        # embedding - GraphCodeBERT特殊的数据流嵌入处理
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings*(~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs[:, 0, :]
        else:
            return logits

# 新增：TinyBERT学生模型（简化的数据流处理）
class TinyBertStudentModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(TinyBertStudentModel, self).__init__()
        self.encoder = encoder  # BERT模型
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = TinyBertClassificationHead(config)
        self.args = args
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        
        # 将两个代码段的输入合并
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        
        # 对于TinyBERT，我们使用简化的注意力掩码（只考虑padding token）
        # 创建基础的attention mask（1表示真实token，0表示padding）
        attention_mask = (inputs_ids != self.tokenizer.pad_token_id).long()
        
        # TinyBERT（BERT）需要token_type_ids，这里设为0（单句任务）
        token_type_ids = torch.zeros_like(inputs_ids)
        
        # 通过TinyBERT模型
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取[CLS] token的输出
        sequence_output = outputs[0]  # (batch_size*2, seq_len, hidden_size)
        logits = self.classifier(sequence_output)
        prob = F.softmax(logits)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, sequence_output[:, 0, :]
        else:
            return logits

class TeacherModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(TeacherModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings*(~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs[:, 0, :]
        else:
            return logits

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings*(~nodes_mask)[:,:,None] + avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs[:, 0, :]
        else:
            return prob
    
    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4, pin_memory=False)

        self.eval()
        logits = [] 
        labels = []
        for batch in eval_dataloader:
            (inputs_ids_1, position_idx_1, attn_mask_1,
            inputs_ids_2, position_idx_2, attn_mask_2,
            label) = [x.to("cuda") for x in batch]
            with torch.no_grad():
                logit = self.forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:,0]]

        return probs, pred_labels