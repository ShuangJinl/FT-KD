import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks for TinyBERT/BERT."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take [CLS] token
        x = x.reshape(-1,x.size(-1)*2)
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

class StudentModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(StudentModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        
        # 自动选择 Classification Head
        if hasattr(config, 'architectures') and 'Bert' in str(config.architectures):
             self.classifier = BertClassificationHead(config)
        else:
             self.classifier = RobertaClassificationHead(config)
             
        self.args = args
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)

        # Embedding logic
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        

        # 兼容 RoBERTa 和 BERT 的 embeddings 获取方式
        if hasattr(self.encoder, 'roberta'):
            base_model = self.encoder.roberta
            word_embeddings = base_model.embeddings.word_embeddings
        elif hasattr(self.encoder, 'bert'):
            base_model = self.encoder.bert
            word_embeddings = base_model.embeddings.word_embeddings
        else:
            base_model = self.encoder
            if hasattr(base_model, 'embeddings'):
                word_embeddings = base_model.embeddings.word_embeddings
            else:
                 # Fallback provided for generic cases, though specific structure expected
                word_embeddings = base_model.embeddings.word_embeddings

        inputs_embeddings = word_embeddings(inputs_ids)

        # Graph-guided embedding calculation
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings * nodes_mask[:,:,None]    
        
        # Forward pass through encoder
        # position_ids logic fits both if passed correctly
        outputs = base_model(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        
        logits = self.classifier(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs[:, 0, :]
        else:
            return logits
        

class TeacherModel(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(TeacherModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config) # UniXCoder assumes RoBERTa head
        self.args = args
    
    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None): 
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs*2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs*2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs*2, l, l)

        # Embedding logic (Teacher is UniXCoder/RoBERTa)
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)        
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        
        nodes_to_token_mask = nodes_mask[:,:,None] & token_mask[:,None,:] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:,:,None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:,:,None] + avg_embeddings * nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs[:, 0, :]
        else:
            return logits

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels=None): 
        bs,l=inputs_ids_1.size()
        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob, outputs[:, 0, :]
        else:
            return prob