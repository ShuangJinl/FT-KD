import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CodeT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model*2, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
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

class TeacherModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(TeacherModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args

    def forward(self, input_ids, labels): 
        input_ids = input_ids.view(-1,self.args.code_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0][:, 0, :]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs
        else:
            return logits

class StudentModel(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(StudentModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args

    def forward(self, input_ids, labels): 
        input_ids = input_ids.view(-1,self.args.code_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0][:, 0, :]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits, outputs
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

    def forward(self, input_ids, labels): 
        input_ids = input_ids.view(-1,self.args.code_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0][:, 0, :]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs
        else:
            return prob

# 新增 TinyBERT 相关类
class TinyBertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TinyBertModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(TinyBertModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=TinyBertClassificationHead(config)
        self.args=args

    def forward(self, input_ids, labels): 
        input_ids = input_ids.view(-1,self.args.code_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0][:, 0, :]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs
        else:
            return prob