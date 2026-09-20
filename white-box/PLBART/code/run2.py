from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer,
                          RobertaForSequenceClassification, 
                          BertConfig, BertModel, BertTokenizer) # 引入 BERT 组件
from torch.optim import AdamW
from tqdm import tqdm
from model import TeacherModel, StudentModel, DistillationLoss
import time

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    'codet5-base': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'distilroberta-base': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'tinybert-base': (BertConfig, BertModel, BertTokenizer), # 添加 TinyBERT
    'bert-base-uncased': (BertConfig, BertModel, BertTokenizer)
}

class InputFeatures(object):
    """修改：同时存储 Teacher 和 Student 的 input_ids"""
    def __init__(self,
                 teacher_input_ids,
                 student_input_ids,
                 label,
                 url1,
                 url2
    ):
        self.teacher_input_ids = teacher_input_ids
        self.student_input_ids = student_input_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2
        
def convert_examples_to_features(code1_str, code2_str, label, url1, url2, teacher_tokenizer, student_tokenizer, args):
    """
    修改：接收原始代码字符串，分别用两个 tokenizer 进行处理
    """
    # --- 1. Teacher Tokenization (PLBART) ---
    # PLBART 使用 SentencePiece，通常直接处理 raw string 或 tokens
    t_tokens1 = teacher_tokenizer.tokenize(code1_str)[:args.code_length-2]
    t_tokens1 = [teacher_tokenizer.cls_token] + t_tokens1 + [teacher_tokenizer.sep_token]
    t_tokens2 = teacher_tokenizer.tokenize(code2_str)[:args.code_length-2]
    t_tokens2 = [teacher_tokenizer.cls_token] + t_tokens2 + [teacher_tokenizer.sep_token]
    
    t_ids1 = teacher_tokenizer.convert_tokens_to_ids(t_tokens1)
    t_ids1 += [teacher_tokenizer.pad_token_id] * (args.code_length - len(t_ids1))
    
    t_ids2 = teacher_tokenizer.convert_tokens_to_ids(t_tokens2)
    t_ids2 += [teacher_tokenizer.pad_token_id] * (args.code_length - len(t_ids2))
    
    teacher_ids = t_ids1 + t_ids2

    # --- 2. Student Tokenization (TinyBERT/WordPiece) ---
    s_tokens1 = student_tokenizer.tokenize(code1_str)[:args.code_length-2]
    s_tokens1 = [student_tokenizer.cls_token] + s_tokens1 + [student_tokenizer.sep_token]
    s_tokens2 = student_tokenizer.tokenize(code2_str)[:args.code_length-2]
    s_tokens2 = [student_tokenizer.cls_token] + s_tokens2 + [student_tokenizer.sep_token]
    
    s_ids1 = student_tokenizer.convert_tokens_to_ids(s_tokens1)
    s_ids1 += [student_tokenizer.pad_token_id] * (args.code_length - len(s_ids1))
    
    s_ids2 = student_tokenizer.convert_tokens_to_ids(s_tokens2)
    s_ids2 += [student_tokenizer.pad_token_id] * (args.code_length - len(s_ids2))
    
    student_ids = s_ids1 + s_ids2
    
    return InputFeatures(teacher_ids, student_ids, label, url1, url2)


def get_example(item):
    url1, url2, label, teacher_tokenizer, student_tokenizer, args, url_to_code = item
    
    # 获取原始代码字符串
    # 注意：这里不再先 tokenize，而是获取 raw string 传给 convert_examples_to_features
    try:
        if url1 in url_to_code:
            code1 = ' '.join(url_to_code[url1].split())
        else:
            code1 = ""
    except:
        code1 = ""

    try:
        if url2 in url_to_code:
            code2 = ' '.join(url_to_code[url2].split())
        else:
            code2 = ""
    except:
        code2 = ""

    return convert_examples_to_features(code1, code2, label, url1, url2, teacher_tokenizer, student_tokenizer, args)


class TextDataset(Dataset):
    def __init__(self, teacher_tokenizer, student_tokenizer, args, file_path='train'):
        postfix = file_path.split('/')[-1].split('.csv')[0]
        self.examples = []
        self.args = args
        index_filename = file_path
        
        # 缓存文件夹
        folder = "../cached_files_distillation"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # 缓存文件名包含 Teacher 和 Student 的名字，防止混淆
        t_name = args.teacher_model_name_or_path.split('/')[-1]
        s_name = args.student_model_name_or_path.split('/')[-1]
        cache_file_path = os.path.join(folder, 'cached_{}_{}_{}.pkl'.format(postfix, t_name, s_name))
        
        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            logger.info("Creating features from dataset file at %s", file_path)
            url_to_code = {}
            import csv
            # 读取代码库
            with open(args.code_db_file) as f:
                file_reader = csv.reader(f)
                next(file_reader)  
                for line in file_reader:  
                    url_to_code[line[0]] = line[1]
                    
            data = []
            # 读取索引文件
            with open(index_filename) as f:
                file_reader = csv.reader(f)
                next(file_reader)
                for line in file_reader:
                    _,_,url1,url2,label = line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    label = 0 if label=='0' else 1
                    # 将两个 tokenizer 都传入
                    data.append((url1, url2, label, teacher_tokenizer, student_tokenizer, args, url_to_code))

            # 处理数据
            self.examples = [get_example(item) for item in tqdm(data, desc="Tokenizing")]
            torch.save(self.examples, cache_file_path)
        
        if 'train' in postfix and len(self.examples) > 0:
            logger.info("*** Example ***")
            logger.info("idx: 0")
            logger.info("Teacher Ids: %s", self.examples[0].teacher_input_ids[:10])
            logger.info("Student Ids: %s", self.examples[0].student_input_ids[:10])

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        # 返回 (teacher_ids, student_ids, label)
        return (torch.tensor(self.examples[item].teacher_input_ids),
                torch.tensor(self.examples[item].student_input_ids),
                torch.tensor(self.examples[item].label))
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, Tmodel, model, teacher_tokenizer, student_tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
   
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        Tmodel = torch.nn.DataParallel(Tmodel)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1 = 0
    model.zero_grad()
    
    # 蒸馏损失函数
    criterion = DistillationLoss(alpha=0.7, temperature=3.0)
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        
        for step, batch in enumerate(bar):
            # 解包：batch[0] 是 teacher_ids, batch[1] 是 student_ids, batch[2] 是 labels
            t_inputs = batch[0].to(args.device)
            s_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device) 
            
            model.train()
            Tmodel.eval() # 确保 Teacher 处于评估模式

            with torch.no_grad():
                # Teacher 前向传播
                _, teacher_logits, _ = Tmodel(t_inputs, labels)

            # Student 前向传播
            _, student_logits, _ = model(s_inputs, labels)

            # 计算蒸馏 Loss
            loss = criterion(student_logits, teacher_logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1

                if global_step % args.save_steps == 0:
                    # 评估
                    results = evaluate(args, model, teacher_tokenizer, student_tokenizer)    
                    
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  Best f1: %s", round(best_f1, 4))
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'student_model.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
        
                      
def evaluate(args, model, teacher_tokenizer, student_tokenizer):
    # 使用包含两个 Tokenizer 的 Dataset
    eval_dataset = TextDataset(teacher_tokenizer, student_tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    logger.info("***** Running evaluation *****")
    eval_loss = 0.0
    model.eval()
    logits = []  
    y_trues = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 即使只用 student，Dataset 也会返回三个值
        t_inputs = batch[0].to(args.device) # 这一步其实不需要，但为了解包一致性
        s_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device) 
        
        with torch.no_grad():
            lm_loss, logit, _ = model(s_inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score, precision_score, f1_score
    recall = recall_score(y_trues, y_preds, average='macro')
    precision = precision_score(y_trues, y_preds, average='macro')   
    f1 = f1_score(y_trues, y_preds, average='macro')             
    
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def test(args, model, teacher_tokenizer, student_tokenizer):
    eval_dataset = TextDataset(teacher_tokenizer, student_tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    logger.info("***** Running Test *****")
    model.eval()
    logits = []  
    y_trues = []
    
    for batch in tqdm(eval_dataloader, desc="Testing"):
        # 同样解包三个值
        _, s_inputs, labels = batch
        s_inputs = s_inputs.to(args.device)
        labels = labels.to(args.device) 
        
        with torch.no_grad():
            _, logit, _ = model(s_inputs, labels)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score, precision_score, f1_score
    recall = recall_score(y_trues, y_preds, average='macro')
    precision = precision_score(y_trues, y_preds, average='macro')   
    f1 = f1_score(y_trues, y_preds, average='macro')             
    
    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

                                           
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--code_db_file", default=None, type=str, required=True)
    parser.add_argument("--requires_grad", default=None, type=int, required=True)
    parser.add_argument("--train_data_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    
    # 模型路径参数
    parser.add_argument("--teacher_model_name_or_path", default=None, type=str)
    parser.add_argument("--student_model_name_or_path", default=None, type=str)
    parser.add_argument("--teacher_config_name", default="", type=str)
    parser.add_argument("--student_config_name", default="", type=str)
    parser.add_argument("--teacher_tokenizer_name", default="", type=str)
    parser.add_argument("--student_tokenizer_name", default="", type=str)
    
    parser.add_argument("--code_length", default=512, type=int) 
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')    

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    set_seed(args)

    Tconfig_class, Tmodel_class, Ttokenizer_class = MODEL_CLASSES[args.teacher_model_name_or_path.split('/')[-1]]
    Sconfig_class, Smodel_class, Stokenizer_class = MODEL_CLASSES[args.student_model_name_or_path.split('/')[-1]]
    Tconfig = Tconfig_class.from_pretrained(args.teacher_config_name if args.teacher_config_name else args.teacher_model_name_or_path)
    Sconfig = Sconfig_class.from_pretrained(args.student_config_name if args.student_config_name else args.student_model_name_or_path)
    Tconfig.num_labels=2
    Sconfig.num_labels=2
    Ttokenizer = Ttokenizer_class.from_pretrained(args.teacher_tokenizer_name)
    Stokenizer = Stokenizer_class.from_pretrained(args.student_tokenizer_name)
    Tmodel = Tmodel_class.from_pretrained(args.teacher_model_name_or_path)
    Smodel  = Smodel_class.from_pretrained(args.student_model_name_or_path)
    Tmodel = TeacherModel(Tmodel, Tconfig, Ttokenizer, args)
    Smodel = StudentModel(Smodel, Sconfig, Stokenizer, args)
    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    Tmodel.load_state_dict(torch.load(output_dir))
    Tmodel.to(args.device)
    Smodel.to(args.device)

    # 冻结部分参数逻辑 (根据 args.requires_grad)
    for name, param in Smodel.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False if args.requires_grad==0 else True

    logger.info("Training/evaluation parameters %s", args)
    
    if args.do_train:
        t1 = time.time()
        train_dataset = TextDataset(Ttokenizer, Stokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, Tmodel, Smodel, Ttokenizer, Stokenizer)
        t2 = time.time()
        print("training time: ", t2-t1, "s")
    
    if args.do_test:
        t3 = time.time()
        checkpoint_prefix = 'checkpoint-best-f1/student_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        Smodel.load_state_dict(torch.load(output_dir))
        Smodel.to(args.device)
        test(args, Smodel, Ttokenizer, Stokenizer)
        t4 = time.time()
        print("testing time: ", (t4-t3)/420, "s")

if __name__ == "__main__":
    main()