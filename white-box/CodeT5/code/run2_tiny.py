from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer,
                          RobertaForSequenceClassification,
                          BertConfig, BertModel, BertTokenizer)
from torch.optim import AdamW
from tqdm import tqdm
from model import Model, TeacherModel, StudentModel, DistillationLoss
import time

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    'codet5-base': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'codebert-base':(RobertaConfig, RobertaModel, RobertaTokenizer),
    'unixcoder-base':(RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'graphcodebert-base':(RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilroberta-base': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5small-base': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'tinybert': (BertConfig, BertModel, BertTokenizer),
}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 label,
                 url1,
                 url2,
                 teacher_input_ids=None # 新增：专门用于教师模型的输入
    ):
        self.input_ids = input_ids # 这始终是 Student 的输入 (或者单模型时的输入)
        self.teacher_input_ids = teacher_input_ids
        self.label=label
        self.url1=url1
        self.url2=url2

def convert_code_to_ids(code_tokens, tokenizer, args):
    """Helper to convert tokens to ids with padding"""
    tokens = code_tokens[:args.code_length-2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.code_length - len(ids)
    ids += [tokenizer.pad_token_id] * padding_length
    return ids

def convert_examples_to_features(code1, code2, label, url1, url2, tokenizer, args, teacher_tokenizer=None):
    # 处理 Student (或默认) Tokenizer
    # 注意：这里接收的是 raw code 字符串或者已经 tokenize 好的列表，为了稳健性，我们在 get_example 里统一处理
    # 假设传入的是 raw code (如果未缓存) 或者 tokens
    
    # 这里的逻辑稍微调整：get_example 负责分词，这里负责转 ID
    # 但由于 T 和 S 分词方式不同（Subword 切分不同），必须传入原始代码重新分词
    
    # 为了简化修改并复用缓存逻辑，我们在 get_example 内部做真正的处理，
    # InputFeatures 只负责存储结果。
    pass 

def get_example(item):
    # item: (url1, url2, label, tokenizer, args, cache, url_to_code, teacher_tokenizer)
    url1, url2, label, tokenizer, args, cache, url_to_code, teacher_tokenizer = item
    
    # --- 1. 获取原始代码 ---
    code1_raw = ""
    code2_raw = ""
    if url1 in url_to_code:
        code1_raw = ' '.join(url_to_code[url1].split())
    if url2 in url_to_code:
        code2_raw = ' '.join(url_to_code[url2].split())

    # --- 2. 处理主要 Tokenizer (Student) ---
    def get_ids(txt, tok):
        tokens = tok.tokenize(txt)
        return convert_code_to_ids(tokens, tok, args)

    # 处理 Code 1
    code1_ids = get_ids(code1_raw, tokenizer)
    # 处理 Code 2
    code2_ids = get_ids(code2_raw, tokenizer)
    source_ids = code1_ids + code2_ids

    # --- 3. 处理 Teacher Tokenizer (如果是蒸馏模式) ---
    teacher_source_ids = None
    if teacher_tokenizer:
        t_code1_ids = get_ids(code1_raw, teacher_tokenizer)
        t_code2_ids = get_ids(code2_raw, teacher_tokenizer)
        teacher_source_ids = t_code1_ids + t_code2_ids

    return InputFeatures(source_ids, label, url1, url2, teacher_input_ids=teacher_source_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', teacher_tokenizer=None):
        postfix = file_path.split('/')[-1].split('.csv')[0]
        self.examples = []
        self.args = args
        self.teacher_tokenizer = teacher_tokenizer
        
        # 区分缓存文件名：如果是蒸馏模式，文件名要包含 _distill
        cache_suffix = "_distill" if teacher_tokenizer else ""
        
        folder = "../cached_files"
        cache_file_path = os.path.join(folder, 'cached_{}{}'.format(postfix, cache_suffix))
        code_pairs_file_path = os.path.join(folder, 'cached_{}{}.pkl'.format(postfix, cache_suffix))
        
        logger.info("Creating features from index file at %s ", file_path)
        url_to_code={}

        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            import csv
            logger.info("Creating features from dataset file at %s", file_path)
            # 加载原始代码库
            with open(args.code_db_file) as f:
                file_reader = csv.reader(f)
                next(file_reader)  
                for line in file_reader:  
                    url_to_code[line[0]] = line[1]
            
            # 加载索引
            data = []
            cache = {} # 这里不再使用基于 token 的缓存，直接从 url_to_code 取原始文本
            with open(file_path) as f:
                file_reader = csv.reader(f)
                next(file_reader)
                for line in file_reader:
                    _,_,url1,url2,label=line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    label = 0 if label=='0' else 1
                    # 将所有需要的信息打包传给 get_example
                    data.append((url1, url2, label, tokenizer, args, cache, url_to_code, teacher_tokenizer))

            # 生成 Features (这里会保存 code pair 信息用于 evaluation dump)
            code_pairs = []
            for sing_example in data:
                code_pairs.append([sing_example[0], sing_example[1], 
                                   url_to_code[sing_example[0]], url_to_code[sing_example[1]]])
            
            with open(code_pairs_file_path, 'wb') as f:
                pickle.dump(code_pairs, f)
                
            self.examples = [get_example(item) for item in data]
            torch.save(self.examples, cache_file_path)
        
        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("student_ids (first 20): {}".format(' '.join(map(str, example.input_ids[:20]))))
                if example.teacher_input_ids:
                    logger.info("teacher_ids (first 20): {}".format(' '.join(map(str, example.teacher_input_ids[:20]))))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        # 返回 (student_input, label) 或者 (teacher_input, student_input, label)
        student_ids = torch.tensor(self.examples[item].input_ids)
        label = torch.tensor(self.examples[item].label)
        
        if self.teacher_tokenizer:
            teacher_ids = torch.tensor(self.examples[item].teacher_input_ids)
            return teacher_ids, student_ids, label
        else:
            return student_ids, label


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, Tmodel, model, tokenizer):
    """ Train the model """
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=0)

    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len(train_dataloader)
    args.warmup_steps=args.max_steps//5

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    
    global_step=0
    tr_loss, logging_loss, avg_loss = 0.0, 0.0, 0.0
    tr_num, train_loss = 0, 0
    best_f1=0
    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        criterion = DistillationLoss(alpha=0.7, temperature=3.0)
        logger.info("-------------------------")
        for step, batch in enumerate(bar):
            # batch 解包：(teacher_inputs, student_inputs, labels)
            t_inputs = batch[0].to(args.device)
            s_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device) 
            
            model.train()
            with torch.no_grad():
                # 教师模型使用 t_inputs (CodeT5 ids)
                _, teacher_logits, _ = Tmodel(t_inputs, labels)

            # 学生模型使用 s_inputs (TinyBERT ids)
            _, student_logits, _ = model(s_inputs, labels)

            loss = criterion(student_logits, teacher_logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1

                if global_step % args.save_steps == 0:
                    # 评估时只使用 Student Tokenizer，所以这里的 evaluate 不需要改动，只需要 dataset 正确
                    results, outputs_and_embeddings  = evaluate(args, model, tokenizer)    
                    
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('TinyBERT/student_model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                        import pickle
                        with open(os.path.join(args.output_dir, 'student_outputs_and_embeddings.pkl'), 'wb') as f:
                            pickle.dump(outputs_and_embeddings, f)
        
                      
def evaluate(args, model, tokenizer):
    # Eval 仅需要 Student 数据，因此不传 teacher_tokenizer，使用 "single" 模式
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    embeddings = []
    for batch in tqdm(eval_dataloader):
        # 此时 batch 只有 (student_inputs, labels)
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit,embedding = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
        nb_eval_steps += 1
    
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    embeddings = np.concatenate(embeddings,0)
    best_threshold=0.5

    # 处理缓存后缀，为了读取正确的 pkl
    # Eval 这里用的是标准缓存，不带 _distill (除非你之前跑 evaluation 也加了 distill，这里保持简单)
    folder = "../cached_files"
    postfix = args.test_data_file.split('/')[-1].split('.csv')[0]
    # 注意：Evaluate 生成 Dataset 时没有传 teacher_tokenizer，所以它读取/生成的是不带后缀的 cache
    # 如果之前运行过旧代码，可能存在旧缓存。
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(postfix))
    
    # 鲁棒性检查
    if not os.path.exists(code_pairs_file_path):
        # 如果不存在，尝试找 distill 的
        code_pairs_file_path = os.path.join(folder, 'cached_{}_distill.pkl'.format(postfix))
    
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = np.array(pickle.load(f))[:,2:]
        
    outputs_and_embeddings = [[code_pairs[i][0], code_pairs[i][1],y_trues[i], np.argmax(logits[i]), \
                               embeddings[i*2],embeddings[i*2+1]] for i in range(len(y_trues))]

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average='macro')   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average='macro')             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    return result, outputs_and_embeddings


def test(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=0)

    logger.info("***** Running Test *****")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit,_ = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0.5
    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average='macro')   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average='macro')             
    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1)
    }
    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    return result

                                           
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_db_file", default=None, type=str, required=True)
    parser.add_argument("--requires_grad", default=None, type=int, required=True)
    parser.add_argument("--train_data_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    set_seed(args)
    
    # Load Classes
    Tconfig_class, Tmodel_class, Ttokenizer_class = MODEL_CLASSES[args.teacher_model_name_or_path.split('/')[-1]]
    
    # Auto-detect TinyBERT/BERT class
    student_key = args.student_model_name_or_path.split('/')[-1]
    if student_key in MODEL_CLASSES:
         Sconfig_class, Smodel_class, Stokenizer_class = MODEL_CLASSES[student_key]
    else:
        Sconfig_class, Smodel_class, Stokenizer_class = BertConfig, BertModel, BertTokenizer

    # Load Configs & Tokenizers
    Tconfig = Tconfig_class.from_pretrained(args.teacher_config_name if args.teacher_config_name else args.teacher_model_name_or_path)
    Sconfig = Sconfig_class.from_pretrained(args.student_config_name if args.student_config_name else args.student_model_name_or_path)
    Tconfig.num_labels=2
    Sconfig.num_labels=2
    Ttokenizer = Ttokenizer_class.from_pretrained(args.teacher_tokenizer_name)
    Stokenizer = Stokenizer_class.from_pretrained(args.student_tokenizer_name)
    
    # Load Models
    Tmodel = Tmodel_class.from_pretrained(args.teacher_model_name_or_path)
    Smodel  = Smodel_class.from_pretrained(args.student_model_name_or_path)
    
    Tmodel = TeacherModel(Tmodel, Tconfig, Ttokenizer, args)
    Smodel = StudentModel(Smodel, Sconfig, Stokenizer, args)
    
    # Load Teacher Weights
    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    if os.path.exists(output_dir):
        Tmodel.load_state_dict(torch.load(output_dir))
    
    Tmodel.to(args.device)
    Smodel.to(args.device)

    for name, param in Smodel.named_parameters():
        if 'classifier' not in name: 
            param.requires_grad = False if args.requires_grad==0 else True

    logger.info("Training/evaluation parameters %s", args)
    
    # Training
    if args.do_train:
        t1 = time.time()
        # 关键修改：传入 Stokenizer 作为主分词器，Ttokenizer 作为辅助分词器
        train_dataset = TextDataset(Stokenizer, args, file_path=args.train_data_file, teacher_tokenizer=Ttokenizer)
        train(args, train_dataset, Tmodel, Smodel, Stokenizer)
        t2 = time.time()
        print("training time: ",t2-t1,"s")
    
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/TinyBERT/student_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        Smodel.load_state_dict(torch.load(output_dir))
        Smodel.to(args.device)
        results, embeddings = evaluate(args, Smodel, Stokenizer)
        
    if args.do_test:
        t1 = time.time()
        checkpoint_prefix = 'checkpoint-best-f1/TinyBERT/student_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        Smodel.load_state_dict(torch.load(output_dir))
        Smodel.to(args.device)
        results = test(args, Smodel, Stokenizer)
        t2 = time.time()
        print("elphase time: ",t2-t1,"s")
    return results

if __name__ == "__main__":
    main()