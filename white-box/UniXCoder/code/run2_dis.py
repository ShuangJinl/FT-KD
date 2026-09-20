from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          BertConfig, BertForSequenceClassification, BertTokenizer, 
                          T5Config, T5ForConditionalGeneration)
from torch.optim import AdamW
from tqdm import tqdm
from model import Model, TeacherModel, StudentModel, DistillationLoss

cpu_cont = 16
logger = logging.getLogger(__name__)

from parser import DFG_java
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
from tree_sitter import Language, Parser
import time

dfg_function = {'java': DFG_java}

# Modified MODEL_CLASSES to include DistilRoBERTa
MODEL_CLASSES = {
    'unixcoder-base': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5small-base': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'distilroberta-base': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'tinybert': (BertConfig, BertForSequenceClassification, BertTokenizer), 
    'bert-base': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'tinybert-base': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 添加这行以兼容你的路径命名 distilrobert-base，使用 RoBERTa 的类进行加载
    'distilrobert-base': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

# Load parsers
parsers = {}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser, dfg_function[lang]]    
    parsers[lang] = parser

def extract_dataflow(code, parser, lang):
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass    
    if lang == "php":
        code = "<?php" + code + "?>"    
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))    
        root_node = tree.root_node  
        tokens_index = tree_to_token_index(root_node)     
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]  
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)  
        try:
            DFG, _ = parser[1](root_node, index_to_code, {}) 
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg

class InputFeatures(object):
    """Stores features for both Teacher (UniXCoder) and Student (DistilRoBERTa/TinyBERT)."""
    def __init__(self,
             # Teacher Features
             input_tokens_1_t, input_ids_1_t, position_idx_1_t, dfg_to_code_1_t, dfg_to_dfg_1_t,
             input_tokens_2_t, input_ids_2_t, position_idx_2_t, dfg_to_code_2_t, dfg_to_dfg_2_t,
             # Student Features
             input_tokens_1_s, input_ids_1_s, position_idx_1_s, dfg_to_code_1_s, dfg_to_dfg_1_s,
             input_tokens_2_s, input_ids_2_s, position_idx_2_s, dfg_to_code_2_s, dfg_to_dfg_2_s,
             # Shared
             label, url1, url2
    ):
        # Teacher
        self.input_tokens_1_t = input_tokens_1_t
        self.input_ids_1_t = input_ids_1_t
        self.position_idx_1_t = position_idx_1_t
        self.dfg_to_code_1_t = dfg_to_code_1_t
        self.dfg_to_dfg_1_t = dfg_to_dfg_1_t
        self.input_tokens_2_t = input_tokens_2_t
        self.input_ids_2_t = input_ids_2_t
        self.position_idx_2_t = position_idx_2_t
        self.dfg_to_code_2_t = dfg_to_code_2_t
        self.dfg_to_dfg_2_t = dfg_to_dfg_2_t
        
        # Student
        self.input_tokens_1_s = input_tokens_1_s
        self.input_ids_1_s = input_ids_1_s
        self.position_idx_1_s = position_idx_1_s
        self.dfg_to_code_1_s = dfg_to_code_1_s
        self.dfg_to_dfg_1_s = dfg_to_dfg_1_s
        self.input_tokens_2_s = input_tokens_2_s
        self.input_ids_2_s = input_ids_2_s
        self.position_idx_2_s = position_idx_2_s
        self.dfg_to_code_2_s = dfg_to_code_2_s
        self.dfg_to_dfg_2_s = dfg_to_dfg_2_s

        self.label = label
        self.url1 = url1
        self.url2 = url2

def create_single_features(code, parser, tokenizer, args, lang='java', is_roberta=True):
    """Helper to create features for a single code snippet given a specific tokenizer."""
    code_tokens, dfg = extract_dataflow(code, parser, lang)
    
    # RoBERTa (UniXCoder/DistilRoBERTa) uses 'Ġ' (byte-level BPE), BERT uses '##'
    if is_roberta:
        code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    else:
        # BERT tokenizer logic (WordPiece)
        code_tokens = [tokenizer.tokenize(x) for x in code_tokens]

    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))    
    code_tokens = [y for x in code_tokens for y in x]  
    
    # Truncating
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][:512-3]
    
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length      
    
    # Reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code] 
    
    return source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

def convert_examples_to_features(item):
    url1, url2, label, tokenizer_t, tokenizer_s, args, cache, url_to_code = item
    parser = parsers['java']
    
    # Detect tokenizer type for correct tokenization logic
    is_roberta_t = isinstance(tokenizer_t, RobertaTokenizer)
    is_roberta_s = isinstance(tokenizer_s, RobertaTokenizer)

    # Process Teacher features (UniXCoder)
    if url1 not in cache:
        cache[url1] = {}
    
    if 'teacher' not in cache[url1]:
        func = url_to_code[url1]
        t_feats = create_single_features(func, parser, tokenizer_t, args, is_roberta=is_roberta_t)
        cache[url1]['teacher'] = t_feats

    if 'student' not in cache[url1]:
        func = url_to_code[url1]
        s_feats = create_single_features(func, parser, tokenizer_s, args, is_roberta=is_roberta_s)
        cache[url1]['student'] = s_feats

    # Repeat for url2
    if url2 not in cache:
        cache[url2] = {}

    if 'teacher' not in cache[url2]:
        func = url_to_code[url2]
        t_feats = create_single_features(func, parser, tokenizer_t, args, is_roberta=is_roberta_t)
        cache[url2]['teacher'] = t_feats

    if 'student' not in cache[url2]:
        func = url_to_code[url2]
        s_feats = create_single_features(func, parser, tokenizer_s, args, is_roberta=is_roberta_s)
        cache[url2]['student'] = s_feats

    # Retrieve
    t1 = cache[url1]['teacher']
    t2 = cache[url2]['teacher']
    s1 = cache[url1]['student']
    s2 = cache[url2]['student']

    return InputFeatures(
        t1[0], t1[1], t1[2], t1[3], t1[4], # Teacher 1
        t2[0], t2[1], t2[2], t2[3], t2[4], # Teacher 2
        s1[0], s1[1], s1[2], s1[3], s1[4], # Student 1
        s2[0], s2[1], s2[2], s2[3], s2[4], # Student 2
        label, url1, url2
    )

class TextDataset(Dataset):
    def __init__(self, tokenizer_t, tokenizer_s, args, file_path='train'):
        postfix = file_path.split('/')[-1].split('.csv')[0]
        self.examples = []
        self.args = args
        index_filename = file_path
        
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}

        folder = '../cached_files_distilled' 
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        cache_file_path = os.path.join(folder, 'cached_{}.pkl'.format(postfix))
        
        try:
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:
            import csv
            logger.info("Creating features from dataset file at %s", file_path)
            with open(args.code_db_file) as f:
                file_reader = csv.reader(f)
                next(file_reader)  
                for line in file_reader:  
                    url_to_code[line[0]] = line[1]
            
            data = []
            cache = {}
            with open(index_filename) as f:
                file_reader = csv.reader(f)
                next(file_reader)
                for line in file_reader:
                    if file_path == args.eval_data_file:
                        _, _, url1, url2, label = line
                    else:
                        if "GCJ" in args.train_data_file or "BCB" in args.train_data_file or "funcEq" in args.train_data_file:
                             _, url1, url2, label = line
                        else:
                             _, _, url1, url2, label = line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    label = 0 if label == '0' else 1
                    data.append((url1, url2, label, tokenizer_t, tokenizer_s, args, cache, url_to_code))

            self.examples = [convert_examples_to_features(x) for x in tqdm(data, total=len(data))]
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)
    
    def calculate_mask(self, input_ids, position_idx, dfg_to_code, dfg_to_dfg):
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        node_index = sum([i > 1 for i in position_idx])
        max_length = sum([i != 1 for i in position_idx])
        attn_mask[:node_index, :node_index] = True
        
        # General fix: special tokens attend to everything
        for idx, i in enumerate(input_ids):
             if i in [0, 2, 101, 102]: # 0/2 for RoBERTa, 101/102 for BERT
                  attn_mask[idx, :max_length] = True

        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        
        for idx, nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(position_idx):
                    attn_mask[idx + node_index, a + node_index] = True 
        return attn_mask

    def __getitem__(self, item):
        ex = self.examples[item]
        
        # Calculate masks for Teacher
        mask_1_t = self.calculate_mask(ex.input_ids_1_t, ex.position_idx_1_t, ex.dfg_to_code_1_t, ex.dfg_to_dfg_1_t)
        mask_2_t = self.calculate_mask(ex.input_ids_2_t, ex.position_idx_2_t, ex.dfg_to_code_2_t, ex.dfg_to_dfg_2_t)

        # Calculate masks for Student
        mask_1_s = self.calculate_mask(ex.input_ids_1_s, ex.position_idx_1_s, ex.dfg_to_code_1_s, ex.dfg_to_dfg_1_s)
        mask_2_s = self.calculate_mask(ex.input_ids_2_s, ex.position_idx_2_s, ex.dfg_to_code_2_s, ex.dfg_to_dfg_2_s)

        return (
            # Teacher Inputs
            torch.tensor(ex.input_ids_1_t), torch.tensor(ex.position_idx_1_t), torch.tensor(mask_1_t),
            torch.tensor(ex.input_ids_2_t), torch.tensor(ex.position_idx_2_t), torch.tensor(mask_2_t),
            # Student Inputs
            torch.tensor(ex.input_ids_1_s), torch.tensor(ex.position_idx_1_s), torch.tensor(mask_1_s),
            torch.tensor(ex.input_ids_2_s), torch.tensor(ex.position_idx_2_s), torch.tensor(mask_2_s),
            # Label
            torch.tensor(ex.label)
        )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, Tmodel, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, train_loss = 0.0, 0.0, 0.0, 0, 0
    best_f1 = 0
    
    model.zero_grad()
    Tmodel.eval()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        train_loss = 0
        criterion = DistillationLoss(alpha=0.7, temperature=3.0)
        
        for step, batch in enumerate(bar):
            # Unpack the massive tuple from dataset
            (ids_1_t, pos_1_t, mask_1_t, ids_2_t, pos_2_t, mask_2_t, 
             ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, 
             labels) = [x.to(args.device) for x in batch]
            
            model.train()
            with torch.no_grad():
                # Teacher Forward (UniXCoder inputs)
                _, teacher_logits, _ = Tmodel(ids_1_t, pos_1_t, mask_1_t, ids_2_t, pos_2_t, mask_2_t, labels)

            # Student Forward (DistilRoBERTa inputs)
            _, student_logits, _ = model(ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels)

            loss = criterion(student_logits, teacher_logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            train_loss += loss.item()
            
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss / (step + 1), 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                
                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, None, eval_when_training=True)    
                    
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('DistilroBERT/student_model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)

def evaluate(args, model, tokenizer, eval_when_training=False):
    # Re-instantiate tokenizers for test set loading
    _, _, Ttokenizer_class = MODEL_CLASSES[args.teacher_model_name_or_path.split('/')[-1]]
    _, _, Stokenizer_class = MODEL_CLASSES[args.student_model_name_or_path.split('/')[-1]]
    
    Ttokenizer = Ttokenizer_class.from_pretrained(args.teacher_tokenizer_name)
    Stokenizer = Stokenizer_class.from_pretrained(args.student_tokenizer_name)

    eval_dataset = TextDataset(Ttokenizer, Stokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    model.eval()
    logits = []  
    y_trues = []
    
    for batch in tqdm(eval_dataloader):
        (ids_1_t, _, _, ids_2_t, _, _, 
         ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, 
         labels) = [x.to(args.device) for x in batch]
         
        with torch.no_grad():
            lm_loss, logit, _ = model(ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels)
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
        "eval_threshold": best_threshold,
    }
    return result

def test(args, model, tokenizer, best_threshold=0):
    _, _, Ttokenizer_class = MODEL_CLASSES[args.teacher_model_name_or_path.split('/')[-1]]
    _, _, Stokenizer_class = MODEL_CLASSES[args.student_model_name_or_path.split('/')[-1]]
    Ttokenizer = Ttokenizer_class.from_pretrained(args.teacher_tokenizer_name)
    Stokenizer = Stokenizer_class.from_pretrained(args.student_tokenizer_name)

    eval_dataset = TextDataset(Ttokenizer, Stokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    
    latency_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1, num_workers=0)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test (Latency Mode) *****")
    model.eval()

    if len(eval_dataset) > 0:
        dummy_batch = next(iter(latency_dataloader))
        dummy_batch = [x.to(args.device) for x in dummy_batch]
        (ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels) = dummy_batch[6:]
        
        for _ in range(10):
            with torch.no_grad():
                 _ = model(ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels)

    logger.info("Starting pure inference timing...")
    t_start = time.time()
    
    for batch in latency_dataloader:
        batch = [x.to(args.device) for x in batch]

        (ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels) = batch[6:]
         
        with torch.no_grad():
            _ = model(ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels)
            
    t_end = time.time()
    
    total_time = t_end - t_start
    avg_latency = total_time / len(eval_dataset)
    
    print(f"==================================================")
    print(f"  [Total Inference Time]: {total_time:.4f} s")
    print(f"  [Avg Latency / Sample]: {avg_latency:.6f} s  <--- 请将此数值填入论文表格")
    print(f"==================================================")

    score_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)
    
    logits = []  
    y_trues = []
    
    for batch in tqdm(score_dataloader, desc="Calculating Metrics"):
        batch = [x.to(args.device) for x in batch]
        (ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels) = batch[6:]
         
        with torch.no_grad():
            lm_loss, logit, _ = model(ids_1_s, pos_1_s, mask_1_s, ids_2_s, pos_2_s, mask_2_s, labels)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

    logits = np.concatenate(logits, 0)
    y_preds = logits[:, 1] > best_threshold
    y_trues = np.concatenate(y_trues, 0)
    
    from sklearn.metrics import recall_score, precision_score, f1_score
    recall = recall_score(y_trues, y_preds, average='macro')
    precision = precision_score(y_trues, y_preds, average='macro')   
    f1 = f1_score(y_trues, y_preds, average='macro')             
    
    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "inference_latency": avg_latency 
    }
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    
    return result

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
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
    
    parser.add_argument("--code_length", default=448, type=int) 
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
    parser.add_argument("--data_flow_length", default=64, type=int) 
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    set_seed(args)

    # Classes setup
    Tconfig_class, Tmodel_class, Ttokenizer_class = MODEL_CLASSES[args.teacher_model_name_or_path.split('/')[-1]]
    Sconfig_class, Smodel_class, Stokenizer_class = MODEL_CLASSES[args.student_model_name_or_path.split('/')[-1]]

    # Load Teacher
    Tconfig = Tconfig_class.from_pretrained(args.teacher_config_name if args.teacher_config_name else args.teacher_model_name_or_path)
    Tconfig.num_labels = 2
    Ttokenizer = Ttokenizer_class.from_pretrained(args.teacher_tokenizer_name)
    Tmodel_raw = Tmodel_class.from_pretrained(args.teacher_model_name_or_path)
    Tmodel = TeacherModel(Tmodel_raw, Tconfig, Ttokenizer, args)
    
    # Load Student
    Sconfig = Sconfig_class.from_pretrained(args.student_config_name if args.student_config_name else args.student_model_name_or_path)
    Sconfig.num_labels = 2
    Stokenizer = Stokenizer_class.from_pretrained(args.student_tokenizer_name)
    Smodel_raw = Smodel_class.from_pretrained(args.student_model_name_or_path)
    Smodel = StudentModel(Smodel_raw, Sconfig, Stokenizer, args)

    # Load Teacher Weights
    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    teacher_weights_path = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    if os.path.exists(teacher_weights_path):
        Tmodel.load_state_dict(torch.load(teacher_weights_path), strict=False)
    else:
        logger.warning(f"Teacher checkpoint not found at {teacher_weights_path}, using pretrained weights.")

    Tmodel.to(args.device)
    Smodel.to(args.device)

    # 初始化 results
    results = {}

    if args.do_train:
        t1 = time.time()
        # Pass BOTH tokenizers to dataset
        train_dataset = TextDataset(Ttokenizer, Stokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, Tmodel, Smodel, Stokenizer)
        t2 = time.time()
        print("training time: ", t2 - t1, "s")

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/DistilroBERT/student_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        Smodel.load_state_dict(torch.load(output_dir))
        Smodel.to(args.device)
        results = evaluate(args, Smodel, Stokenizer)
        
    if args.do_test:
        t1 = time.time()
        checkpoint_prefix = 'checkpoint-best-f1/DistilroBERT/student_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        Smodel.load_state_dict(torch.load(output_dir))
        Smodel.to(args.device)
        test(args, Smodel, Stokenizer, best_threshold=0.5)
        t2 = time.time()
        print("elphasing time: ", t2 - t1, "s")

    return results

if __name__ == "__main__":
    main()