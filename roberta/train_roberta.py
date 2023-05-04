from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import random
import time
import os
from sklearn.metrics import f1_score, roc_auc_score

seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# define model

class BERTModelForClassification(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask)[0] # batch, seq_len, emb_dim

        # get [CLS] embeddings
        cls_embeddings = hidden_states[:,0,:]
        logits = self.linear(cls_embeddings)

        outputs = torch.sigmoid(logits)
        return outputs        


# define loader

class PropaFakeDataset(Dataset):
    def __init__(self, jsonl_path):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        self.data = []
        
        for line in open(jsonl_path,'r'):
            inst = json.loads(line)
            label = inst['label']
            inputs = self.tokenizer(inst['txt'], max_length=args.max_sequence_length, padding="max_length", truncation=True)
            self.data.append({
                'input_ids':inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'label': label
            })
            
        
    def __len__(self):
        # 200K datapoints
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]['input_ids'], self.data[idx]['attention_mask'], self.data[idx]['label']
    
    def collate_fn(self, batch):
        # print(batch)
        input_ids = torch.cuda.LongTensor([inst[0] for inst in batch])
        attention_masks = torch.cuda.LongTensor([inst[1]for inst in batch])
        labels = torch.cuda.FloatTensor([inst[2]for inst in batch])

        return input_ids, attention_masks, labels

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--max_sequence_length', default=512, type=int)
parser.add_argument('--model_name', default='roberta-large')
parser.add_argument('--data_dir', default='../../data/')
parser.add_argument('--warmup_epoch', default=5, type=int)
parser.add_argument('--max_epoch', default=30, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--eval_batch_size', default=2, type=int)
parser.add_argument('--accumulate_step', default=8, type=int)
parser.add_argument('--output_dir', required=True)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_dir = os.path.join(args.output_dir, timestamp)
os.makedirs(output_dir)
# init model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = BERTModelForClassification(args.model_name).cuda()

# init loader
train_set = PropaFakeDataset(os.path.join(args.data_dir,'politifact_test.jsonl'))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
dev_set = PropaFakeDataset(os.path.join(args.data_dir,'dev.jsonl'))
dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn)
test_set = PropaFakeDataset(os.path.join(args.data_dir,'snopes_test.jsonl'))
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)

# define loss
critera = nn.BCELoss()

state = dict(model=model.state_dict())

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': 5e-5, 'weight_decay': 1e-05
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
        'lr': 1e-3, 'weight_decay': 0.001
    },
    
]

batch_num = len(train_set) // (args.batch_size * args.accumulate_step)
+ (len(train_set) % (args.batch_size * args.accumulate_step) != 0)

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*args.warmup_epoch,
                                           num_training_steps=batch_num*args.max_epoch)

best_dev_accuracy = 0
model_path = os.path.join(output_dir,'best.pt')
for epoch in range(args.max_epoch):
    training_loss = 0
    model.train()
    for batch_idx, (input_ids, attn_mask, labels) in enumerate(tqdm(train_loader)):        
        

        
        outputs = model(input_ids, attention_mask=attn_mask).view(-1)

        # loss
        
        loss = critera(outputs, labels)
        loss.backward()
        training_loss += loss.item()
        if (batch_idx + 1) % args.accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)
        
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

    print(f"Trainin Loss: {training_loss:4f}" )
    # train the last batch
    if batch_num % args.accumulate_step != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    # validation
    with torch.no_grad():
        model.eval()
        dev_outputs = []
        dev_labels = []
        for _, (input_ids, attn_mask, labels) in enumerate(dev_loader):
            outputs = model(input_ids, attention_mask=attn_mask).view(-1)
            dev_outputs.append(outputs) 
            dev_labels.append(labels)
        dev_outputs = torch.cat(dev_outputs, dim=0) # n_sample,
        dev_labels = torch.cat(dev_labels, dim=0) # n_sample,
        
        dev_outputs_bool = dev_outputs > 0.5
        dev_labels_bool = dev_labels == 1 # convert to float tensor
        

        dev_accuracy = torch.sum(dev_labels_bool == dev_outputs_bool) / len(dev_labels)
        
        dev_auc = roc_auc_score(dev_labels.cpu().numpy(), dev_outputs.detach().cpu().numpy())
        print(f"Dev AUC: {dev_auc}. ")
        
        dev_f1 = f1_score(dev_labels.cpu().numpy(), np.array([1 if l > 0.5 else 0 for l in dev_outputs]))
        print(f"Dev F1: {dev_f1}. ")
        
        dev_accuracy = dev_auc

        if dev_accuracy > best_dev_accuracy:
            
            print(f"Saving to {model_path}")
            best_dev_accuracy = dev_accuracy
            torch.save(state, model_path)

        print(f"Epoch {epoch} dev accuracy: {dev_accuracy * 100:.2f}. Best dev accuracy: {best_dev_accuracy*100:.2f}.")            


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'], strict=True)    
test_output_file = os.path.join(output_dir, 'test_pred.json')

with torch.no_grad():
    model.eval()
    test_outputs = []
    test_labels = []
    
    for _, (input_ids, attn_mask, labels) in enumerate(test_loader):
        outputs = model(input_ids, attention_mask=attn_mask).view(-1)
        test_outputs.append(outputs) 
        test_labels.append(labels)
    test_outputs = torch.cat(test_outputs, dim=0) # n_sample,
    test_labels = torch.cat(test_labels, dim=0) # n_sample,
    
    test_outputs_bool = test_outputs > 0.5
    test_labels_bool = test_labels == 1 # convert to float tensor

    test_accuracy = torch.sum(test_outputs_bool == test_labels_bool) / len(test_labels)
    print(f"Epoch {epoch} test accuracy: {test_accuracy*100:.2f}. ")
    
    test_auc = roc_auc_score(test_labels.cpu().numpy(), test_outputs.detach().cpu().numpy())
    print(f"Test AUC: {test_auc}. ")
    
    test_f1 = f1_score(test_labels.cpu().numpy(), np.array([1 if l > 0.5 else 0 for l in test_outputs]))
    print(f"Test F1: {test_f1}. ")
    
    test_outputs = [float(o) for o in test_outputs]
       
    with open(test_output_file,'w') as f:
        json.dump({'output':test_outputs}, f)
print("model path:", model_path)