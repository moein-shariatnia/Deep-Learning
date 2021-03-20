![huggingface](https://huggingface.co/favicon.ico)

# HuggingFace Tutorial - Sequence Classification with DistilBERT and PyTorch


## Introduction

* A week ago, I decided to learn more about NLP beacuse my previous year was mainly focused on Computer Vision applications and I couldn't put much time on NLP. I watched NVIDIA GrandMaster Series episode "[Grandmaster Series – Building World-Class NLP Models with Transformers and Hugging Face](https://youtu.be/PXc_SlnT2g0)" and I realized that this amazing **HuggingFace** library makes it really easy to use state-of-the-art models and get perfect results. I was also afraid of **Transformers** because I thought they are too complicated and it's not easy to understand them! But, I was totally wrong! I watched a bunch of good tutorials on Transformers and how to code them on YouTube which I'm going to introduce them bellow. I also share some of the good tutorials on HuggingFace itself which I found there:


1. Pytorch Transformers from Scratch (Attention is all you need):
[YouTube Link](https://youtu.be/U0s0f995w14)
2. Grandmaster Series – Building World-Class NLP Models with Transformers and Hugging Face: [YouTube Link](https://youtu.be/PXc_SlnT2g0)
3. Deep learning for (almost) any text classification problem (binary, multi-class, multi-label): [YouTube Link](https://youtu.be/oreIJQZ40H0)

I also found HuggingFace Official Examples really helpful: [Link](https://huggingface.co/transformers/examples.html)

---

Although you can watch them and be good to go with your NLP/Transformer journey, I though it will be helpful to make a tutorial on using HuggingFace models based on the things I've learned so far and make it easier to start this journey for others; because, some of the details are missing in these tutorials and I'm gonna focus more on them in this one. So, stay tuned!


```python
import os
import copy
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn


from sklearn.model_selection import train_test_split, KFold

# importing HuggingFace transformers library which is all we need to get SOTA results :)
import transformers
from transformers import get_linear_schedule_with_warmup

print(transformers.__version__)
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:5: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
      """
    

    4.2.2
    

# Building A Custom PyTorch Dataset

* One important thing that I was looking for was how to build an efficient PyTorch Dataset from my own data (actually, Kaggle data in this case!). Because, in the [HuggingFace official examples](https://huggingface.co/transformers/examples.html) they were using their own datasets library with ready-to-use datasets but most of the time, we need to build our own datasets with our own data. 

* So, I searched and found this amazing short tutorial from HuggingFace: [Fine-tuning with custom datasets](https://huggingface.co/transformers/custom_datasets.html). The following code uses the idea from this tutorial on building a custom dataset:


```python
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, mode="train", max_length=None):
        self.dataframe = dataframe
        if mode != "test":
            self.targets = dataframe['target'].values
        texts = list(dataframe['text'].values)
        self.encodings = tokenizer(texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=max_length)
        self.mode = mode
        
        
    def __getitem__(self, idx):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        # when testing, there are no targets so we won't do the following
        if self.mode != "test":
            item['labels'] = torch.tensor(self.targets[idx])
        return item
    
    def __len__(self):
        return len(self.dataframe)
```

Just a wrapper to easier build the Dataset and DataLoader


```python
def make_loaders(dataframe, tokenizer, mode="train", max_length=None):
    dataset = TweetDataset(dataframe, tokenizer, mode, max_length=max_length)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=options.batch_size, 
                                             shuffle=True if mode == "train" else False,
                                             num_workers=options.num_workers)
    return dataloader
```

# Custom Classification Model based on DistilBERT

* This part needs some explanation. As the title said in the beginning of this tutorial, we are going to use DistilBERT model. But as you might have guessed, DistilBERT is a Language Model which needs to be fine-tuned on a final task of interestl; here being Classification. For those of you that are familiar with Computer Vision, it's like using a fancy ResNet model pre-trained on ImageNet and then building a custom head for our specific task!

* So, we need to build that custom head here. Before doing so, we need to know something about BERT family models (I recommend to study [original BERT paper](https://arxiv.org/abs/1810.04805)). In the paper, they introduce some special tokens named [CLS] and [SEP] which they add to the sequence which is being fed to the model. [CLS] is used at the beginning of the sequence and [SEP] tokens are used to notify the end of each part in a sequence (a sequence which is going to be fed to BERT model can be made up of two parts; e.x question and corresponding text). 
 
* In the paper they explain that they use [CLS] hidden state representation to do classification tasks for the sequence. So, in our case, we are going to the same. DistilBERT model will produce a vector of size 768 as a hidden representation for this [CLS] token and we will give it to some nn.Linear layers to do our own specific task. 


```python
class CustomModel(nn.Module):
    def __init__(self,
                 bert_model,
                 num_labels, 
                 bert_hidden_dim=768, 
                 classifier_hidden_dim=768, 
                 dropout=None):
        
        super().__init__()
        self.bert_model = bert_model
        # nn.Identity does nothing if the dropout is set to None
        self.head = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                  nn.Linear(classifier_hidden_dim, num_labels))
    
    def forward(self, batch):
        # feeding the input_ids and masks to the model. These are provided by our tokenizer
        output = self.bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state # shape: (batch_size, seq_length, bert_hidden_dim)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation 
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.head(CLS_token_state)
        return logits
```

# Training and Evaluation functions

* There is nothing NLP/Transformer specific here! Just some functions to the training and eval loops and print stuff while the model is being trained

* Pay attention to the comments in the codes below; I've explained the parts that could be confusing or new to you!


```python
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def one_epoch(model, criterion, loader, device, optimizer=None, lr_scheduler=None, mode="train", step="batch"):
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    tqdm_object = tqdm(loader, total=len(loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(batch)
        loss = criterion(preds, batch['labels'])
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()
                
        count = batch['input_ids'].size(0)
        loss_meter.update(loss.item(), count)
        
        accuracy = get_accuracy(preds.detach(), batch['labels'])
        acc_meter.update(accuracy.item(), count)
        if mode == "train":
            tqdm_object.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg, lr=get_lr(optimizer))
        else:
            tqdm_object.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg)
    
    return loss_meter, acc_meter

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def get_accuracy(preds, targets):
    """
    preds shape: (batch_size, num_labels)
    targets shape: (batch_size)
    """
    preds = preds.argmax(dim=1)
    acc = (preds == targets).float().mean()
    return acc
```


```python
def train_eval(epochs, model, train_loader, valid_loader, 
               criterion, optimizer, device, options, lr_scheduler=None):
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print("*" * 30)
        print(f"Epoch {epoch + 1}")
        current_lr = get_lr(optimizer)
        
        model.train()
        train_loss, train_acc = one_epoch(model, 
                                          criterion, 
                                          train_loader, 
                                          device,
                                          optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          mode="train",
                                          step=options.step)                     
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = one_epoch(model, 
                                              criterion, 
                                              valid_loader, 
                                              device,
                                              optimizer=None,
                                              lr_scheduler=None,
                                              mode="valid")
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'{options.model_path}/{options.model_save_name}')
            print("Saved best model!")
        
        # or you could do: if step == "epoch":
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(valid_loss.avg)
            # if the learning rate changes by ReduceLROnPlateau, we are going to
            # reload our previous best model weights and start from there with a lower LR
            if current_lr != get_lr(optimizer):
                print("Loading best model weights!")
                model.load_state_dict(torch.load(f'{options.model_path}/{options.model_save_name}', 
                                                 map_location=device))
        

        print(f"Train Loss: {train_loss.avg:.5f}")
        print(f"Train Accuracy: {train_acc.avg:.5f}")
        
        print(f"Valid Loss: {valid_loss.avg:.5f}")
        print(f"Valid Accuracy: {valid_acc.avg:.5f}")
        print("*" * 30)
```


```python
class Options:
    model_name = 'distilbert-base-uncased'
    batch_size = 64
    num_labels = 2
    epochs = 10
    num_workers = 2
    learning_rate = 3e-5
    scheduler = "ReduceLROnPlateau"
    patience = 2
    dropout = 0.5
    model_path = "."
    max_length = 140
    model_save_name = "model.pt"
    n_folds = 5
```

# Taking care of Cross Validation


```python
def make_folds(dataframe, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (_, valid_idx) in enumerate(kf.split(X=dataframe['id'])):
        dataframe.loc[valid_idx, 'fold'] = i
    return dataframe
```


```python
def one_fold(fold, options):  
    print(f"Training Fold: {fold}")
    
    # Here, we load the pre-trained DistilBERT model from transformers library
    bert_model = transformers.DistilBertModel.from_pretrained(options.model_name)
    # Loading the corresponding tokenizer from HuggingFace by using AutoTokenizer class.
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name, use_fast=True)
    
    dataframe = pd.read_csv("./input/train.csv")
    dataframe = make_folds(dataframe, n_splits=options.n_folds)
    train_dataframe = dataframe[dataframe['fold'] != fold].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe['fold'] == fold].reset_index(drop=True)

    train_loader = make_loaders(train_dataframe, tokenizer, "train", options.max_length)
    valid_loader = make_loaders(valid_dataframe, tokenizer, "valid", options.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel(bert_model, options.num_labels, dropout=options.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    if options.scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode="min", 
                                                                  factor=0.5, 
                                                                  patience=options.patience)
        
        # when to step the scheduler: after an epoch or after a batch
        options.step = "epoch"
        
    elif options.scheduler == "LinearWarmup":
        num_train_steps = len(train_loader) * options.epochs
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=0, 
                                                       num_training_steps=num_train_steps)
        
        # when to step the scheduler: after an epoch or after a batch
        options.step = "batch"
    
    criterion = nn.CrossEntropyLoss()
    options.model_save_name = f"model_fold_{fold}.pt"
    train_eval(options.epochs, model, train_loader, valid_loader,
               criterion, optimizer, device, options, lr_scheduler=lr_scheduler)
```


```python
def train_folds(options):
    n_folds = options.n_folds
    for i in range(n_folds):
        one_fold(fold=i, options=options)
```


```python
options = Options()
train_folds(options)
```


```python
def test_one_model(options):  
    test_dataframe = pd.read_csv("./input/test.csv")

    bert_model = transformers.DistilBertModel.from_pretrained(options.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name, use_fast=True)
    
    test_loader = make_loaders(test_dataframe, tokenizer, mode="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel(bert_model, options.num_labels, dropout=options.dropout).to(device)
    model.load_state_dict(torch.load(f"{options.model_path}/{options.model_save_name}", 
                                     map_location=device))
    model.eval()
    
    all_preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch)
            if all_preds is None:
                all_preds = preds
            else:
                all_preds = torch.cat([all_preds, preds], dim=0)
    
    return all_preds
```


```python
def test_all_models(options):
    n_folds = options.n_folds
    all_model_preds = []
    for fold in range(n_folds):
        options.model_save_name = f"model_fold_{fold}.pt"
        all_preds = test_one_model(options)
        all_model_preds.append(all_preds)
    
    all_model_preds = torch.stack(all_model_preds, dim=0)
    print(all_model_preds.shape)
    # I will return the mean of the final predictions of all the models
    # You could do other things like 'voting' between the five models
    return all_model_preds.mean(0)
```


```python
all_preds = test_all_models(options)
predictions = all_preds.argmax(dim=1).cpu().numpy()
sample_submission = pd.read_csv("./input/sample_submission.csv")
sample_submission['target'] = predictions
sample_submission.to_csv("sample_submission.csv", index=False)
pd.read_csv("sample_submission.csv")
```

### Thanks for reading my tutorial. I'll be really happy to know what you think about it and if learned something new! Happy Learning!
