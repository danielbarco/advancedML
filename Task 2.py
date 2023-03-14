"""
Task2: Pretrained SOTA model to determine whether Yelp reviews are positive or negative
Currently using accuracy as evaluation metric, but will change to F1 score or AUC score
https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

"""

from torchtext.datasets import YelpReviewPolarity
from transformers import BertModel, BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import time
import torch
from torch.utils.data import DataLoader, Dataset

# Define a collate function that applies the tokenizer to each batch
def collate_fn(batch):
    # Convert the list of inputs and labels to separate lists
    labels, inputs = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, padding=True, add_special_tokens=True,  truncation=True, return_tensors="pt")
    tokenizer(inputs, padding='max_length', max_length=512, add_special_tokens=True, truncation=True, return_tensors="pt")
    # Convert the labels to a PyTorch tensor
    labels = torch.tensor(labels)

    # Return the tokenized inputs and labels as a tuple
    return tokenized_inputs, labels

class BertClassifier(nn.Module):
    """BERT model for classification.This module is composed of the BERT model with a linear layer on top of the pretrained BERT model.
    """
    def __init__(self, freeze_bert=False, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze the BERT parameters if desired
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        prob = self.softmax(linear_output)

        return prob

class TimingCallback(torch.nn.Module):
    def __init__(self,total_epochs ):
        super(TimingCallback, self).__init__()
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def forward(self, epoch, total_epochs):
        elapsed_time = time.time() - self.start_time
        print("Epoch [{}/{}] - time: {:.2f}s".format(epoch + 1, total_epochs, elapsed_time))

    def on_epoch_end(self, epoch, logs=None):
        self.forward(epoch, self.total_epochs)

def train(model, train_data):
    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader)):
            if i == sample:
                break
            train_input, train_label = data
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask) #

            batch_loss = criterion(output, (train_label-1).long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label-1).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        timing_callback.on_epoch_end(epoch, epochs)
        print(f'Train Loss: {total_loss_train / (i*batchSize): .3f} \
                | Train Accuracy: {total_acc_train / (i*batchSize): .3f}' )

def evaluate(model, test_data):
    test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    total_loss_test = 0
    total_acc_test = 0
    with torch.no_grad():

        for i, data in tqdm(enumerate(test_loader)):
            if i == sample:
                break
            test_input, test_label = data
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            batch_loss = criterion(output, (test_label-1).long())
            total_loss_test += batch_loss.item()

            acc = (output.argmax(dim=1) == test_label-1).sum().item()
            total_acc_test += acc

    print(f'Test Loss: {total_loss_test / (i * batchSize): .3f} \
            | Test Accuracy: {total_acc_test / (i * batchSize): .3f}')

# initialize the parameters
batchSize = 16
epochs = 10
lr = 1e-6
sample = 1000 # number of batches
model = BertClassifier(freeze_bert=False) # initialize the model

# Load YelpReviewPolarity dataset, split into train and test
# datatype: ShardingFilterIterDataPipe
train_data, test_data = YelpReviewPolarity(root='.data', split=('train', 'test'))

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Train the model and evaluate its performance
timing_callback = TimingCallback(epochs)
train(model, train_data)
evaluate(model, test_data)

