"""
Task2: Pretrained SOTA model to determine whether Yelp reviews are positive or negative
https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

"""
import torch
from torchtext.datasets import YelpReviewPolarity
from torchtext.data.utils import get_tokenizer
from transformers import BertModel, BertTokenizer


import torch
import torchtext
from torch.utils.data import DataLoader, Dataset

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batchSize = 32



# Define a collate function that applies the tokenizer to each batch
def collate_fn(batch):
    # Convert the list of inputs and labels to separate lists
    labels, inputs = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, padding=True, max_length =16, truncation=True, return_tensors="pt")

    # Convert the labels to a PyTorch tensor
    labels = torch.tensor(labels)

    # Return the tokenized inputs and labels as a tuple
    return tokenized_inputs, labels


# Load YelpReviewPolarity dataset, split into train and test
# datatype: ShardingFilterIterDataPipe
train_data, test_data = YelpReviewPolarity(root='.data', split=('train', 'test'))
test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
#




# Create a DataLoader that uses the collate function
# batch_size = 2
# num_workers = 4
# shuffle = True
# loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn)
#



# # Create the datasets
# train_data = YelpDataset([d.text for d in train_data], [d.label for d in train_data], tokenizer)
# test_data = YelpDataset([d.text for d in test_data], [d.label for d in test_data], tokenizer)
# train_data, test_data = YelpReviewPolarity.splits(text_field=torchtext.data.Field(tokenize='spacy'), label_field=torchtext.data.LabelField(dtype=torch.float))

# Load YelpReviewPolarity dataset
# val_size = 20000
# train_size = len(train_data) - val_size
# train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
#




# dataloader
from torch.utils.data import DataLoader

#test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=True)

from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn

def train(model, train_data, learning_rate, epochs):
    #train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    #val_loader = DataLoader(val_data, batch_size=batchSize, shuffle=True)

    # for batch in train_loader:
    #     inputs, labels = batch
    #

    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)


    model = model.to(device)
    criterion = criterion.to(device)

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for train_input, train_label in tqdm(train_loader):

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

        total_acc_val = 0
        total_loss_val = 0

        # with torch.no_grad():
        #
        #     for val_input, val_label in val_loader:
        #         val_label = val_label.to(device)
        #         mask = val_input['attention_mask'].to(device)
        #         input_id = val_input['input_ids'].squeeze(1).to(device)
        #
        #         output = model(input_id, mask)
        #
        #         batch_loss = criterion(output, val_label.long())
        #         total_loss_val += batch_loss.item()
        #
        #         acc = (output.argmax(dim=1) == val_label).sum().item()
        #         total_acc_val += acc

        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f}' )

class BertClassifier(nn.Module):

    def __init__(self, freeze_bert=False, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
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

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

train(model, train_data, LR, EPOCHS)
