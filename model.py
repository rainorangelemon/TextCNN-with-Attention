# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 09月 21日 星期五 10:25:44 CST
# ***
# ************************************************************************************/

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=4, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N=1):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def create_mask(input, pad_id):
    input_msk = (input != pad_id).unsqueeze(1)
    return input_msk


class SentenceEncoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, x, mask):
        for i in range(self.N):
            x = (self.layers[i])(x, mask)
        x = self.norm(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, args, pad_id):
        super(TextCNN, self).__init__()
        self.args = args
        self.pad_id = pad_id

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.sentence_encoder = SentenceEncoder(len(Ks) * Co, 1, 3)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        mask = create_mask(x,self.pad_id)

        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.sentence_encoder(x, mask)
        logit = self.fc1(x)  # (N, C)
        return logit


def train(train_iter, model, args):
    """
    Train Text CNN Model
    """
    def save_model(model, steps):
        if not os.path.isdir("logs"):
            os.makedirs("logs")
        save_path = 'logs/textcnn.model-{}'.format(steps)
        torch.save(model, save_path)

    def save_steps(epochs):
        n = int((epochs + 1) / 10)
        if n < 10:
            n = 10
        n = 10 * int((n + 9) / 10)  # round to 10x times
        return n

    print("Start training ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    if args.cuda:
        model.cuda()

    save_interval = save_steps(args.epochs)

    for epoch in range(1, args.epochs+1):
        training_loss = 0.0
        training_acc = 0.0
        training_count = 0.0

        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature = feature.data.t()
            target = target.data.sub_(1)  # batch first, index align

            # print("-"*80)
            # print("feature: ", feature, feature.size())
            # print("target: ", target, target.size())

            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            # print("logit:", logit, logit.size())

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

            training_loss += loss.item()
            training_acc += corrects.item()
            training_count += batch.batch_size

        training_loss /= training_count
        training_acc /= training_count
        accuracy = 100.0 * training_acc
        print('Training epoch [{}/{}] - loss: {:.6f}  acc: {:.2f}%'.format(
            epoch, args.epochs, training_loss, accuracy))

        if epoch % save_interval == 0:
            save_model(model, epoch)
    print("Training finished.")


def eval(data_iter, model, args):
    print("Start evaluating ...")
    model.eval()
    if args.cuda:
        model.cuda()

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%'.format(avg_loss, accuracy))
    print("Evaluating finished.")
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    if cuda_flag:
        model.cuda()

    text = text_field.preprocess(text)

    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.LongTensor(text)
    if cuda_flag:
        x = x.cuda()
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0] + 1]
