from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class MABert(nn.Module):
    def __init__(self, bert, num_classes, bert_trainable=True, device=0):
        super(MABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.class_weight.requires_grad = True

        self.discriminator = Parameter(torch.Tensor(1, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.discriminator.requires_grad = True

        self.relu = nn.ReLU()
        self.output = nn.Softmax(dim=-1)

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0] #N, L, hidden_size

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool() .byte()
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1-masks.byte()), torch.tensor(-np.inf))

        attention = F.softmax(attention, -1) #N, labels_num, L

        attention_out = torch.matmul(attention,token_feat)   # N, labels_num, hidden_size
        attention_out_feat = attention_out
        attention_out = attention_out * self.class_weight
        attention_out = torch.sum(attention_out, -1)

        logit = torch.sigmoid(attention_out)

        feat = feat * self.class_weight
        feat = torch.sum(feat, -1)

        prob = torch.cat((torch.sum(feat, -1, keepdim=True), torch.sum(attention_out, -1, keepdim=True)),-1)

        prob = self.output(prob)

        return prob[:,1], logit, prob[:,0], attention

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.class_weight, 'lr': lr},
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.Linear1.parameters(), 'lr': lr},
            {'params': self.Linear2.parameters(), 'lr': lr},
        ]


class Generator(nn.Module):
    def __init__(self, bert,num_classes, hidden_dim=768, input_dim=768, num_hidden_generator=2, hidden_dim_generator=2000):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2) #nn.Sigmoid()#

        self.num_classes = num_classes

        input_dim += num_classes

        self.num_hidden_generator = num_hidden_generator

        self.hidden_list_generator = nn.ModuleList()
        for i in range(num_hidden_generator):
            dim = input_dim if i == 0 else hidden_dim_generator
            self.hidden_list_generator.append(nn.Linear(dim, hidden_dim_generator).cuda(0))

        self.output = nn.Linear(hidden_dim_generator, hidden_dim)

    def forward(self, feat, encoded_tag, tag_mask):

        feat = feat.expand(feat.shape[0], self.num_classes,feat.shape[2])
        tag_embedding = torch.eye(self.num_classes).cuda(0).unsqueeze(0).expand(feat.shape[0],self.num_classes,self.num_classes)
        x = torch.cat((feat,tag_embedding),-1)
        # x = feat
  
        for i in range(self.num_hidden_generator):
            x = self.hidden_list_generator[i](x)
            x = self.act(x)

        y = self.output(x)

        return y

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_generator.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]


class MLPBert(nn.Module):
    def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable):
        super(MLPBert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.name_layer = nn.Linear(768, 1024)
        self.act = nn.Tanh()#nn.ReLU()

        self.hidden_dim = 512

        self.num_classes = num_classes
        self.hidden_layer_num = hidden_layer_num
        self.hidden_list = nn.ModuleList()

        self.conv1 = torch.nn.Conv2d(1,32,(3,3),padding=1)
        self.conv2 = torch.nn.Conv2d(32,1,(1,1))
        self.dropout = nn.Dropout(p=0.5)

        self.layer1=nn.LSTM(input_size=768, hidden_size=self.hidden_dim, \
                        num_layers=1,batch_first=True, \
                        bidirectional=True)

        self.output = nn.Linear(1024, num_classes)

        self.w1 = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True).cuda(0) #
        self.w2 = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True).cuda(0) #

    def init_hidden(self,batch_size ):
            # 定义初始的hidden state
            return (torch.rand(1*2,batch_size,self.hidden_dim).uniform_(0, 1).cuda(0), #randn
                    torch.rand(1*2,batch_size,self.hidden_dim).uniform_(0, 1).cuda(0))
            # return (torch.zeros(1*2,batch_size,self.hidden_dim).cuda(0), #randn
            #         torch.zeros(1*2,batch_size,self.hidden_dim).cuda(0))

    def forward(self, ids, token_type_ids, attention_mask, title_ids, title_token_type_ids,  title_attention_mask, encoded_tag, tag_mask, feat):

        title_feat = self.bert(title_ids,
                               token_type_ids=title_token_type_ids,
                               attention_mask=title_attention_mask)[1]

        title_feat = self.name_layer(title_feat)
        title_feat = self.act(title_feat)
        title_feat = nn.Dropout(p=0.1)(title_feat)

        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]

        x = token_feat

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.Dropout(p=0.1)(x)
        x = self.conv2(x)
        x = x.squeeze(1)
        
        hidden1 = self.init_hidden(token_feat.shape[0])
        out,hidden1 = self.layer1(x,hidden1)

        # x = out[:,-1,:]

        x = hidden1[1].permute(1,0,2)
        x = x.reshape(token_feat.shape[0],-1)
        x = nn.Dropout(p=0.1)(x)

        x = self.w1 * title_feat + self.w2 * x

        y = torch.sigmoid(self.output(x))
        return None, y, None, None

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.conv1.parameters(), 'lr': lr},
            {'params': self.conv2.parameters(), 'lr': lr},
            {'params': self.layer1.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]

# class MLPBert(nn.Module):
#     def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable=True):
#         super(MLPBert, self).__init__()

#         self.add_module('bert', bert)
#         if not bert_trainable:
#             for m in self.bert.parameters():
#                 m.requires_grad = False

#         self.num_classes = num_classes
#         self.hidden_layer_num = hidden_layer_num
#         self.hidden_list = nn.ModuleList()
#         for i in range(hidden_layer_num):
#             if i == 0:
#                 self.hidden_list.append(nn.Linear(768, hidden_dim))
#             else:
#                 self.hidden_list.append(nn.Linear(hidden_dim, hidden_dim))
#         self.output = nn.Linear(hidden_dim, num_classes)
#         self.act = nn.ReLU()

#     def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):

#         token_feat = self.bert(ids,
#                                token_type_ids=token_type_ids,
#                                attention_mask=attention_mask)[0]
#         sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
#                         / torch.sum(attention_mask, dim=1, keepdim=True)

#         x = sentence_feat
#         for i in range(self.hidden_layer_num):
#             x = self.hidden_list[i](x)
#             x = self.act(x)
#         y = torch.sigmoid(self.output(x))
#         return None, y, None, None

#     def get_config_optim(self, lr, lrp):
#         return [
#             {'params': self.bert.parameters(), 'lr': lrp},
#             {'params': self.hidden_list.parameters(), 'lr': lr},
#             {'params': self.output.parameters(), 'lr': lr},
#         ]