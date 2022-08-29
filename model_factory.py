################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

from vocab import *

import torch
from torch import LongTensor
from torch.nn import Embedding, LSTMCell, RNNCell
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    temperature = config_data['generation']['temperature']
    deterministic = config_data['generation']['deterministic']
    batch_size = config_data['dataset']['batch_size']
    max_length = config_data['generation']['max_length']

    #change the first to be 2 if we want to use the second architecture of model
    encoder_decoder = Encoder_Decoder(1, model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size)
    # encoder_decoder = Encoder_Decoder(2, model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size)
    return encoder_decoder

#The encoder-decoder structure
class Encoder_Decoder(nn.Module):

    def __init__(self, arch,model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(embedding_size)

        self.arch = arch
        #determine whether to use the combined feature or word embedding later
        if arch == 1:
            #choose model if using architecture1
            self.decoder = Decoder_arch1(model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size)
        else:
            self.decoder = Decoder_arch2(vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size)

    def forward(self, images, captions):
        encoded = self.encoder(images) 
        #forward features to the decoder
        out = self.decoder(encoded, captions)
        return out

    def generate(self, images):
        #change the dimensions
        images = images.unsqueeze(0)
        encoded = self.encoder(images)
        #forward features to the decoder
        out = self.decoder.generate(encoded)
        return out

#The encoder structure using ResNet50 transfer learning
class Encoder(nn.Module):

    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        # transfer learning resnet50
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        params = list(model.children())[:-1]
        self.cnn = nn.Sequential(*params)

        # add a fc linear layer and input embedding_size
        self.fc = nn.Linear(in_features=2048, out_features=embedding_size)

        #add initialization
        nn.init.xavier_uniform(self.fc.weight)

    def forward(self, images):
        #pass images into this encoder
        out = self.cnn(images)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#The architecture1 decoder structure with LSTM/RNN cell
class Decoder_arch1(nn.Module):
    def __init__(self, model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size):
        super(Decoder_arch1, self).__init__()

        self.max_length = max_length
        self.batch_size = batch_size
        self.deterministic = deterministic
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        #the embedding dictionary length
        self.dict_len = len(vocab)
        #construct the decoder
        self.embedding = nn.Embedding(self.dict_len, embedding_size)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(hidden_size, self.dict_len)
        #if the model has been trained
        self.past = None

        #determine if the model uses LSTM/RNN cell
        self.model_type = model_type
        if model_type == "LSTM":
            #use pytorch LSTM model
            self.decoder = LSTMCell(input_size=embedding_size,hidden_size=hidden_size)
        else:
            #use pytorch RNN model
            self.decoder = RNNCell(input_size=embedding_size,hidden_size=hidden_size)

    #a function for initializing hidden part
    def hidden_unit(self, batches):
        if self.model_type == 'LSTM':
            return (torch.zeros(batches, self.hidden_size).cuda(), torch.zeros(batches, self.hidden_size).cuda())
        else:
            return torch.zeros(batches, self.hidden_size).cuda()
    
    #teacher forcing learning to append captions to features
    def forward(self, features, captions):

        #concatenate embeddings with the feature vectors
        embedded = self.embedding(captions)
        embedded = torch.cat((features.unsqueeze(dim=1).float(), embedded.float()), dim=1)

        #determine if the model is LSTM or RNN
        if self.model_type == 'LSTM':
            hidden_state, cell_state = self.hidden_unit(embedded.shape[0])
        else:
            hidden_state = self.hidden_unit(embedded.shape[0])
        #initialize and send to gpu
        outs = torch.empty(embedded.shape[0], embedded.shape[1], self.hidden_size).cuda()

        #in each time step, use decoder
        for t in range(embedded.shape[1]):
            t_step = embedded[:, t, :]
            #determine if the model is LSTM or RNN
            if self.model_type == 'LSTM':
                hidden_state, cell_state = self.decoder(t_step, (hidden_state, cell_state))
            else:
                hidden_state = self.decoder(t_step, hidden_state)
            outs[:, t, :] = hidden_state

        outs = self.linear(outs)
        # remove last prediction and reorder the predicted result
        predicted = outs[:, :-1, :]	
        predicted = predicted.permute(0, 2, 1)

        return predicted.cuda()

    def generate(self, features):
        generated = []
        cnts = 0
        #generate the hidden part
        if self.model_type == 'LSTM':
            hidden_state, cell_state = self.hidden_unit(1)
        else:
            hidden_state = self.hidden_unit(1)

        while True:
            # decode with the decoder
            if self.model_type == 'LSTM':
                hidden_state, cell_state = self.decoder(features, (hidden_state, cell_state))
            else:
                hidden_state = self.decoder(features, hidden_state)

            #add a linear layer
            out = self.linear(hidden_state)

            #determine whether the caption generatiing process is deterministic or not
            if self.deterministic:
                out = self.softmax(out)
                word = torch.argmax(out, dim=1)
            else:
                #incorporate temperature
                out = out / self.temperature
                #sample from the softmax
                word = torch.Tensor(list(WeightedRandomSampler(self.soft(out), 1))).long().cuda()
                word = word.squeeze(0)
            
            cnts += 1
            generated.append(int(word))

            # if the word is at its max length or end token
            if cnts == self.max_length or int(word) == 2:
                break

            #use word embeddings as the features
            features = self.embedding(word)

        return torch.tensor(generated)



#The architecture 2 decoder structure with LSTM cell
class Decoder_arch2(nn.Module):
    def __init__(model_type, vocab, max_length, temperature, deterministic, batch_size, hidden_size, embedding_size):

        super(Decoder_arch2, self).__init__()

        self.vocab = vocab
        self.max_length = max_length
        self.temperature = temperature
        self.deterministic = deterministic
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # the length of the word embedding dictionary
        self.dict_len = len(vocab)
       
        #construct the decoder architecture 2
        self.embedding = nn.Embedding(self.dict_len, embedding_size)
        self.soft = nn.Softmax()
        self.linear = nn.Linear(hidden_size, self.dict_len)
        self.decoder = nn.LSTMCell(input_size= embedding_size,hidden_size=hidden_size*2)

    
    def get_word(self, word):
        #get the index of the word from the dictionary
        w_idx = self.vocab.word2idx[word]
        embedded = self.embedding(torch.tensor(w_idx).cuda())
        return embedded.unsqueeze(0)


    #teacher force learning to append the captions to features
    def forward(self, features, captions):
        # embed and change dimensions
        embedded = self.embedding(captions)

        #embed for each image and add paddings
        padded = self.get_word("<pad>")
        padded = padded.repeat(embedded.shape[0], 1, 1) 
        embedded = torch.cat((padded, embedded), dim = 1) 

        #get the features
        feat = features.unsqueeze(1).repeat(1, embedded.shape[1], 1)
        concated = torch.cat((embedded, feat), dim = 2)

        # initialize hidden states
        hidden_state, cell_state = None, None
        outs = torch.empty(concated.shape[0], concated.shape[1], self.hidden_size).cuda()

        #for each timestep, add the decoder
        for t in range(concated.shape[1]):
            step = concated[:, t, :] 
            # perform per caption, but 64 images at a time
            if (type(hidden_state) == type(None)) and (type(cell_state) == type(None)):
                hidden_state, cell_state = self.decoder(step)
            else:
                hidden_state, cell_state = self.decoder(step,(hidden_state, cell_state))
            outs[:, t, :] = hidden_state

        outs = self.linear(outs)
        # remove last prediction and reorder the predicted result
        predicted = outs[:, :-1, :]	
        predicted = predicted.permute(0, 2, 1)

        return predicted.cuda()

    def token_to_string_2(self, tokens):
        return self.vocab.token_to_string(tokens)

    def generate(self, features):
        generated = []
        cnts = 0
        # add paddings for the models to predict <start> as the first
        padded = self.get_word("<pad>")
        concated = torch.cat((padded, features), dim = 1)
        hidden_state, cell_state = None, None

        while True:
            #determine if hidden state exist
            if (type(hidden_state) == type(None)) and (type(cell_state) == type(None)):
                hidden_state, cell_state = self.decoder(concated)
            else:
                hidden_state, cell_state = self.decoder(concated,(hidden_state, cell_state))
            out = self.linear(hidden_state)

            #determine whether the caption generatiing process is deterministic or not
            if self.deterministic:
                out = self.soft(out)
                word = torch.argmax(out, dim=1)
            else:
                out = out / self.temperature
                word = torch.Tensor(
                    list(WeightedRandomSampler(self.soft(out), 1))).long().cuda()
                word = word.squeeze(0)
                
            generated.append(int(word))
            cnts += 1

            # reached max length or end token
            if cnts == self.max_length or int(word) == 2:
                break
            concated = self.embedding(word)
            concated = torch.cat((concated, features), dim = 1)

        return torch.tensor(generated)
