################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

from cmath import inf
import matplotlib.pyplot as plt
import numpy as np
import nltk
import copy
from PIL import Image

import torch
from torch import LongTensor
from torch.nn import Embedding, LSTMCell, RNNCell
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model



# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        #compare loss across epochs
        self.__best_loss = inf
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.0005)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        #record batches
        batches = 0

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):

            batches = batches+1
            
            self.__optimizer.zero_grad()
            #to cuda and calculate loss
            out = self.__model(images.to("cuda:0"), captions.to("cuda:0"))
            losses = self.__criterion(out, captions.to("cuda:0"))
            training_loss += losses.item()
            losses.backward()
            self.__optimizer.step()

            #apply softmax to select tokens
            if i % 10 == 0:
                word = out[0, :, :]
                word = word.permute(1, 0)
                word = torch.argmax(word, dim = 1)
                #check if the tokens look right
                #print("train number",str(i),word)

            #train the image
            if i % 100 == 0:
                self.__model.eval()
                #generate captions
                with torch.no_grad():
                    predicted = self.__model.generate(images[0, :].to("cuda:0"))
                    #check if the tokens look right
                    # predicted_str = self.__vocab.token_to_string(predicted)
                    # print("Caption number",str(i),predicted_str)
                self.__model.train()

        return training_loss/batches

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        #calculate the gradient
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                out = self.__model(images.to("cuda:0"), captions.to("cuda:0"))
                loss = self.__criterion(out, captions.to("cuda:0"))
                val_loss += loss.item()
                if i % 10 == 0:
                    word = out[0, :, :]
                    #permute with probability and softmax
                    word = word.permute(1, 0)
                    word = torch.argmax(word, dim=1)
                    #check if the tokens look right
                    #print("validation number",str(i),elf.__vocab.token_to_string(word))\
        
        #calculate validation loss
        val_loss = val_loss / len(self.__val_loader)

        # store the best model
        if val_loss < self.__best_loss:
            #create a deep copy to preserve the wholeness of the model
            self.__best_model = copy.deepcopy(self.__model.state_dict())
        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bl1 = 0
        bl4 = 0
        batches = 0
        
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                batches += 1 

                #send data and model to gpu
                images = images.cuda()
                outs = self.__model(images, captions.to("cuda:0"))

                #calculate the loss
                loss = self.__criterion(outs, captions.to("cuda:0"))
                test_loss += loss.item()
    
                batch_size = 0
                bleu1_batch = 0
                bleu4_batch = 0

                #generate test captions
                for i, image_id in enumerate(img_ids):    
                    batch_size += 1

                    #retrieve test captions for comparison later
                    test_caps = []
                    for image in self.__coco_test.imgToAnns[image_id]:
                        test_caps.append(nltk.word_tokenize(image['caption'].lower()))
                    
                    #compare the predicted result with the captions in test dataset
                    predicted = self.__model.generate(images[i, :])
                    predicted_str = nltk.word_tokenize(self.__vocab.token_to_string(predicted).lower())

                    # Generate image
                    root_dir="./data/images/"
                    root_test = os.path.join(root_dir, 'test')
                    path = self.__coco_test.loadImgs(image_id)[0]['file_name']
                    image = Image.open(os.path.join(root_test, path)).convert('RGB')
                    image = self.resize(image)
                    image = np.asarray(image)
                    img = Image.fromarray(image, 'RGB')
                    img.show()
                    out=" ".join(predicted_str)
                    #show image
                    print(out)

                    #calculate blue scores
                    bleu1_temp = bleu1(test_caps, predicted_str)
                    bleu4_temp = bleu4(test_caps, predicted_str)
                    bleu1_batch += bleu1_temp
                    bleu4_batch += bleu4_temp

                bleu1_batch = bleu1_batch / batch_size
                bleu4_batch = bleu4_batch / batch_size

                bl1 += bleu1_batch
                bl4 += bleu4_batch

        #log test statistics
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss / batches,
                                                                               bl1 /batches,
                                                                               bl4 / batches)
        self.__log(result_str)
        
        return test_loss / batches, bl1 / batches, bl4 / batches

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
