# Import necessary libraries
import numpy as np
import csv
import torch
import pandas as pd
import torch.nn as nn
import os
import heapq
from tqdm import tqdm
import torch.optim as optim
import random
import math
import torch.nn.functional as F
import warnings
import wandb
from torch.nn.utils import clip_grad_norm_
warnings.filterwarnings("ignore")

import argparse

# Import warnings
import warnings

wandb.login(key="") 

parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL_A3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs24m030')
parser.add_argument('-d', '--datapath', help='give data path e.g. /kaggle/input/vocabs/Dataset', type=str, default='/kaggle/input/dakshina-dataset-ass-3/dakshina_dataset_v1.0')
parser.add_argument('-l', '--lang', help='language', type=str, default='hin')
parser.add_argument('-e', '--epochs', help="Number of epochs to train network.", type=int, default=10)
parser.add_argument('-nl', '--num_layers', help="number of layers in encoder & decoder", type=int, default=2)
parser.add_argument('-b', '--batch_size', help="Batch size used to train network.", type=int, default=64)
parser.add_argument('-bw', '--beam_width', help="Beam Width for beam Search", type=int, default=1)
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate for training', type=float, default=0.001)
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "adagrad", "adam", "rmsprop"]', type=str, default = 'adagrad', choices= ["sgd", "rmsprop", "adam", "adagrad"])
parser.add_argument('-dp', '--dropout', help="dropout probablity in Ecoder & Decoder", type=float, default=0.3)
parser.add_argument('-cell', '--cell_type', help="Cell Type of Encoder and Decoder", type=str, default="GRU", choices=["LSTM", "RNN", "GRU"])
parser.add_argument('-hdn_size', '--hidden_size', help="Hidden Size", type=int, default=512)
parser.add_argument('-emb_size', '--embadding_size', help="Embadding Size", type=int, default=512)
parser.add_argument('-lp', '--length_penalty', help="Length Panelty", type=float, default=0.6)
parser.add_argument('-bi_dir', '--bidirectional', help="Bidirectional", type=bool, default=True)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help="Teacher Forcing Ratio", type=float, default=0.7)
parser.add_argument('-eval', '--evaluate', help='get test accuarcy and test loss', choices=[0, 1], type=int, default=1)
parser.add_argument('-p', '--console', help='print training_accuracy + loss, validation_accuracy + loss for every epochs', choices=[0, 1], type=int, default=1)
parser.add_argument('-wl', '--wandb_log', help='log on wandb', choices=[0, 1], type=int, default=0)

arguments = parser.parse_args()
def get_hyper_perameter(arguments):
    HYPER_PARAM = {
        'dataset_path' : arguments.datapath,
        'embedding_size': arguments.embadding_size,
        'hidden_size': arguments.hidden_size,
        'num_layers_enc': arguments.num_layers,
        'num_layers_dec': arguments.num_layers,
        'cell_type': arguments.cell_type,
        'dropout': arguments.dropout,
        'optimizer' : arguments.optimizer,
        'learning_rate': arguments.learning_rate,
        'batch_size': arguments.batch_size,
        'num_epochs':arguments.epochs,
        'teacher_fr' : arguments.teacher_forcing_ratio,
        'length_penalty' : arguments.length_penalty,
        'beam_width': arguments.beam_width,
        'bi_dir' : arguments.bidirectional,
        'w_log' : arguments.wandb_log
    }
    
    return HYPER_PARAM



# Check if CUDA is available
val=torch.cuda.is_available()
if val == 1:
    
 # If CUDA is available, use it as the device    
    device= torch.device('cuda')
else:
    
# If GPU is also unavailable, default to CPU
    device = torch.device('gpu')
    


# This function loads and preprocesses the dataset for training a sequence-to-sequence model.
def loadData(params):
    
    # Define path to dataset based on configuration
    data_path = params['dataset_path']
    
    # Open data files for training, validation, and testing
    tr_data = csv.reader(open(data_path + '/hi/lexicons/hi.translit.sampled.train.tsv', encoding='utf8'), delimiter='\t')
    vl_data = csv.reader(open(data_path + '/hi/lexicons/hi.translit.sampled.dev.tsv', encoding='utf8'), delimiter='\t')
    tt_data = csv.reader(open(data_path + '/hi/lexicons/hi.translit.sampled.test.tsv', encoding='utf8'), delimiter='\t')
    
    # Initialize empty lists to store data
    tr_translations = []
    tt_words=[]
    vl_translations = []
    vl_words=[]
    tr_words =[]
    tt_translations = []
    pad=''
    start='$'
    end ='&'
    
    # Load training data
    train_data_list = list(tr_data)
    train_len = len(train_data_list)
    i = 0
    while i < train_len:
        pair = train_data_list[i]
        tr_words.append(pair[0] + end)
        tr_translations.append(start + pair[1] + end)
        i += 1  
    
    # Load validation data 
    i=0
    val_data_list = list(vl_data)
    val_len = len(val_data_list)
    while i < val_len :
        pair=val_data_list[i]
        vl_words.append(pair[0]+end)
        vl_translations.append(start+pair[1]+end)
        i+=1
    
    # Load validation data   
    i=0
    test_data_list = list(tt_data)
    test_len = len(test_data_list)
    while i < test_len :
        pair=test_data_list[i]
        tt_words.append(pair[0]+end)
        tt_translations.append(start+pair[1]+end)
        i+=1   
        
 
    # Convert lists to NumPy arrays for better performance
    tt_words =np.array(tt_words)
    tr_translations = np.array(tr_translations)
    vl_translations =np.array(vl_translations)
    tr_words = np.array(tr_words)
    tt_translations = np.array(tt_translations)
    vl_words =np.array(vl_words)
    
    
    # Build input and output vocabularies
    output_vocab,input_vocab = set() , set()
    
    # Add characters from train_words to input_vocab
    i = 0
    word_len=len(tr_words)
    while i < word_len :
        word = tr_words[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            input_vocab.add(character)
            char_index += 1
        i += 1         
    
    # Add characters from val_words to input_vocab
    i = 0
    word_len=len(vl_words)
    while i < word_len :
        word = vl_words[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            input_vocab.add(character)
            char_index += 1
        i += 1
        
    # Add characters from test_words to input_vocab
    i = 0
    word_len=len(tt_words)
    while i < word_len :
        word = tt_words[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            input_vocab.add(character)
            char_index += 1
        i += 1
    
    # Add characters from train_translations, val_translations, and test_translations to output_vocab
    i = 0
    word_len=len(tr_translations)
    while i < word_len :
        word = tr_translations[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            output_vocab.add(character)
            char_index += 1
        i += 1
        
    i = 0
    word_len=len(vl_translations)
    while i < word_len :
        word = vl_translations[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            output_vocab.add(character)
            char_index += 1
        i += 1
        
    i = 0
    word_len=len(tt_translations)
    while i < word_len :
        word = tt_translations[i]
        char_index = 0
        while char_index < len(word):
            character = word[char_index]
            output_vocab.add(character)
            char_index += 1
        i += 1
    # Remove special tokens from output_vocab
    output_vocab.remove(start)
    input_vocab.remove(end)
    output_vocab.remove(end)
    
    # Sort vocabularies and add special tokens
    output_vocab= [pad, start, end] + list(sorted(output_vocab))
    input_vocab = [pad, start, end] + list(sorted(input_vocab))
            
    # Create index mappings for vocabularies
    output_index,input_index = {char: idx for idx, char in enumerate(output_vocab)},{char: idx for idx, char in enumerate(input_vocab)}
    output_index_rev,input_index_rev = {idx: char for char, idx in output_index.items()},{idx: char for char, idx in input_index.items()}
    
    # Determine maximum sequence length
    max_len = max(max([len(word) for word in np.hstack((tr_words, tt_words, vl_words))]), max([len(word) for word in np.hstack((tr_translations, vl_translations, tt_translations))]))
    
    # Prepare preprocessed data dictionary
    preprocessed_data = {
        'SOS' : start,
        'EOS' : end,
        'PAD' : pad,
        'train_words' : tr_words,
        'train_translations' : tr_translations,
        'val_words' : vl_words,
        'val_translations' : vl_translations,
        'test_words' : tt_words,
        'test_translations' : tt_translations,
        'max_enc_len' : max([len(word) for word in np.hstack((tr_words, tt_words, vl_words))]),
        'max_dec_len' : max([len(word) for word in np.hstack((tr_translations, vl_translations, tt_translations))]),
        'max_len' : max_len,
        'input_index' : input_index,
        'output_index' : output_index,
        'input_index_rev' : input_index_rev,
        'output_index_rev' : output_index_rev
    }
    return preprocessed_data




def create_tensor(preprocessed_data):
    
    # Extract max sequence length and the number of training examples
    prop_data=preprocessed_data['max_len']
    leng=len(preprocessed_data['train_words'])
    d_type='int64'
    
    # Initialize arrays for data
    input_data = np.zeros((prop_data,leng), dtype = d_type)
    output_data = np.zeros((prop_data,leng), dtype = d_type)
    leng=len(preprocessed_data['val_words'])
    vl_input_data = np.zeros((prop_data,leng), dtype = d_type)
    vl_output_data = np.zeros((prop_data,leng), dtype = d_type)
    leng=len(preprocessed_data['test_words'])
    tt_input_data = np.zeros((prop_data,leng), dtype = d_type)
    tt_output_data = np.zeros((prop_data,leng), dtype = d_type)
    
    # Fill in training data arrays
    idx = 0
    while idx < len(preprocessed_data['train_words']):
        w = preprocessed_data['train_words'][idx]
        t = preprocessed_data['train_translations'][idx]

        i = 0
        while i < len(w):
            char = w[i]
            input_data[i, idx] = preprocessed_data['input_index'][char]
            i += 1

        i = 0
        while i < len(t):
            char = t[i]
            output_data[i, idx] = preprocessed_data['output_index'][char]
            i += 1
        idx += 1            
        

    # Fill in validation data arrays        
    idx = 0
    while idx < len(preprocessed_data['val_words']):
        w = preprocessed_data['val_words'][idx]
        t = preprocessed_data['val_translations'][idx]

        i = 0
        while i < len(w):
            char = w[i]
            vl_input_data[i, idx] = preprocessed_data['input_index'][char]
            i += 1

        i = 0
        while i < len(t):
            char = t[i]
            vl_output_data[i, idx] = preprocessed_data['output_index'][char]
            i += 1
        idx += 1            
        
    # Fill in test data arrays        
    idx = 0
    while idx < len(preprocessed_data['test_words']):
        w = preprocessed_data['test_words'][idx]
        t = preprocessed_data['test_translations'][idx]

        i = 0
        while i < len(w):
            char = w[i]
            tt_input_data[i, idx] = preprocessed_data['input_index'][char]
            i += 1

        i = 0
        while i < len(t):
            char = t[i]
            tt_output_data[i, idx] = preprocessed_data['output_index'][char]
            i += 1
        idx += 1            
        
            
    # Convert NumPy arrays to PyTorch tensors        
    output_data=torch.tensor(output_data, dtype = torch.int64)
    input_data = torch.tensor(input_data,dtype = torch.int64)
    vl_output_data=torch.tensor(vl_output_data, dtype = torch.int64)
    vl_input_data = torch.tensor(vl_input_data,dtype = torch.int64)
    tt_output_data=torch.tensor(tt_output_data, dtype = torch.int64)
    tt_input_data= torch.tensor(tt_input_data,dtype = torch.int64)
    
    #Store tensors in a dictionary
    tensors = {
        'input_data' : input_data,
        'output_data' : output_data,
        'val_input_data' : vl_input_data,
        'val_output_data' : vl_output_data, 
        'test_input_data' : tt_input_data,
        'test_output_data' : tt_output_data
    }
    return tensors



# Encoder module for a sequence-to-sequence model.
class Encoder(nn.Module): 
    
    #Initializes the Encoder module.
    def __init__(self, params, preprocessed_data):
        super(Encoder, self).__init__()
        
        # Extract parameters
        self.cell_type = params['cell_type']
        self.dropout = nn.Dropout(params['dropout'])
        
        # Embedding layer
        self.embedding = nn.Embedding(len(preprocessed_data['input_index']), params['embedding_size'])
        
        # RNN or GRU cell based on cell_type
        if self.cell_type == 'RNN':
            self.cell = nn.RNN(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout = params['dropout'], bidirectional = params['bi_dir'])
        elif self.cell_type == 'GRU':
            self.cell = nn.GRU(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout = params['dropout'], bidirectional = params['bi_dir'])
    #Forward pass of the Encoder    
    def forward(self, x):
        
        # Embedding layer
        drop_par = self.embedding(x)
        # Pass through RNN/GRU cell
        _ , hidden = self.cell(self.dropout(drop_par))
        
        # Return hidden state
        return hidden



#Decoder module for a sequence-to-sequence model.
class Decoder(nn.Module):
    
    #Initializes the Decoder module
    def __init__(self, params, preprocessed_data):
        super(Decoder, self).__init__()
        
        # Extract parameters
        self.cell_type = params['cell_type']
        self.dropout = nn.Dropout(params['dropout'])
        self.embedding = nn.Embedding(len(preprocessed_data['output_index']), params['embedding_size'])
        
        # RNN or GRU cell based on cell_type
        if self.cell_type == 'RNN':
            self.cell = nn.RNN(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout = params['dropout'], bidirectional = params['bi_dir'])
        elif self.cell_type == 'GRU':
            self.cell = nn.GRU(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout = params['dropout'], bidirectional = params['bi_dir'])
        
        # Fully connected layer for output prediction
        self.fc = nn.Linear(params['hidden_size'] * 2 if params['bi_dir'] == True else params['hidden_size'], len(preprocessed_data['output_index']))
    
    #Forward pass of the Decoder.
    def forward(self, x, hidden, cell):
        
        # Embedding layer
        emb = self.embedding(x.unsqueeze(0))
        outputs, hidden = self.cell(self.dropout(emb), hidden)
        
        # Predictions with fully connected layer
        pred = self.fc(outputs).squeeze(0)
        
        # Return predictions and updated hidden state
        return pred, hidden



#Sequence-to-sequence model consisting of an Encoder and a Decoder.
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, params,  preprocessed_data):
        #Initializes the Seq2Seq model
        super(Seq2Seq, self).__init__()
        
        # Extract parameters
        self.cell_type = params['cell_type']
        self.decoder, self.encoder  = decoder, encoder
        self.output_index_len = len(preprocessed_data['output_index'])
        self.tfr = params['teacher_fr']
    
    #Forward pass of the Seq2Seq model
    def forward(self, source, target):
        
        # Extract batch size and target sequence length
        bs, target_len = source.shape[1], target.shape[0]
        x = target[0]
        outputs = torch.zeros(target_len, bs, self.output_index_len).to(device)
        
        # Encode the source sequence to obtain the initial hidden state
        hidden = self.encoder(source)
        
        # Iterate over each step in the target sequence
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden, None)
            
            # Store the decoder output in the outputs tensor
            outputs[t], best_guess = output, output.argmax(1)
            
            # Determine the next input (x) for the decoder
            x = best_guess if random.random() >= self.tfr else target[t]
            
        # Return the predicted outputs from the decoder    
        return outputs



# Encoder module using LSTM (Long Short-Term Memory) cells.
class Encoder_LSTM(nn.Module): 
    #Initializes the Encoder_LSTM module.
    def __init__(self, params, preprocessed_data):
        super(Encoder_LSTM, self).__init__()
        # Initialize dropout layer
        self.dropout = nn.Dropout(params['dropout'])
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(len(preprocessed_data['input_index']), params['embedding_size'])
        
        # Initialize LSTM cell
        self.cell = nn.LSTM(params['embedding_size'], params['hidden_size'], params['num_layers_enc'], dropout = params['dropout'], bidirectional = params['bi_dir'])
    
    #Forward pass of the Encoder_LSTM    
    def forward(self, x):
        # Embedding layer
        drop_par = self.embedding(x)
        # applying dropout to the embedding layer
        outputs , (hidden, cell) = self.cell(self.dropout(drop_par))
        
        # Return hidden and cell states
        return hidden, cell




#Decoder module using LSTM (Long Short-Term Memory) cells.
class Decoder_LSTM(nn.Module):
    
    #Initializes the Decoder_LSTM module
    def __init__(self, params, preprocessed_data):
        super(Decoder_LSTM, self).__init__()
        
        # Dropout layer to randomly zero out input elements to prevent overfitting
        self.dropout = nn.Dropout(params['dropout'])
        
        # Embedding layer to convert output tokens into dense vectors
        self.embedding = nn.Embedding(len(preprocessed_data['output_index']), params['embedding_size'])
        
        # LSTM cell for sequence decoding
        self.cell =  nn.LSTM(params['embedding_size'], params['hidden_size'], params['num_layers_dec'], dropout = params['dropout'], bidirectional = params['bi_dir'])
        self.fc = nn.Linear(params['hidden_size'] *  2 if params['bi_dir'] == True else params['hidden_size'], len(preprocessed_data['output_index']))
    
    #Forward pass of the Decoder_LSTM.
    def forward(self, x, hidden, cell):
        
        # Embedding layer: maps input token to dense vector
        emb = self.embedding(x.unsqueeze(0))
        
        # Pass the embedded and dropout-processed input through the LSTM cell
        outputs , (hidden, cell) = self.cell(self.dropout(emb), (hidden, cell))
        
        # Predictions with fully connected layer
        pred  = self.fc(outputs).squeeze(0)
        
        # Apply log softmax activation to obtain output probabilities
        pred = F.log_softmax(pred, dim = 1)
        
        # Return predicted output probabilities, updated hidden, and cell states
        return pred, hidden, cell




#Sequence-to-sequence model using LSTM cells for both encoding and decoding.
class Seq2Seq_LSTM(nn.Module):
    def __init__(self, encoder, decoder, params,  preprocessed_data):
        super(Seq2Seq_LSTM, self).__init__()
        
        # Store references to encoder, decoder, and other attributes
        self.cell_type = params['cell_type']
        self.decoder, self.encoder  = decoder, encoder
        self.output_index_len = len(preprocessed_data['output_index'])
        self.tfr = params['teacher_fr']
    
    #Forward pass of the Seq2Seq_LSTM model.
    def forward(self, source, target):
        
        # Extract batch size and target sequence length
        batch_size, target_len = source.shape[1], target.shape[0]
        
        # Initial input to the decoder (start token)
        x = target[0]
        outputs = torch.zeros(target_len, batch_size, self.output_index_len).to(device)
        
        # Encode the source sequence to obtain initial hidden and cell state
        hidden, cell = self.encoder(source)
        
        # Iterate over each step in the target sequence
        for t in range(1, target_len):
            
            # Pass input (x), hidden, and cell states to the decoder
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            # Store the decoder output in the outputs tensor
            outputs[t], best_guess = output, output.argmax(1)
            
            # Determine the next input (x) for the decoder using teacher forcing strategy
            x = best_guess if random.random() >= self.tfr else target[t]
        
        # Return the predicted outputs from the decoder for each time step
        return outputs




# Function to get the optimizer based on specified parameters
def get_optim(model, params):
    # Extract the optimizer type from params and convert to lowercase
    val = params['optimizer'].lower()
    
    if  val== 'sgd':
        # Use Stochastic Gradient Descent (SGD) optimizer
        opt = optim.SGD(model.parameters(), lr = params['learning_rate'], momentum = 0.9)
    
    if val == 'adagrad':
        # Use adagrad optimizer
        opt = optim.Adagrad(model.parameters(), lr = params['learning_rate'], lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10)
    
    if val == 'adam':
        # Use adam optimizer
        opt = optim.Adam(model.parameters(), lr = params['learning_rate'], betas = (0.9, 0.999), eps = 1e-8)
    
    if val == 'rmsprop':
        # Use rmsprop optimizer
        opt = optim.RMSprop(model.parameters(), lr = params['learning_rate'], alpha = 0.99, eps = 1e-8)
    
    return opt




#Beam search function to generate predictions using a sequence-to-sequence model
def beam_search(model, word, preprocessed_data, bw, lp, ct):
    
    # Prepare input data tensor for the model
    val=preprocessed_data['max_len']+1
    data = np.zeros((val, 1), dtype=np.int32)
    
    # Map input character to index
    idx = 0
    while idx < len(word):
        char = word[ idx ]
        data[ idx , 0 ] = preprocessed_data['input_index'][char]
        idx += 1
    
    # Append EOS token index to indicate end of input sequence
    data[idx , 0] = preprocessed_data['input_index'][preprocessed_data['EOS']]
    
    # Convert input data to torch tensor and move to appropriate device
    data = torch.tensor(data, dtype=torch.int32).to(device)
    
    # Encode the input sequence to obtain initial hidden state
    with torch.no_grad():
        # For RNN encoder, obtain only hidden state
        val = ct !='LSTM' 
        if  val :
            hidden = model.encoder(data)
        else:
           # For LSTM encoder, obtain both hidden and cell states 
           hidden, cell = model.encoder(data)
    hidden_par = hidden.unsqueeze(0)
    
    # Reshape the SOS token index for initializing the sequence
    out_reshape = np.array(preprocessed_data['output_index'][preprocessed_data['SOS']]).reshape(1,)
    initial_seq = torch.tensor(out_reshape).to(device)
    
    # Initialize beam with the initial sequence and its score
    beam = [(0.0, initial_seq, hidden_par)]
    
    # Beam search loop to generate sequences
    i = 0
    leng=len(preprocessed_data['output_index'])
    while i < leng:
        candidates = []
        index = 0
        while index < len(beam):
            score, seq, hidden = beam[index]
            
            # Check if sequence ends with EOS token
            val=seq[-1].item() == preprocessed_data['output_index'][preprocessed_data['EOS']]
            if val:
                candidates.append((score, seq, hidden))
                index+=1
                continue
            
            # Prepare input token for the decoder based on the last token of the sequence
            reshape_last = np.array(seq[-1].item()).reshape(1,)
            hdn = hidden.squeeze(0)
            x = torch.tensor(reshape_last).to(device)
            
            # Decode the input token to get output probabilities and updated hidden state
            val= ct == 'LSTM'
            if val!=1:
                output ,  hidden = model.decoder(x, hdn, None)
            else:
                output, hidden , cell = model.decoder(x, hdn, cell)
            val=F.softmax(output, dim=1)
            
            # Apply softmax to obtain probabilities over output tokens
            topk_probs , topk_tokens = torch.topk(val, k=bw)
            
            # Generate candidate sequences based on top-k tokens
            ii = 0
            while ii < len(topk_probs[0]):
                prob = topk_probs[0][ii]
                token = topk_tokens[0][ii]
                new_seq = torch.cat((seq, token.unsqueeze(0)), dim=0)
                
                # Calculate length normalization factor (penalty) for the new sequence
                ln_ns = len(new_seq)
                ln_pf = ((ln_ns - 1) / 5)
                candidate_score = score + torch.log(prob).item() / (ln_pf ** lp)
                
                # Append candidate (score, sequence, hidden state) to candidates list
                candidates.append((candidate_score, new_seq, hidden.unsqueeze(0)))
                ii += 1
            index += 1
        # Select top beam width candidates based on scores    
        beam = heapq.nlargest(bw, candidates, key=lambda x: x[0])
        i += 1
    m = max(beam, key=lambda x: x[0]) 
    _, best_sequence, _ = m
    
    # Convert predicted sequence tokens to characters and concatenate them
    pred = ''.join([preprocessed_data['output_index_rev'][token.item()] for token in best_sequence[1:]])
    
    # Return the predicted sequence (excluding the EOS token)
    return pred[:-1]



# Function to train the model
def train(model, crit, optimizer, preprocessed_data, tensors, params):
    val=1
    bs='batch_size'
    # Split the input and output data into batches
    tr_result = torch.split(tensors['output_data'], params[bs], dim = val)
    tr_data=torch.split(tensors['input_data'], params[bs], dim = val)
    vl_result= torch.split(tensors['val_output_data'], params[bs], dim=val)
    vl_data=torch.split(tensors['val_input_data'], params[bs], dim=val)
    
    # Loop through epochs
    epoch = 0
    while epoch < params['num_epochs'] :
        epoch +=1
        
        # Initialize counters for metrics
        correct_prediction,total_loss,total_words = 0,0,0
        model.train()
        leng=len(tr_data)
        
        # Use tqdm for progress visualization during training
        val='Training'
        with tqdm(total = leng, desc = val) as pbar:
            index = 0
            lenn = len(tr_data)
            
            # Loop through each batch in training data
            while index < lenn:
                # Move input and target data to device (e.g., GPU)
                y = tr_result[index]
                x = tr_data[index] 
                inp_data = x.to(device)
                target= y.to(device) 
                optimizer.zero_grad()
                output = model(inp_data, target)
                
                # Reshape target and output for loss calculation
                target = target.reshape(-1)
                output = output.reshape(-1, output.shape[2])
                
                # Create a mask to ignore padding tokens
                pad_mask = (target != preprocessed_data['output_index'][preprocessed_data['PAD']])
                output = output[pad_mask]
                target = target[pad_mask]
                
                # Compute loss and perform backpropagation
                loss = crit(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                
                # Update metrics
                total_loss = total_loss +loss.item()
                total_words = total_words + target.size(0)
                correct_prediction = correct_prediction + torch.sum(torch.argmax(output, dim=1) == target).item()

                index += 1
                pbar.update(1)
        
        # Calculate training accuracy and loss
        cal=correct_prediction / total_words
        train_accuracy = cal * 100
        len_train=len(tr_data)
        train_loss = total_loss / len_train
        model.eval()
        
        # Evaluate model on validation data
        with torch.no_grad():
            val_total_words,val_correct_pred,val_total_loss = 0,0,0
            with tqdm(total = len(vl_data), desc = 'Validation') as pbar:
                
                index = 0
                lenn=len(vl_data)
                # Loop through each batch in validation data
                while index < lenn:
                    
                    y_val =vl_result[index]
                    x_val= vl_data[index]
                    # Move validation input and target data to device
                    inp_data_val = x_val.to(device)
                    target_val=y_val.to(device)
                    
                    # Forward pass through the model for validation
                    output_val = model(inp_data_val, target_val)
                    target_val = target_val.reshape(-1)
                    output_val = output_val.reshape(-1, output_val.shape[2])
                    
                    # Create mask to ignore padding tokens
                    pad_mask = (target_val != preprocessed_data['output_index'][preprocessed_data['PAD']])
                    output_val = output_val[pad_mask]
                    target_val = target_val[pad_mask]
                    
                    # Calculate validation loss and metrics
                    val_loss = crit(output_val, target_val)
                    val_total_loss = val_total_loss+ val_loss.item()
                    val_total_words = val_total_words+ target_val.size(0)
                    val_correct_pred = val_correct_pred+ torch.sum(torch.argmax(output_val, dim=1) == target_val).item()
                    index += 1
                    pbar.update(1)
            # Calculate validation accuracy and loss        
            cal=val_correct_pred / val_total_words        
            val_accuracy = cal * 100
            lengg=len(vl_data)
            val_loss = val_total_loss / lengg
            
            # Evaluate model using beam search and calculate word-level accuracy
            correct_prediction = 0
            total_words = len(preprocessed_data['val_words'])
            with tqdm(total = total_words, desc = 'Beam') as pbar_:
                index = 0
                # Loop through each word in validation set for beam search evaluation
                while index < len(preprocessed_data['val_words']):
                    word, translation = preprocessed_data['val_words'][index], preprocessed_data['val_translations'][index]
                    ans = beam_search(model, word, preprocessed_data, params['beam_width'], params['length_penalty'], params['cell_type'])
                    val= translation[1:-1]
                    # Check if beam search translation matches reference translation
                    if ans == val:
                        correct_prediction = correct_prediction +1

                    index += 1
                    pbar_.update(1)
        # Calculate word-level accuracy using beam search            
        cal=correct_prediction / total_words
        val_accuracy_beam = cal * 100
        
        # Print and log results
        print(f'''Epoch : {epoch}
              Train Accuracy Char Level : {train_accuracy:.4f}, Train Loss : {train_loss:.4f}
              Validation Accuracy Char Level : {val_accuracy:.4f}, Validation Loss : {val_loss:.4f}
              Validation Accuracy Word Level : {val_accuracy_beam:.4f},  Correctly predicted : {correct_prediction}/{total_words}''')
        if params['w_log']:
            wandb.log(
                    {
                        'epoch': epoch,
                        'training_loss' : train_loss,
                        'training_accuracy_char' : train_accuracy,
                        'validation_loss' : val_loss,
                        'validation_accuracy_char' : val_accuracy,
                        'validation_accuracy_word' : val_accuracy_beam,
                        'correctly_predicted' : correct_prediction
                    }
                )
    
    # Return the trained model and validation accuracies
    return model, val_accuracy, val_accuracy_beam



#providing the parameters for the function
params = get_hyper_perameter(arguments)




# Load preprocessed data based on specified parameters
preprocessed_data = loadData(params)

# Create tensors from the preprocessed data
tensors = create_tensor(preprocessed_data)

# Initialize the model based on the cell type specified in parameters
if params['cell_type'] == 'LSTM':
    # Use LSTM-based encoder, decoder, and Seq2Seq model
    encoder = Encoder_LSTM(params, preprocessed_data).to(device)
    decoder = Decoder_LSTM(params, preprocessed_data).to(device)
    model = Seq2Seq_LSTM(encoder, decoder, params, preprocessed_data).to(device) 
else:
    #Use RNN-based encoder, decoder, and Seq2Seq model 
    encoder = Encoder(params, preprocessed_data).to(device)
    decoder = Decoder(params, preprocessed_data).to(device)
    model = Seq2Seq(encoder, decoder, params, preprocessed_data).to(device)  
# print(model)

# Define the criterion (loss function) for training
crit = nn.CrossEntropyLoss(ignore_index = 0)

# Get the optimizer based on specified parameters
opt = get_optim(model,params)

# Initialize Weights & Biases (wandb) if logging is enabled
if params['w_log']:
    # Set the name of the run based on the model and training parameters
    wandb.init(project = 'DL_A3')
    wandb.run.name = f"c:{params['cell_type']}_e:{params['num_epochs']}_es:{params['embedding_size']}_hs:{params['hidden_size']}_nle:{params['num_layers_enc']}_nld:{params['num_layers_dec']}_o:{params['optimizer']}_lr:{params['learning_rate']}_bs:{params['batch_size']}_tf:{params['teacher_fr']}_lp:{params['length_penalty']}_b:{params['bi_dir']}_bw:{params['beam_width']}"


def evaluate(preprocessed_data,trained_model):

  # Set the trained model to evaluation mode
  trained_model.eval()

  # Initialize variables for tracking predictions and evaluation
  correct_prediction = 0
  words = []
  translations = []
  prediction = []
  results = []

  # Use tqdm to visualize progress during inference
  total_words = len(preprocessed_data['test_words'])
  with tqdm(total = total_words) as pbar_:
      
      # Loop through each word in the test set
      index = 0
      while index < len(preprocessed_data['test_words']):
          word, translation = preprocessed_data['test_words'][index], preprocessed_data['test_translations'][index]
          
          # Perform beam search to generate a translation using the trained model
          ans = beam_search(trained_model, word, preprocessed_data, params['beam_width'], params['length_penalty'], params['cell_type'])
          
          # Store the word (without end token), translation (without start/end tokens), and predicted translation
          words.append(word[:-1])
          translations.append(translation[1:-1])
          prediction.append(ans)
          
          # Check if the predicted translation matches the reference translation
          val= ans == translation[1:-1]
          if val!=1 :
              results.append('No')
          else:
              correct_prediction = correct_prediction + 1
              results.append('Yes')
          index += 1
          pbar_.update(1)

  # Calculate accuracy based on correct predictions  
  cal=correct_prediction / total_words    
  accuracy = cal * 100
  print(f'Test Accuracy Word Level : {accuracy}, Correctly Predicted : {correct_prediction}')

trained_model,valaccuracy,valaccuracy_beam =  train(model, crit, opt, preprocessed_data, tensors, params)

if(arguments.evaluate == 1):
    evaluate(preprocessed_data,trained_model)

# Finish Weights & Biases logging if enabled
if params['w_log']:
    wandb.finish()

  



