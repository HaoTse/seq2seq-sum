'''
This script handling the training process.
'''

import argparse
import random
import math
import time
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

import utils.Constants as Constants
from dataset import SumDataset, paired_collate_fn
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.optim import Optimizer
from seq2seq.loss import Perplexity


def train_epoch(model, training_data, loss, optimizer, device, teacher_forcing_ratio=0):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    cnt = 0
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        input_variables, input_lengths, target_variables, _ = map(lambda x: x.to(device), batch)

        # forward
        # optimizer.zero_grad()
        decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths, target_variables,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)

        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variables.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
        # Backward propagation
        model.zero_grad()

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.get_loss()
        cnt += 1

    return total_loss / cnt

def eval_epoch(model, validation_data, loss, device):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0

    loss.reset()
    match = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            input_variables, input_lengths, target_variables, _ = map(lambda x: x.to(device), batch)

            # forward
            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(Constants.EOS)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                match += correct
                total += non_padding.sum().item()

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

    return loss.get_loss(), accuracy

def train(model, training_data, validation_data, loss, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = os.path.join('log',
                f'{opt.d_model}_{opt.log}.train.log')
        log_valid_file = os.path.join('log',
                f'{opt.d_model}_{opt.log}.valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss\n')
            log_vf.write('epoch,loss\n')

    all_valid_loss = [10000]
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(
            model, training_data, loss, optimizer, device, teacher_forcing_ratio=opt.teacher_forcing_ratio)
        print('  - (Training)   ppl: {ppl: 8.5f}, elapse: {elapse:3.3f} min'.format(
                  ppl=train_loss, elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, loss, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=valid_loss, accu=100*valid_accu,
                    elapse=(time.time()-start)/60))
        all_valid_loss.append(valid_loss)

        state_dict = model.state_dict()
        checkpoint = {
            'model': state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            model_name = f'{opt.d_model}_{opt.save_model}_epoch_{epoch_i}.chkpt'
            if opt.save_mode == 'all':
                torch.save(checkpoint, os.path.join('model', model_name))
            elif opt.save_mode == 'best':
                if valid_loss < max(all_valid_loss):
                    torch.save(checkpoint, os.path.join('model', model_name))
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss))
                log_vf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=valid_loss))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=3)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_model', type=int, default=1024)
    parser.add_argument('-n_layer', type=int, default=1)

    parser.add_argument('-dropout', type=float, default=0)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.log = opt.save_model

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')

    # model
    opt.bidirectional = True
    encoder = EncoderRNN(opt.src_vocab_size, opt.max_token_seq_len, opt.d_model,
                            bidirectional=opt.bidirectional, variable_lengths=True)
    decoder = DecoderRNN(opt.tgt_vocab_size, opt.max_token_seq_len, opt.d_model * 2 if opt.bidirectional else opt.d_model,
                            n_layers=opt.n_layer, dropout_p=opt.dropout, use_attention=True, bidirectional=opt.bidirectional,
                            eos_id=Constants.BOS, sos_id=Constants.EOS)
    seq2seq = Seq2seq(encoder, decoder).to(device)
    for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)
    
    seq2seq = nn.DataParallel(seq2seq)

    # loss
    weight = torch.ones(opt.tgt_vocab_size)
    pad = Constants.PAD
    loss = Perplexity(weight, pad)
    if opt.cuda:
        loss.cuda()

    # optimizer
    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)

    train(seq2seq, training_data, validation_data, loss, optimizer, device ,opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        SumDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        # num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        SumDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        # num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
