''' Summarize input text with trained model. '''

import torch
import torch.nn as nn
import torch.utils.data
import argparse
from tqdm import tqdm

import utils.Constants as Constants
from dataset import SumDataset
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.evaluator import Predictor

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='sum_file.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case,
        preprocess_settings.mode)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    # prepare model
    device = torch.device('cuda' if opt.cuda else 'cpu')
    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['settings']
    
    model_opt.bidirectional = True
    encoder = EncoderRNN(model_opt.src_vocab_size, model_opt.max_token_seq_len, model_opt.d_model,
                            bidirectional=model_opt.bidirectional, variable_lengths=True)
    decoder = DecoderRNN(model_opt.tgt_vocab_size, model_opt.max_token_seq_len, model_opt.d_model * 2 if model_opt.bidirectional else model_opt.d_model,
                            n_layers=model_opt.n_layer, dropout_p=model_opt.dropout, use_attention=True, bidirectional=model_opt.bidirectional,
                            eos_id=Constants.BOS, sos_id=Constants.EOS)
    model = Seq2seq(encoder, decoder).to(device)
    model = nn.DataParallel(model) # using Dataparallel because training used

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')

    predictor = Predictor(model, preprocess_data['dict']['tgt'])

    with open(opt.output, 'w') as f:
        for src_seq in tqdm(test_src_insts, mininterval=2, desc='  - (Test)', leave=False):
            pred_line = ' '.join(predictor.predict(src_seq))
            f.write(pred_line + '\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
