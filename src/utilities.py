import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import readline
import select
import torch

def word_to_number(string):
    amino_letters = '-GALMFWKQESPVICYHRNDT'
    letter_map = {l : n for l,n in zip(amino_letters, range(21))}
    n_list = []
    for l in string:
        n_list.append(letter_map[l])
    return n_list

def number_to_word(X):
    amino_letters = '-GALMFWKQESPVICYHRNDT'
    seq_list = []
    for l in X:
        seq = []
        for aa in l:
            seq.append(amino_letters[aa])
        seq_list.append(''.join(seq))
    return seq_list

def catch_file(main_repo : str, message : str) -> str:
    """Allows to dynamically navigate among the files with text autocompletion.
    
    Parameters:
        main_repo (str): Repository to start with.
        message (str): Message to print when asking to insert the filename.
        
    Returns:
        str: selected filename.
    """
    current_repo = os.getcwd()
    os.chdir(main_repo)
    
    def completer(text, state):
        dir_path = '/'.join([dir for dir in text.split('/')[:-1]])
        filename = text.split('/')[-1]
        if dir_path != '':
            files = os.listdir('./' + dir_path)
        else:
            files = os.listdir('.' + dir_path)
        matches = [f for f in files if f.startswith(filename)]
        if dir_path != '':
            try:
                return dir_path + '/' + matches[state]
            except IndexError:
                return None
        else:
            try:
                return matches[state]
            except IndexError:
                return None
   
    # Enable name completion
    readline.parse_and_bind('tab: complete')
    # Set the completer function
    readline.set_completer(completer)
    # Set the delimiters
    readline.set_completer_delims(' \t\n')
    filename = input(message)
    
    os.chdir(current_repo)
    
    return main_repo + '/' + filename

def select_device():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        print(f'Found n.{num_devices} cuda devices.')
        if num_devices == 1:
            print('cuda device automatically selected.')
            device = torch.device('cuda')
        else:
            dev_idx = input(f'Select the cuda device {[i for i in range(num_devices)]}: ')
            device = torch.device(f'cuda:' + dev_idx)
    else:
        device = torch.device('cpu')
    
    return device

def check_if_exists(fname):
    if os.path.exists(fname):
        good_input = False
        overwrite = 'y'
        while not good_input:
            print(f'The file {fname} already exists. Do you want to overwrite it? [y/n] ')
            i, _, _ = select.select([sys.stdin], [], [], 10) # 10 seconds to answer
            if i:
                overwrite = sys.stdin.readline().strip()
            if overwrite in ['n', 'N']:
                sys.exit('Execution aborted')
            elif overwrite in ['y', 'Y']:
                os.remove(fname)
                good_input = True
