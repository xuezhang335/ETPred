import pandas as pd
import random
import math
from Bio import SeqIO


def subtype_selection(subtype):
    global subtype_flag, data_path
    if subtype == 'H1N1':
        subtype_flag = 0
        data_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/raw/H1N1/'
    elif subtype == 'H3N2':
        subtype_flag = 1
        data_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/raw/H3N2/'
    elif subtype == 'H5N1':
        subtype_flag = 2
        data_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/raw/H5N1/'
    elif subtype == 'covid19':
        subtype_flag = 3
        data_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/raw/covid19_raw/'
        
    return subtype_flag, data_path


def read_trigram_vec():
    """
    Reads the csv file containing 100 dimensional prot vecs.
    Returns a dictionary that maps a 3gram of amino acids to its
    index and a numpy array containing the trigram vecs.
    """
    data_file = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/protVec_100d_3grams.csv'
    
    df = pd.read_csv(data_file, delimiter = '\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values
  
    return trigram_to_idx, trigram_vecs

def read_strains_from(data_files, data_path):
    """
    Reads the raw strains from the data_files located by the data_path.
    Returns a pandas series for each data file, contained in a ordered list.
    """
    #_, data_path = subtype_selection(subtype)
    raw_strains = []
    for file_name in data_files:
        df = pd.read_csv(data_path + file_name)
        #strains = df.iloc[:, 1]
        strains = df['seq']
        raw_strains.append(strains)
    
    return raw_strains

def train_test_split_strains(strains_by_month, test_split, cluster):

    """
    Shuffles the strains in each month and splits them into two disjoint sets,
    of size indicated by the test_split.
    Expects and returns pandas dataframe or series.
    """
    train_strains, test_strains = [], []
    
    if cluster == 'random':
        for strains in strains_by_month:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
            train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
            test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    else:
        #change the starting index for the time-series training samples for multiple experiments
        for strains in strains_by_month:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            train = strains.iloc[:num_of_training_examples].reset_index(drop=True)
            test = strains.iloc[num_of_training_examples:].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    return train_strains, test_strains
