import math
import ast
import pandas as pd
import numpy as np
from src.data import make_dataset
from src.features import build_features
from src.data import cluster
import random

def process_month(strains_by_month, squeeze=True, extract_epitopes=False):
    if(len(strains_by_month[0]) == 0): return [],[]
    trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vec()
    trigrams_by_month = build_features.split_to_trigrams(strains_by_month)

    if extract_epitopes:
        epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168]
        epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198]
        epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312]
        epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248]
        epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
        epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
        epitope_positions.sort()

        trigrams_by_month = build_features.extract_positions_by_month(epitope_positions, trigrams_by_month)

    if squeeze:
        trigrams_by_month = build_features.squeeze_trigrams(trigrams_by_month)

    #f, t = len_trigram(trigrams_by_month)
    #print('wrong trigram = ' + str(f))
    #print('true trigram = ' + str(t))

    trigram_idxs = build_features.map_trigrams_to_idxs(trigrams_by_month, trigram_to_idx)

    trigram_vecs = build_features.map_idxs_to_vecs(trigram_idxs, trigram_vecs_data)

    return trigram_vecs, trigram_idxs


def read_dataset(path, limit=0, concat=False):
    """
    Reads the data set from given path, expecting it to contain a 'y' column with
    the label and each month in its own column containing a number of trigram indexes.
    Limit sets the maximum number of examples to read, zero meaning no limit.
    If concat is true each of the trigrams in a month is concatenated, if false
    they are instead summed elementwise.
    """
    #subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    _, trigram_vecs_data = make_dataset.read_trigram_vec()

    df = pd.read_csv(path)

    if limit != 0:
        df = df.head(limit)

    labels = df['y'].values
    trigram_idx_strings = df.loc[:, df.columns != 'y'].values
    parsed_trigram_idxs = [list(map(lambda x: ast.literal_eval(x), example)) for example in trigram_idx_strings]
    trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))

    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        # Sum trigram vecs instead of concatenating them
        trigram_vecs = np.sum(trigram_vecs, axis=2)
        trigram_vecs = np.moveaxis(trigram_vecs, 1, 0)

    return trigram_vecs, labels


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '
  
    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string
