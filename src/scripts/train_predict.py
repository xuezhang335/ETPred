import os, sys
sys.path.append(os.path.abspath("/share/home/biopharm/xuxiaobin/covid19/COVID_etpred"))

from src.models import models, train_model
from src.data import make_dataset
from src.features import build_features
from src.utils import utils
import torch
import numpy as np

# pos for output the prob, 0 means all sites
def main(position = 0):
    #subtype_flag = {0: H1N1, 1: H3N2, 2: H5N1, 3: covid19}
    if subtype_flag == 0:
        if position_type == 'epitopes':
            data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H1N1/triplet_cluster'
        elif position_type == 'single':    
            data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H1N1_single_position/triplet_cluster'
        save_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/model_save/H1N1'
    
    elif subtype_flag == 1:
        data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H3N2/triplet_cluster'  
        save_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/model_save/H3N2'

    elif subtype_flag == 2:
        data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H5N1/triplet_cluster'
        save_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/model_save/H5N1'

    elif subtype_flag == 3:
        if position_type == 'epitopes':
            data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/covid19/triplet_MiniBatchKMeans_2021-1_to_2023-1'
        elif position_type == 'single':
            data_set = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/covid19_single_position/triplet_MiniBatchKMeans_2021-1_to_2023-1'
        save_path = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/model_save/covid19'


    parameters = {
            
      # Exlude _train/_test and file ending
      'data_set': data_set,

      # path to save the model
      'save_path':save_path,
    
      # 'svm', lstm', 'gru', 'attention', 'da-rnn' 
      'model': model,
    
      # Number of hidden units in the encoder
      'hidden_size': 100,
    
      # Droprate (applied at input)
      'dropout_p': 0.5,
    
      # Note, no learning rate decay implemented
      'learning_rate': 0.001,
    
      # Size of mini batch
      'batch_size': 256,
    
      # Number of training iterations
      'num_of_epochs': 50
    }
    
    torch.manual_seed(1)
    np.random.seed(1)

    if position_type == 'epitopes':
        train_trigram_vecs, train_labels = utils.read_dataset(parameters['data_set'] + '_train.csv', concat=False)
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test.csv', concat=False)
    else:
        train_trigram_vecs, train_labels = utils.read_dataset(parameters['data_set'] + '_train_' + str(position) + '.csv', concat=False)
        test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test_' + str(position) + '.csv', concat=False)
    
    X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
    Y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)
    
    #give weights for imbalanced dataset
    _, counts = np.unique(Y_train, return_counts=True)
    train_imbalance = max(counts) / Y_train.shape[0]
    _, counts = np.unique(Y_test, return_counts=True)
    test_imbalance = max(counts) / Y_test.shape[0]    
    

    print('Class imbalances:')
    print(' Training %.3f' % train_imbalance)
    print(' Testing  %.3f' % test_imbalance)
    if position_type == 'epitopes':
        with open(parameters['data_set'] + '_train_baseline.txt', 'r') as f:
            print('Train baselines:')
            print(f.read())
        with open(parameters['data_set'] + '_test_baseline.txt', 'r') as f:
            print('Test baselines:')
            print(f.read())
    elif position_type == 'single':
        with open(parameters['data_set'] + f'_train_{position}_baseline.txt', 'r') as f:
            print('Train baselines:')
            print(f.read())
        with open(parameters['data_set'] + f'_test_{position}_baseline.txt', 'r') as f:
            print('Test baselines:')
            print(f.read())
          
            
    if parameters['model'] == 'svm':
        window_size = 1
        train_model.logistic_regression_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    elif parameters['model'] == 'random forest':
        window_size = 1
        train_model.random_forest_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)  
    elif parameters['model'] == 'logistic regression':
        window_size = 1
        train_model.bayes_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels, 
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels) 
    else:
        input_dim = X_train.shape[2]
        seq_length = X_train.shape[0]
        output_dim = 2 # 2 or 1
        num_layers = 15 # length of sample
        
        # hidden size = 100, output_dim = 2
        if parameters['model'] == 'lstm':
            net = models.RnnModel(num_layers, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='LSTM')
        # hidden size = 100, output_dim = 2
        elif parameters['model'] == 'gru':
            net = models.RnnModel(num_layers, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='GRU')
        # hidden size = 100, output_dim = 2
        elif parameters['model'] == 'rnn':
            net = models.RnnModel(num_layers, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], cell_type='RNN')
        # hidden size = 128, output_dim = 2
        elif parameters['model'] == 'attention':
            net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])
        elif parameters['model'] == 'da-rnn':
            net = models.DaRnnModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'])
        # output_dim = 1
        elif parameters['model'] == 'transformer':
            net = models.TransformerModel(100, 2, parameters['dropout_p'])
        
        # use gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        X_test = X_test.to(device)
        Y_test = Y_test.to(device) 

        if position_type == 'epitopes':
            position = 0
        

        if parameters['model'] == 'attention' or parameters['model'] == 'da-rnn':
            train_model.train_rnn(net,
                                  False,
                                  parameters['num_of_epochs'],
                                  parameters['learning_rate'],
                                  parameters['batch_size'],
                                  X_train, Y_train,
                                  X_test, Y_test,
                                  True,
                                  parameters['model'],
                                  subtype_flag,
                                  position,
                                  model_path = save_path)
        else:
            train_model.train_rnn(net,
                                  False,
                                  parameters['num_of_epochs'],
                                  parameters['learning_rate'],
                                  parameters['batch_size'],
                                  X_train, Y_train,
                                  X_test, Y_test,
                                  False,
                                  parameters['model'],
                                  subtype_flag,
                                  position,
                                  model_path = save_path)
        

if __name__ == '__main__':
    
    subtype = ['H1N1', 'H3N2', 'H5N1', 'covid19']
    position_mode = ['epitopes', 'single']
    
    subtype_flag, data_path = make_dataset.subtype_selection(subtype[3])
    position_type = position_mode[0]

    # 'svm', 'random forest', 'logistic regression', 'lstm', 'gru', 'rnn', 'attention', 'da-rnn', 'transformer'
    # models_use = ['logistic regression', 'random forest']
    # models_use = ['attention']
    models_use = ['transformer']
    # models_use = ['gru','rnn', 'da-rnn']


    if position_type == 'epitopes':
        for model in models_use:
            print('\n')
            print("Experimental results with model %s on subtype_flag %s:" % (model, subtype_flag))
            main()
        
    elif position_type == 'single':

        if subtype_flag == 0:
            epitope_a = [118, 120, 121, 122, 126, 127, 128, 129, 132, 133, 134, 135, 137, 139, 140, 141, 142, 143, 146, 147, 149, 165, 252, 253]
            epitope_b = [124, 125, 152, 153, 154, 155, 156, 157, 160, 162, 163, 183, 184, 185, 186, 187, 189, 190, 191, 193, 194, 196]
            epitope_c = [34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 269, 270, 271, 272, 273, 274, 276, 277, 278, 283, 288, 292, 295, 297, 298, 302, 303, 305, 306, 307, 308, 309, 310]
            epitope_d = [89, 94, 95, 96, 113, 117, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 179, 198, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 222, 223, 224, 225, 226, 227, 235, 237, 239, 241, 243, 244, 245]
            epitope_e = [47, 48, 50, 51, 53, 54, 56, 57, 58, 66, 68, 89, 70, 71, 72, 73, 74, 75, 78, 79, 80, 82, 83, 84, 85, 86, 102, 257, 258, 259, 260, 261, 263, 267]
            epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
            epitope_positions.sort()

        if subtype_flag == 1:
            epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 136, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168]
            epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198]
            epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312]
            epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248]
            epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
            epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
            epitope_positions.sort()
    
        if subtype_flag == 2:
            epitope_positions = [36, 48, 53, 55, 56, 57, 62, 65, 71, 77, 78, 80, 81, 82, 83, 84, 86, 87, 91, 94, 115, 116, 117, 118, 119, 
                                 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 136, 138, 140, 141, 142, 143, 144, 145, 
                                 149, 150, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 
                                 172, 173, 174, 179, 182, 185, 186, 187, 189, 190, 191, 193, 200, 205, 206, 207, 212, 222, 226, 230, 242, 
                                 244, 245, 246, 252, 256, 259, 261, 262, 263, 273, 274, 276, 278, 282]
            epitope_positions.sort()

        if subtype_flag == 3:
            epitope_positions = [pos for pos in range(318,541)]
        
        for position in epitope_positions:
            for model in models_use:
                main(position)



    

            
            
            
            
            
            
            
            
            