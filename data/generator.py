import torch
import numpy as np
from torch.nn import functional as F
import gc

def generate_waterbirds_dataset(X_train, g_train, spuriosity_percent):
    gc.collect()
    spuriosity_num_samples_dict = {100.: [3498, 0, 0, 1057],
                                    99.: [3498, 35, 11, 1057],
                                    95.: [3498, 184, 56, 1057],
                                    90.: [1650, 184, 56, 500], 
                                    80.: [735, 184, 56, 223],
                                    70.: [429, 184, 56, 130]}
    num_samples_group0 = spuriosity_num_samples_dict[spuriosity_percent][0]
    num_samples_group1 = spuriosity_num_samples_dict[spuriosity_percent][1]
    num_samples_group2 = spuriosity_num_samples_dict[spuriosity_percent][2]
    num_samples_group3 = spuriosity_num_samples_dict[spuriosity_percent][3]
    
    indices_group0 = np.random.choice(np.where(g_train[:, 0] == 1)[0], num_samples_group0, replace=False)
    indices_group1 = np.random.choice(np.where(g_train[:, 1] == 1)[0], num_samples_group1, replace=False)
    indices_group2 = np.random.choice(np.where(g_train[:, 2] == 1)[0], num_samples_group2, replace=False)
    indices_group3 = np.random.choice(np.where(g_train[:, 3] == 1)[0], num_samples_group3, replace=False)
    
    selected_indices = np.concatenate((indices_group0, indices_group1, indices_group2, indices_group3))
    
    X_spurious = X_train[selected_indices]
    g_spurious = g_train[selected_indices]
    y_spurious = F.one_hot(g_spurious.argmax(1)//2)
    
    return X_spurious, y_spurious, g_spurious