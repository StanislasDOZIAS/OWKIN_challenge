import numpy as np
from tqdm import tqdm

def hausdorf_matrix_single(bags):
    """Hausdorf distance for a single set of bags."""
    n = bags.shape[0]
    hausdorf = np.zeros((n, n, 3))
    for i in tqdm(range(n)):
        for j in range(i, n):
            dist = np.einsum('ij,ij->i', bags[i], bags[i])[:,None] + np.einsum('ij,ij->i', bags[j], bags[j]) - 2 * np.dot(bags[i], bags[j].T)
            
            min1 = np.min(dist, axis=-1)
            min2 = np.min(dist, axis=0)

            # Vanilla Hausdorf
            dist1 = np.max(min1)
            dist2 = np.max(min2)
            H_max = max(dist1, dist2)

            # Mean Hausdorf
            H_mean = (np.sum(min1) + np.sum(min2)) / (np.size(min1) + np.size(min2)) 

            # Min Hausdorf
            H_min = np.min(min1)

            hausdorf[i, j] = np.array([H_max, H_mean, H_min])
            hausdorf[j, i] = np.array([H_max, H_mean, H_min])
    return hausdorf

def hausdorf_matrix_dual(bags_pred, bags_train):
    """Hausdorf distance between two set of bags."""
    n = bags_pred.shape[0]
    m = bags_train.shape[0]
    hausdorf = np.zeros((n, m, 3))
    for i in tqdm(range(n)):
        for j in range(m):
            dist = np.einsum('ij,ij->i', bags_pred[i], bags_pred[i])[:,None] + np.einsum('ij,ij->i', bags_train[j], bags_train[j]) - 2 * np.dot(bags_pred[i], bags_train[j].T)
            
            min1 = np.min(dist, axis=-1)
            min2 = np.min(dist, axis=0)

            # Vanilla Hausdorf
            dist1 = np.max(min1)
            dist2 = np.max(min2)
            H_max = max(dist1, dist2)

            # Mean Hausdorf
            H_mean = (np.sum(min1) + np.sum(min2)) / (np.size(min1) + np.size(min2)) 

            # Min Hausdorf
            H_min = np.min(min1)

            hausdorf[i, j] = np.array([H_max, H_mean, H_min])
    return hausdorf

def get_matrices(center1, center2, center_test, y_dict, norm=''):
    # Get labels
    y1 = y_dict[center1]
    y2 = y_dict[center2]
    y_train = np.concatenate((y1, y2))
    y_test = y_dict[center_test]
    
    # Get matrices
    n1 = y1.size
    n2 = y2.size
    n_train = n1 + n2
    n_test = y_test.size

    name = 'hausdorf'
    if norm != '':
        name += '_' + norm

    distances_train = np.zeros((n_train, n_train, 3))
    distances_train[:n1, :n1] = np.load(f'{name}_{center1}.npy')
    distances_train[n1:, n1:] = np.load(f'{name}_{center2}.npy')
    distances_train[:n1, n1:] = np.load(f'{name}_{center1}_{center2}.npy')
    distances_train[n1:, :n1] = np.moveaxis(np.load(f'{name}_{center1}_{center2}.npy'), 0, 1)
    distances_train[distances_train < 0] = 0

    distances_pred = np.zeros((n_train, n_test, 3))
    if center1 < center_test:
        distances_pred[:n1] = np.load(f'{name}_{center1}_{center_test}.npy')
    else:
        distances_pred[:n1] = np.moveaxis(np.load(f'{name}_{center_test}_{center1}.npy'), 0, 1)
    
    if center2 < center_test:
        distances_pred[n1:] = np.load(f'{name}_{center2}_{center_test}.npy')
    else:
        distances_pred[n1:] = np.moveaxis(np.load(f'{name}_{center_test}_{center2}.npy'), 0, 1)
    distances_pred[distances_pred < 0] = 0

    return distances_train, y_train, distances_pred, y_test

def get_train_matrix(y_dict, norm=''):
    center1 = 'C_1'
    center2 = 'C_2' 
    center_test = 'C_5' 

    name = 'hausdorf'
    if norm != '':
        name += '_' + norm

    distances_train, y_train, distances_pred, y_test = get_matrices(center1, center2, center_test, y_dict, norm=norm)

    n1, n2, _ = distances_pred.shape

    distances_matrix = np.zeros((n1 + n2, n1 + n2, 3))
    distances_matrix[:n1, :n1] = distances_train
    distances_matrix[:n1, n1:] = distances_pred
    distances_matrix[n1:, :n1] = np.moveaxis(distances_pred, 0, 1)
    distances_matrix[n1:, n1:] = np.load(f'{name}_{center_test}.npy')

    y = np.zeros(n1 + n2)
    y[:n1] = y_train
    y[n1:] = y_test

    return distances_matrix, y