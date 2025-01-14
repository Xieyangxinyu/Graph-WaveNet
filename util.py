import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import torch.nn.functional as F


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def get_mask(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def get_masked_loss(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = (preds-labels)**2
    return get_masked_loss(loss, mask)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)
    return get_masked_loss(loss, mask)


def masked_focal_mae_loss(preds, labels, null_val=np.nan, activate='sigmoid', beta=.2, gamma=1):
    mask = get_mask(labels, null_val=null_val)

    loss = torch.abs(preds-labels)
    loss *= (torch.tanh(beta * torch.abs(preds - labels))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - labels)) - 1) ** gamma
    
    return get_masked_loss(loss, mask)


def masked_focal_mse_loss(preds, labels, null_val=np.nan, activate='sigmoid', beta=.2, gamma=1):
    mask = get_mask(labels, null_val=null_val)

    loss = (preds-labels) ** 2
    loss *= (torch.tanh(beta * torch.abs(preds - labels))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(preds - labels)) - 1) ** gamma
    
    return get_masked_loss(loss, mask)


def masked_mape(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)
    loss = torch.abs(preds-labels)/labels
    return get_masked_loss(loss, mask)


def masked_bmc_loss_1(pred, labels, null_val, noise_var = 1.0):
    pred = pred.squeeze().flatten(0,1)
    labels = labels.squeeze().flatten(0,1)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    pred = pred * mask
    logits = -(pred.unsqueeze(1) - labels.unsqueeze(0)).pow(2).sum(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = torch.nan_to_num(loss)
    return loss


def masked_bmc_loss_9(pred, labels, null_val, noise_var = 9.0):
    pred = pred.squeeze().flatten(0,1)
    labels = labels.squeeze().flatten(0,1)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    pred = pred * mask
    logits = -(pred.unsqueeze(1) - labels.unsqueeze(0)).pow(2).sum(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = torch.nan_to_num(loss)
    return loss


def masked_kirtosis(preds, labels, null_val=np.nan):
    mask = get_mask(labels, null_val=null_val)

    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    aux_loss = (preds-labels) ** 2
    aux_loss = aux_loss * mask
    aux_loss = torch.where(torch.isnan(aux_loss), torch.zeros_like(aux_loss), aux_loss)

    mean = torch.mean(aux_loss)
    std  = torch.std(aux_loss)
    aux_loss = ((aux_loss - mean) / std) ** 4
    loss = loss + 0.01 * aux_loss
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=np.nan, beta=1.):
    mask = get_mask(labels, null_val=null_val)

    l1_loss = torch.abs(preds-labels)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)

    return get_masked_loss(loss, mask)


def masked_Gumbel(preds, labels, null_val=np.nan, gamma = 1.1):
    mask = get_mask(labels, null_val=null_val)

    l2_loss = (preds-labels) ** 2
    loss = ((1 - torch.exp(-l2_loss)) ** gamma) * l2_loss

    return get_masked_loss(loss, mask)


def masked_Frechet(preds, labels, null_val=np.nan, alpha = 13, s = 1.7):
    mask = get_mask(labels, null_val=null_val)
    
    l1_loss = torch.abs(preds-labels)
    transform = (l1_loss + s * ( (alpha / (1 + alpha)) ** (1 / alpha) )) / s
    loss = (-1 - alpha) * ( (-transform) ** (-alpha)) + torch.log(transform)

    return get_masked_loss(loss, mask)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse