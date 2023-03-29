import torch
import torch.nn as nn
import torch_geometric
import numpy as np
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops


def asymmetric_gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
                        add_self_loops=True, dtype=None):
    """
    edge_index: the edge list, indices starting from zero
    # adapted from
    # https://github.com/pyg-team/pytorch_geometric/blob/901a255346009c7294fd3cc1e825aa441f1dbd4f/
    # torch_geometric/nn/conv/gcn_conv.py#L33

    """
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    from torch_geometric.utils import add_remaining_self_loops, degree
    num_nodes = maybe_num_nodes(edge_index, num_nodes)  # infer # of nodes in the graph

    if edge_weight is None:  # set dummy weights if weight missing
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

    fill_value = 2. if improved else 1.
    if add_self_loops:  # add self loop if absent
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]  # in adj mat, row is the origin, col is the target

    # normalization for directed graph
    out_degree = degree(row)  # in_degree=degree(col)
    out_degree_inv = out_degree.pow_(-1)
    out_degree_inv.masked_fill_(out_degree_inv == float('inf'), 0)
    # a diag mat left-mult a mat means the diag elements row-times a mat
    return edge_index, out_degree_inv[row] * edge_weight


def pairwise_sqrdist(X, epsilon=1e-2):
    """ Computes pairwise distances between node pairs
    :param X: n*d embedding matrix
    :param epsilon: add a small value to distances for numerical stability
    :return: n*n matrix of squared euclidean distances
    """
    s1 = torch.sum(X * X, dim=1, keepdim=True)
    s2 = torch.matmul(X, X.T)
    return s1 + s1.T - 2 * s2 + epsilon


def get_negative_edges(subgraph_data, full_graph_adj, debug=False):
    # ensure there is a local-global map in the data
    assert "n_id" in subgraph_data.keys
    L2G = subgraph_data.n_id

    # add self-loops first so that self-loops won't occur in negative samples
    local_edge_index = add_self_loops(remove_self_loops(subgraph_data.edge_index)[0])[0]

    # get negative edges of subgraph (may contain false negative edges)
    neg_edge_index = negative_sampling(edge_index=local_edge_index,
                                       num_nodes=local_edge_index.max() + 1,
                                       num_neg_samples=subgraph_data.num_nodes)

    if debug:
        print('subgraph nNodes:', subgraph_data.num_nodes, '\n'
                                                           'subgraph nEdges:', subgraph_data.edge_index.shape[1])
        print('neggraph nEdges before filtering:', neg_edge_index.shape[1])
    src, dst = L2G[neg_edge_index[0]], L2G[neg_edge_index[1]]

    # select links that are present in the full graph
    mask = full_graph_adj[src, dst] <= 0

    # cast masks back to local indices
    src = torch.masked_select(neg_edge_index[0], mask)
    dst = torch.masked_select(neg_edge_index[1], mask)
    neg_edge_index = torch.stack([src, dst])
    if debug:
        print('neggraph nEdges after filtering:', neg_edge_index.shape[1])
    return neg_edge_index


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased '
                            '({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
