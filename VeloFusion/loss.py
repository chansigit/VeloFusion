import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj


def edge_recon_loss(recon_adj: Tensor, edge_index: Tensor, edge_weight: Tensor,
                    use_negative_balancing=True, loss_method='cross_entropy', EPS=1e-15):
    """
    cross_entropy = $- \sum_ij A_ij * log P_ij + (1-A_ij) * log(1-P_ij)$
    """
    if loss_method == "cross_entropy":
        if not use_negative_balancing:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)
            # recon_adj, adj = recon_adj.reshape(-1), adj.reshape(-1)
            ce = -(adj * torch.log(recon_adj+EPS) + (1-adj)*torch.log(1-recon_adj+EPS)).type(torch.float32).mean()
            return ce
        else:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)
            adj_w, recon_w = adj.reshape(-1), recon_adj.reshape(-1)
            pos_mask = (adj_w > 0)
            pos_adj_w, neg_adj_w = torch.masked_select(adj_w, pos_mask), torch.masked_select(adj_w, ~pos_mask)
            pos_recon_w, neg_recon_w = torch.masked_select(recon_w, pos_mask), torch.masked_select(recon_w, ~pos_mask)

            pos_ce = -(pos_adj_w * torch.log(pos_recon_w + EPS) +
                       (1-pos_adj_w) * torch.log(1-pos_recon_w+EPS)
                       ).type(torch.float32).mean()
            neg_ce = -(neg_adj_w * torch.log(neg_recon_w + EPS) +
                       (1 - neg_adj_w) * torch.log(1 - neg_recon_w + EPS)
                       ).type(torch.float32).mean()
            return pos_ce+neg_ce

    elif loss_method == 'mse':
        # Naive version
        # this implementation doesn't take the graph sparsity into consideration.
        # we want false negatives and false positives contributed equally to the loss
        if not use_negative_balancing:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)
            recon_loss = torch.norm(recon_adj - adj, p='fro')
            return recon_loss
        else:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)
            graph_density = (edge_index.shape[1]) / (recon_adj.shape[0]*recon_adj.shape[0])
            adj_w, recon_w = adj.reshape(-1), recon_adj.reshape(-1)
            pos_mask = (adj_w > 0)
            pos_adj_w, neg_adj_w = torch.masked_select(adj_w,  pos_mask), torch.masked_select(adj_w, ~pos_mask)
            pos_recon_w, neg_recon_w = torch.masked_select(recon_w, pos_mask), torch.masked_select(recon_w, ~pos_mask)

            pos_loss = torch.norm(pos_adj_w - pos_recon_w, p='fro') / graph_density
            neg_loss = torch.norm(neg_adj_w - neg_recon_w, p='fro') / (1-graph_density)
            recon_loss= pos_loss+neg_loss
            print("pos %f neg %f"%( pos_loss, neg_loss), ' density=', graph_density)
            return recon_loss
    print('loss not implemented!')
    return None



def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.
       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       alphas: Tensor
       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def partition(data, partitions, num_partitions):
    """
    borrowed from
    https://github.com/theislab/scarches/blob/f595ce61605e26418e14ec06360a0772df3e3738/scarches/
    models/trvae/_utils.py#L18

    """
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def mmd_loss_calc(source_features, target_features):
    """
    borrowed from
    https://github.com/theislab/scarches/blob/3558b3cefc6b5156f20fe5ecaa105e8914deb85c/scarches/models/
    trvae/losses.py

    Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.
    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.
    Parameters
    ----------
    source_features: torch.Tensor,
        Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor,
        Tensor with shape [batch_size, z_dim]
    Returns
    -------
    Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost


def mmd(y, c, n_conditions, beta, boundary):
    """
    Borrowed from
    https://github.com/theislab/scarches/blob/3558b3cefc6b5156f20fe5ecaa105e8914deb85c/scarches/models/trvae/losses.py

    Initializes Maximum Mean Discrepancy(MMD) between every different condition.
    Parameters
    ----------
    n_conditions: integer
        Number of classes (conditions) the data contain.
    beta: float
        beta coefficient for MMD loss.
    boundary: integer
        If not 'None', mmd loss is only calculated on #new conditions.
    y: torch.Tensor
        Torch Tensor of computed latent data.
    c: torch.Tensor
        Torch Tensor of condition labels.
    Returns
    -------
    Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = partition(y, c, n_conditions)
    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            for j in range(boundary, n_conditions):
                if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i in range(len(conditions_mmd)):
            if conditions_mmd[i].size(0) < 1:
                continue
            for j in range(i):
                if conditions_mmd[j].size(0) < 1 or i == j:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])

    return beta * loss


def kl_loss(mu: Tensor, logstd: Tensor) -> Tensor:
    r"""Computes the KL loss, either for the passed arguments :obj:`mu` and :obj:`logstd`
    """
    MAX_LOGSTD = 10.0 # very important trick
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl = -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
    return kl


def feature_recon_loss(x: Tensor, recon: Tensor, w_v: float) -> Tensor:
    variable_len = x.shape[1] // 2
    recon_e, recon_v = recon.split(variable_len, dim=1)
    x_e, x_v = x.split(variable_len, dim=1)
    MSE_x = F.mse_loss(recon_e, x_e, reduction='mean')  # or sum
    MSE_v = F.mse_loss(recon_v, x_v, reduction='mean')
    recon_loss = (1 - w_v) * MSE_x + w_v * MSE_v
    return recon_loss, MSE_x, MSE_v
