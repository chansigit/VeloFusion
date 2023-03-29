import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.typing import Adj, OptTensor
from .util import asymmetric_gcn_norm, pairwise_sqrdist


class GiGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, cached=False, improved=False,
                 add_self_loops=True, normalize=True,
                 **kwargs):
        """
        A wrapper class of GCNConv, with a different normalize behaviour.
        Supported arguments are inherited from torch_geometric.nn.GCNConv, including:
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = False, (default value is False in our implementation, because we have our own norm way)
        bias: bool = True
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.conv = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, normalize=False)
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """
        perform directed graph normalization first
        and then pass the normalized matrix to base class GCNConv forward()
        """
        # do normalization
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = asymmetric_gcn_norm(edge_index, edge_weight,
                                                              num_nodes=x.size(0),
                                                              improved=self.improved,
                                                              add_self_loops=self.add_self_loops,
                                                              dtype=x.dtype
                                                              )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
        else:
            print('edge_index must be torch.Tensor! sparse tensor unsupported yet!')

        return self.conv.forward(x, edge_index, edge_weight)


class GiEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, cond_dim=1, use_residual=False):
        super(GiEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.use_residual = use_residual
        
        if self.use_residual:
            self.conv0 = GiGCNConv(in_channels=-1, out_channels=500) # input from x_
            self.conv1 = GiGCNConv(in_channels=-1, out_channels=500) # input from conv0
            self.ln1 = LayerNorm(500)
            self.conv2 = GiGCNConv(in_channels=-1, out_channels=500) # input from conv1+ln1
            self.ln2 = LayerNorm(500) # outputs merged with conv0
        
        self.gcn_shared = GiGCNConv(in_channels=-1, out_channels=self.hidden_dim)  # 自动推维度
        self.gcn_mu = GiGCNConv(in_channels=self.hidden_dim, out_channels=self.z_dim)
        self.gcn_logstd = GiGCNConv(in_channels=self.hidden_dim, out_channels=self.z_dim)

    def forward(self, x, edge_index, edge_weight, cond):
        c = torch.nn.functional.one_hot(cond, self.cond_dim)
        x_ = torch.cat([x, c], 1)
        if not self.use_residual:
            x_ = F.relu(self.gcn_shared(x_, edge_index, edge_weight))
            mu = self.gcn_mu(x_, edge_index, edge_weight)
            logstd = self.gcn_logstd(x_, edge_index, edge_weight)
            return mu, logstd
        else:
            residual = self.conv0(x_, edge_index, edge_weight)
            conv1_out = F.relu(self.ln1(self.conv1(residual,  edge_index, edge_weight)))
            conv1_out = F.dropout(conv1_out, p=0.1, training=self.training)
            conv2_out = self.ln2(self.conv2(conv1_out, edge_index, edge_weight))
            out = F.relu(conv2_out+residual)
            out = F.relu(self.gcn_shared(out, edge_index, edge_weight))
            mu = self.gcn_mu(out, edge_index, edge_weight)
            logstd = self.gcn_logstd(out, edge_index, edge_weight)
            return mu, logstd


class GiEdgePredictor(torch.nn.Module):
    """Gravity-Inspired asymmetric decoder for directed link prediction.
    """

    def __init__(self, latent_size, cond_dim, normalize=False, dropout=0.,
                 act=torch.sigmoid, epsilon=1e-2, gravity_lambda=1, **kwargs):
        super(GiEdgePredictor, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.normalize = normalize
        self.latent_size = latent_size
        self.gravity_lambda = gravity_lambda
        self.cond_dim = cond_dim
        self.epsilon = epsilon

    def forward_all(self, inputs, cond_num):
        """
        predict links from nodes' latent embeddings
        TODO: revise the dropout implementation. - DONE! 2023-03-04
        """
        # inputs = torch.nn.Dropout(p=self.dropout)(inputs)
        # maybe we should apply dropouts on input_z| NOW,dropout input_z|
        # Mass parameter = last dimension of input
        # Embedding vector = all dimensions on input except the last
        cond = torch.nn.functional.one_hot(cond_num, self.cond_dim)
        inputs_m = inputs[:, (self.latent_size - 1):self.latent_size]
        inputs_z = inputs[:, 0:(self.latent_size - 1)]
        # https://github.com/deezer/gravity_graph_autoencoders/issues/11
        inputs_z = torch.nn.Dropout(p=self.dropout)(inputs_z)
        if self.normalize:
            inputs_z = torch.nn.functional.normalize(inputs_z, p=2, dim=1)

        inputs_z = torch.cat([inputs_z, cond], 1)

        # get pairwise Euclidean distances between nodes in the latent space
        dist = pairwise_sqrdist(inputs_z, epsilon=self.epsilon)  # nxn mat
        # expand the mass to rows
        mass = torch.matmul(torch.ones([inputs_m.shape[0], 1], device=inputs.device), inputs_m.T)  # nxn mat

        # Gravity-Inspired decoding
        outputs = mass - self.gravity_lambda * torch.log(dist)
        # outputs = torch.reshape(outputs, [-1])
        outputs = self.act(outputs)  # the output is a nxn adj mat
        return outputs


class FCDecoder(nn.Module):
    """Decoder class. It will transform the
       constructed latent space to the previous space of data with n_dimensions = x_dimension.
       Parameters
       ----------
       latent_size: Integer,
            Bottleneck layer (z)  size.
       layer_sizes: List,
            List of hidden and last layer sizes
       class_size: Integer,
            Number of classes (batches) the data contain.
       norm_method: str,
            Normalization method applied to layers. Valid values are: "none", bn", "ln"
       dr_rate: Float,
            Dropout rate applied to all layers, if `dr_rate`==0 no dropout will be applied.

    """

    def __init__(self,
                 layer_sizes: list,
                 latent_size: int,
                 class_size: int,
                 norm_method: str,
                 dr_rate: float,
                 activation_func=nn.GELU,
                 ):
        super().__init__()
        self.class_size = class_size
        assert len(layer_sizes) > 1
        layer_sizes = [latent_size] + layer_sizes

        ######################################################################
        # ========================  Add FC layers  ========================= #
        ######################################################################
        # the design of layer orders follows:
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        self.FC = nn.Sequential()
        for layer_id in range(len(layer_sizes) - 2):
            in_size, out_size = layer_sizes[layer_id], layer_sizes[layer_id + 1]
            if layer_id == 0:
                in_size += class_size
            # linear layer
            L = nn.Linear(in_size, out_size)
            self.FC.add_module(name="L%d" % layer_id, module=L)

            # normalization
            if norm_method.lower() == "bn":
                Norm = nn.BatchNorm1d(out_size, affine=True)
                self.FC.add_module("N%d" % layer_id, module=Norm)
            elif norm_method.lower() == "ln":
                Norm = nn.LayerNorm(out_size, elementwise_affine=False)
                self.FC.add_module("N%d" % layer_id, module=Norm)

            # activation
            A = activation_func()
            self.FC.add_module(name="A%d" % layer_id, module=A)

            # dropout
            if dr_rate > 0:
                D = nn.Dropout(p=dr_rate)
                self.FC.add_module(name="D%d" % layer_id, module=D)
        # --------------------------------end---------------------------------

        ######################################################################
        # ===================  Reconstruction Decoder  ===================== #
        ######################################################################
        self.expr_decoder = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1] // 2),
            nn.ReLU()
        )
        self.velo_decoder = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1] // 2),
            nn.LeakyReLU()
        )
        # --------------------------------end---------------------------------

    def forward(self, z, class_num):
        class_num = torch.nn.functional.one_hot(class_num, self.class_size)
        # batch = one_hot(batch, class_size=self.class_size) 就不手写one_hot了
        z_ = torch.cat([z, class_num], dim=1)
        h = self.FC(z_)
        recon_expr = self.expr_decoder(h)
        recon_velo = self.velo_decoder(h)
        recon_x = torch.cat([recon_expr, recon_velo], dim=1)
        return recon_x
