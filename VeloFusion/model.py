from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .layer import GiEncoder, GiEdgePredictor, FCDecoder
from .loss import kl_loss, feature_recon_loss, edge_recon_loss, mmd


class VeloFusion(nn.Module):
    def __init__(self, feature_dim=None, hidden_dim=None, z_dim=None, cond_dim=None,
                 FC_layer_sizes = [100, 500, 800],
                 residual_encoder =False,
                 gravity_lambda=10.0,
                 KLD_beta=1.0,
                 use_negative_balancing=True,
                 feature_gamma1=1.0,
                 edge_gamma2=1.0, feature_recon_w_v=0.5,
                 mmd_weight=1.0, mmd_on=None, mmd_beta=1.0, mmd_boundary: Optional[int] = None
                 ):
        super().__init__()
        # 网络结构基本参数
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.FC_layer_sizes = FC_layer_sizes
        self.residual_encoder = residual_encoder
        self.gravity_lambda = gravity_lambda
        self.feature_recon_w_v = feature_recon_w_v

        # 层定义
        self.encoder = GiEncoder(input_dim=self.feature_dim, hidden_dim=self.hidden_dim,
                                 z_dim=self.z_dim, cond_dim=self.cond_dim,
                                 use_residual=self.residual_encoder)
        self.edge_decoder = GiEdgePredictor(latent_size=self.z_dim, normalize=True,
                                            gravity_lambda=self.gravity_lambda,
                                            cond_dim=self.cond_dim)
        self.feature_decoder = FCDecoder(latent_size=self.z_dim,
                                         layer_sizes=self.FC_layer_sizes+[self.feature_dim],
                                         class_size=self.cond_dim,
                                         norm_method='bn', dr_rate=0.2)

        # 损失函数参数
        self.use_negative_balancing = use_negative_balancing
        self.KLD_beta = KLD_beta
        self.feature_gamma1 = feature_gamma1
        self.edge_gamma2 = edge_gamma2
        self.mmd_on = mmd_on
        self.mmd_beta = mmd_beta
        self.mmd_boundary = mmd_boundary
        self.mmd_weight = mmd_weight
        self.MAX_LOGSTD = 10
        pass

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            MAX_LOGSTD = 10  # very important trick
            logstd = logstd.clamp(max=MAX_LOGSTD)
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, edge_index, edge_weight, cond_num):
        """
        forward() 接受的输入是：一组迁移图上的局部邻域
        x: 表示节点上的特征
        edge_index: 表示边集数组
        edge_index: 表示边权
        batch_num: 为cVAE的生成条件，是一个从零开始的整数，encoder会把这个整数编码程one-hot向量的。
        """
        z_mu, z_logstd = self.encoder(x, edge_index, edge_weight, cond_num)
        z_logstd = z_logstd.clamp(max=self.MAX_LOGSTD)
        z = self.reparametrize(z_mu, z_logstd)
        adj_pred = self.edge_decoder.forward_all(z, cond_num)
        x_pred = self.feature_decoder(z, cond_num)
        return x_pred, adj_pred, z_mu, z_logstd

    def loss(self, x, edge_index, edge_weight, cond_num):
        """
        loss() 接受的输入是：一组迁移图上的局部子图，输出loss
        x 表示节点上的特征
        edge_index: 表示边集数组,
        edge_index: 表示边权,
        batch_num: 为cVAE的生成条件，是一个从零开始的整数，encoder会把这个整数编码程one-hot向量的,
        full_graph_adj: 全图邻接矩阵，方便生成negatively sampled links
        """
        z_mu, z_logstd = self.encoder(x, edge_index, edge_weight, cond_num)
        z_logstd = z_logstd.clamp(max=self.MAX_LOGSTD)
        z = self.reparametrize(z_mu, z_logstd)
        adj_pred = self.edge_decoder.forward_all(z, cond_num)
        x_pred = self.feature_decoder(z, cond_num)

        N = x.shape[0]
        KLD = kl_loss(z_mu, z_logstd) / N
        ERL = edge_recon_loss(recon_adj=adj_pred,
                              edge_index=edge_index,
                              edge_weight=edge_weight,
                              use_negative_balancing=self.use_negative_balancing, loss_method='cross_entropy')
        FRL, FRL_x, FRL_v = feature_recon_loss(x=x, recon=x_pred, w_v=self.feature_recon_w_v)

        if self.mmd_on == "latent":
            MMD = mmd(z, cond_num, self.cond_dim, self.mmd_beta, self.mmd_boundary)
        else:
            MMD = torch.tensor(0.0, device=z.device)

        Loss = self.KLD_beta * KLD + \
               self.edge_gamma2 * ERL + \
               self.feature_gamma1 * FRL + \
               self.mmd_weight * MMD
        loss_info = {
            'KLD': KLD, 'FRL': FRL, 'ERL': ERL, 'MMD': MMD,
            'FRL_x': FRL_x, 'FRL_v':FRL_v,
            'KLD_beta': self.KLD_beta,
            'feature_gamma1': self.edge_gamma2,
            'feature_recon_w_v': self.feature_recon_w_v,
            'edge_gamma2': self.edge_gamma2,
            'mmd_weights': self.mmd_weight,
            'mmd_beta': self.mmd_beta,
            'gravity_lambda': self.gravity_lambda
        }
        return Loss, loss_info
