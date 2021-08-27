import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_features, num_of_vertices):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn((num_of_timesteps, )))
        self.w_2 = nn.Parameter(torch.randn((num_of_features, num_of_timesteps)))
        self.w_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_s = nn.Parameter(torch.randn((1, num_of_vertices, num_of_vertices)))
        self.v_s = nn.Parameter(torch.randn((num_of_vertices, num_of_vertices)))

    def forward(self, x):
        # import pdb;pdb.set_trace()
        lhs = torch.matmul(torch.matmul(x,self.w_1), self.w_2)
        # rhs = torch.matmul(self.w_3, x.permute(2,0,3,1))
        rhs = (self.w_3 * x.permute(2,0,3,1)).squeeze(0)

        product = torch.matmul(lhs, rhs)
        S =torch.matmul(self.v_s,
                  F.sigmoid(product + self.b_s)
                     .permute(1, 2, 0)).permute(2, 0, 1)
        S = S - torch.max(S, axis=1, keepdims=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return S_normalized

# X = SpatialAttention(3,1,4)

# a = torch.randn(1,4,1,3)
# X(a)

class cheb_conv_with_SAt(nn.Module):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''
    def __init__(self, num_of_filters, cheb_polynomials,num_of_features,K = 3, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.randn(self.K, num_of_features, self.num_of_filters))

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):

                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[k]

                # shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.T((0, 2, 1)),
                                   graph_signal)

                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return F.relu(torch.concat(*outputs, dim=-1))



class Temporal_Attention_layer(nn.Module):
    '''
    compute temporal attention scores
    '''
    def __init__(self,num_of_vertices, num_of_features, num_of_timesteps, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        self.U_1 = nn.Parameter(torch.randn((num_of_vertices, )))
        self.U_2 = nn.Parameter(torch.randn((num_of_features, num_of_vertices)))
        self.U_3 = nn.Parameter(torch.randn((num_of_features, )))
        self.b_e = nn.Parameter(torch.randn((1, num_of_timesteps, num_of_timesteps)))
        self.V_e = nn.Parameter(torch.randn((num_of_timesteps, num_of_timesteps)))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape


        # compute temporal attention scores
        # shape is (N, T, V)
        print(f"$$$$$$$$$$$$$$$${x.permute(0, 3, 2, 1).shape},&&&&&&&&{self.U_1.shape}^^^^^^^{self.U_2.shape}&&&&{rhs.size()}")
        lhs = torch.matmul((torch.matmul((x.permute(0, 3, 2, 1).reshape(x.permute(0, 3, 2, 1).size()[2],-1)).T).reshape(1,13,2), self.U_1),
                     self.U_2)

        # shape is (N, V, T)
        # rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
        # rhs = (self.U_3 * x.permute(2, 0, 1, 3)).squeeze(0)
        rhs = torch.einsum('bnvl,v->bnl', (self.U_3, x.permute(2, 0, 1, 3))).contiguous()



        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.V_e,
                   F.sigmoid(product + self.b_e)
                     .permute(1, 2, 0)).permute(2, 0, 1)

        # normailzation
        E = E - torch.max(E, axis=1, keepdims=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, axis=1, keepdims=True)
        return E_normalized

def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, cat_feat_gc=False,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 apt_size=10):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=207, nhead=3)
        self.temporal_attention = Temporal_Attention_layer(num_nodes,2, 13)
        self.t_h = nn.Parameter(torch.empty((13)))
        nn.init.uniform_(self.t_h, -(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))
        self.h_x = nn.Parameter(torch.empty(13, 320, 207))
        nn.init.uniform_(self.h_x,-(1.0 / np.sqrt(2)), (1.0 / np.sqrt(2)))

        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d(in_channels=in_dim - 1,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                                              for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.matmul(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.matmul(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict(dropout=args.dropout, supports=supports,
                        do_graph_conv=args.do_graph_conv, addaptadj=args.addaptadj, aptinit=aptinit,
                        in_dim=args.in_dim, apt_size=args.apt_size, out_dim=args.seq_length,
                        residual_channels=args.nhid, dilation_channels=args.nhid,
                        skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                        cat_feat_gc=args.cat_feat_gc)
        defaults.update(**kwargs)
        model = cls(device, args.num_nodes, **defaults)
        return model

    def load_checkpoint(self, state_dict):
        """It is assumed that ckpt was trained to predict a subset of timesteps."""
        bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']  # only weights that depend on seq_length
        b, w = state_dict.pop(bk), state_dict.pop(wk)
        self.load_state_dict(state_dict, strict=False)
        cur_state_dict = self.state_dict()
        cur_state_dict[bk][:b.shape[0]] = b
        cur_state_dict[wk][:w.shape[0]] = w
        self.load_state_dict(cur_state_dict)

    def forward(self, x):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        # import pdb;pdb.set_trace()
        tmp = torch.clone(x)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else:
            x = self.start_conv(x)
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=1)
            adjacency_matrices = self.fixed_supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*x_in*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i](x)  # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]  # TODO(SS): Mean/Max Pool?
            x = self.bn[i](x)
        x = F.relu(skip)  # ignore last X?
        x = x + torch.einsum('nc,cva->nva', (self.temporal_attention(tmp) @ self.t_h, self.h_x)).contiguous() 
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # downsample to (bs, seq_length, 207, nfeatures)
        return x





