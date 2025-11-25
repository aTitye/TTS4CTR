import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureEmbedding(torch.nn.Module):

    def __init__(self, feature_num, latent_dim, initializer=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_num, latent_dim))
        initializer(self.embedding)

    def forward(self, x):
        """
        :param x: tensor of size (batch_size, num_fields)
        :return: tensor of size (batch_size, num_fields, embedding_dim)
        """
        return F.embedding(x, self.embedding)


class FeaturesLinear(torch.nn.Module):

    def __init__(self, feature_num, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(feature_num, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim, )))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return : tensor of size (batch_size, 1)
        """
        return torch.sum(torch.squeeze(self.fc(x)), dim=1, keepdim=True) + self.bias


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1) if reduce_sum
                  tensor of size (batch_size, embed_dim) else   
        """
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False, use_ln=False):
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, mlp_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(mlp_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(mlp_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        return self.mlp(x)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim, ))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CrossNetworkV2(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim, ))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * (xw + self.b[i]) + x
        return x


class InnerProduct(torch.nn.Module):

    def __init__(self, field_num):
        super().__init__()
        self.rows = []
        self.cols = []
        for row in range(field_num):
            for col in range(row + 1, field_num):
                self.rows.append(row)
                self.cols.append(col)
        self.rows = torch.tensor(self.rows)
        self.cols = torch.tensor(self.cols)

    def forward(self, x):
        """
        :param x: Float tensor of size (batch_size, field_num, embedding_dim)
        :return: (batch_size, field_num*(field_num-1)/2)
        """
        batch_size = x.shape[0]
        trans_x = torch.transpose(x, 1, 2)

        self.rows = self.rows.to(trans_x.device)
        self.cols = self.cols.to(trans_x.device)

        gather_rows = torch.gather(trans_x, 2, self.rows.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans_x, 2, self.cols.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding


class LinearCompressBlock(torch.nn.Module):

    def __init__(self, num_embed_in, num_embed_out):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros((num_embed_in, num_embed_out)))
        torch.nn.init.kaiming_uniform_(self.w)

    def forward(self, x):
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = torch.permute(x, (0, 2, 1))

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.w

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = torch.permute(outputs, (0, 2, 1))
        return outputs


class FactorizationMachineBlock(torch.nn.Module):

    def __init__(self, num_emb_in, num_emb_out, latent_dim, rank, fm_mlp_dims, dropout):
        super().__init__()
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.latent_dim = latent_dim
        self.rank = rank
        self.w = torch.nn.Parameter(torch.zeros((num_emb_in, rank)))
        self.norm = torch.nn.LayerNorm(num_emb_in * rank)
        self.mlp = MultiLayerPerceptron(
            num_emb_in * rank, fm_mlp_dims + [num_emb_out * latent_dim], dropout, output_layer=False
        )

        torch.nn.init.kaiming_uniform_(self.w)

    def forward(self, x):
        """
        :param x: Float tensor of size (batch_size, num_emb_in, dim_emb)
        :return: Float tensor of size (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = x.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.w

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = torch.bmm(x, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = outputs.view(-1, self.num_emb_in * self.rank)

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.view(-1, self.num_emb_out, self.latent_dim)

        return outputs


class ResidualProjection(torch.nn.Module):

    def __init__(self, num_emb_in, num_emb_out):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros((num_emb_in, num_emb_out)))
        torch.nn.init.kaiming_normal_(self.w)

    def forward(self, x):
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = x.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.w

        # # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)

        return outputs


class WukongLayer(torch.nn.Module):

    def __init__(self, num_emb_in, latent_dim, num_emb_lcb, num_emb_fmb, rank_fmb, fm_mlp_dims, dropout):
        super().__init__()
        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            latent_dim,
            rank_fmb,
            fm_mlp_dims,
            dropout,
        )
        self.norm = torch.nn.LayerNorm(latent_dim)

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb)
        else:
            self.residual_projection = torch.nn.Identity()

    def forward(self, inputs):
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = torch.concat((fmb, lcb), dim=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs


class TokenMixer(torch.nn.Module):

    def __init__(self, num_T, num_D, num_H):
        super().__init__()
        assert num_D % num_H == 0, "num_D should be divisible by num_H"
        assert num_T == num_H, "num_T should be equal to num_H"
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D // num_H

    def forward(self, x):  # (bs, num_T, num_D)
        x = x.contiguous().view(-1, self.num_T, self.num_H, self.d_k)  # (bs, num_T, num_H, d_k)
        x = x.permute(0, 2, 1, 3)  # (bs, num_H, num_T, d_k)
        x = x.contiguous().view(-1, self.num_H, self.num_T * self.d_k)  # (bs, num_H, num_T * d_k)
        return x


class PerTokenFN(torch.nn.Module):

    def __init__(self, num_T, num_D, expansion_rate):
        super(PerTokenFN, self).__init__()
        self.pffn = torch.nn.ModuleList()
        for _ in range(num_T):
            self.pffn.append(
                torch.nn.Sequential(
                    torch.nn.Linear(num_D, num_D * expansion_rate),
                    torch.nn.GELU(),
                    torch.nn.Linear(num_D * expansion_rate, num_D),
                )
            )

    def forward(self, x):  # (bs, num_T, num_D)
        outputs = []
        for i in range(x.shape[1]):
            outputs.append(self.pffn[i](x[:, i, :]))
        return torch.stack(outputs, dim=1)


class RankMixerLayer(torch.nn.Module):

    def __init__(self, num_T, num_D, num_H, expansion_rate):
        super(RankMixerLayer, self).__init__()
        self.token_mixing = TokenMixer(num_T, num_D, num_H)
        self.pffn = PerTokenFN(num_T, num_D, expansion_rate)

    def forward(self, x):
        mixed_x = self.token_mixing(x)
        x = F.layer_norm(mixed_x + x, x.shape)
        x = F.layer_norm(x + self.pffn(x), x.shape)
        return x


class InteractionAggregrationLayer(torch.nn.Module):

    def __init__(self, dim_1, dim_2, output_dim=1, num_heads=1):
        super().__init__()
        assert dim_1 % num_heads == 0, "dim_1 should be divisible by num_heads"
        assert dim_2 % num_heads == 0, "dim_2 should be divisible by num_heads"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim1 = dim_1 // num_heads
        self.head_dim2 = dim_2 // num_heads
        self.w_1 = torch.nn.Linear(dim_1, output_dim)
        self.w_2 = torch.nn.Linear(dim_2, output_dim)
        self.w = torch.nn.Parameter(torch.zeros((num_heads * self.head_dim1 * self.head_dim2, output_dim)))
        torch.nn.init.xavier_normal_(self.w)

    def forward(self, x1, x2):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        output = self.w_1(x1) + self.w_2(x2)  # (bs, num_fields, output_dim)
        head_1 = x1.view(-1, self.num_heads, self.head_dim1)  # (bs, num_fields, num_heads, head_dim1)
        head_2 = x2.view(-1, self.num_heads, self.head_dim2)  # (bs, num_fields, num_heads, head_dim2)
        residual = torch.matmul(head_1.unsqueeze(2), self.w.view(self.num_heads, self.head_dim1, -1))
        residual = torch.matmul(residual.view(-1, self.num_heads, self.output_dim, self.head_dim2),
                                head_2.unsqueeze(-1)).squeeze(-1)
        output += residual.sum(dim=1)  # (bs, output_dim)
        return output
