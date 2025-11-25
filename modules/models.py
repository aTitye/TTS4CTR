import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from modules.layers import MultiLayerPerceptron, FactorizationMachine, FeaturesLinear, FeatureEmbedding, WukongLayer, InteractionAggregrationLayer
import modules.layers as layer


class BasicModel(torch.nn.Module):

    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def get_embedding(self, x):
        """
        :param x: Long tensor of size ``(batch_size, field_num)``
        :return: Float tensor of size ``(batch_size, field_num, latent_dim)``
        """
        return self.embedding(x.type(torch.LongTensor).to(x.device()))

    def get_embedding_with_mask(self, x, index):
        """
        :param x: Long tensor of size ``(batch_size, field_num)``
        :param index: int, the index of the field to mask
        :return: Float tensor of size ``(batch_size, field_num, latent_dim)``
        """
        x_embedding = self.embedding(x)
        non_index = [i for i in range(self.field_num) if i != index]
        x_embedding[:, non_index, :] = 0.0
        return x_embedding

    def get_embedding_with_noise(self, x):
        x_embedding = self.embedding(x)
        x_embedding = x_embedding + torch.normal(0, 1, size=x_embedding.shape).to(x_embedding.device) * 0.005  # Add noise
        return x_embedding

    def reg(self):
        return 0.0


class GroupModel(torch.nn.Module):

    def __init__(self, opt, modellist):
        super(GroupModel, self).__init__()
        self.models = nn.ModuleList()
        model_dims = opt.get("model_dims", {}) if opt is not None else {}

        for model_name in modellist:
            name = model_name.lower()
            sub_opt = dict(opt)
            sub_opt.pop("model_dims", None)
            override_dim = model_dims.get(name)
            if override_dim is not None:
                sub_opt["latent_dim"] = override_dim
            self.models.append(build_model(name, sub_opt))

    def forward(self, x):
        logits = [model(x).unsqueeze(-1) for model in self.models]
        stacked = torch.cat(logits, dim=-1)
        return torch.mean(stacked, dim=-1)

    def reg(self):
        tensor_total: Optional[torch.Tensor] = None
        scalar_total = 0.0
        for sub_model in self.models:
            reg_fn = getattr(sub_model, "reg", None)
            if not callable(reg_fn):
                continue
            reg_value = reg_fn()
            if reg_value is None:
                continue
            if isinstance(reg_value, torch.Tensor):
                tensor_total = reg_value if tensor_total is None else tensor_total + reg_value
            elif isinstance(reg_value, (int, float)):
                scalar_total += float(reg_value)
            else:
                continue

        if tensor_total is None:
            return scalar_total
        if scalar_total:
            tensor_total = tensor_total + tensor_total.new_tensor(scalar_total)
        return tensor_total


class FM(BasicModel):

    def __init__(self, opt):
        super(FM, self).__init__(opt)
        # self.linear = FeaturesLinear(self.feature_num)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_embedding = self.embedding(x)
        # output_linear = self.linear(x)
        output_fm = self.fm(x_embedding)

        logit = output_fm
        return logit


class DeepFM(FM):

    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        # output_linear = self.linear(x)

        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        # logit = output_dnn +  output_fm + output_linear
        logit = output_dnn + output_fm
        return logit

    def forward_with_mask(self, x, index):
        x_embedding = self.get_embedding_with_mask(x, index)
        # output_linear = self.linear(x)

        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        # logit = output_dnn +  output_fm + output_linear
        logit = output_dnn + output_fm
        return logit

    def forward_with_random_mask(self, x):
        x_embedding = self.get_embedding_with_noise(x)
        # output_linear = self.linear(x)

        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        # logit = output_dnn +  output_fm + output_linear
        logit = output_dnn + output_fm
        return logit


class DeepCrossNet(BasicModel):

    def __init__(self, opt):
        super(DeepCrossNet, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

    def forward_with_mask(self, x, index):
        x_embedding = self.get_embedding_with_mask(x, index)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

    def forward_with_random_mask(self, x):
        x_embedding = self.get_embedding_with_noise(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit


class DeepCrossNetV2(BasicModel):

    def __init__(self, opt):
        super(DeepCrossNetV2, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross = layer.CrossNetworkV2(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit


class InnerProductNet(BasicModel):

    def __init__(self, opt):
        super(InnerProductNet, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim + \
            int(self.field_num * (self.field_num - 1) / 2)
        self.inner = layer.InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit

    def forward_with_mask(self, x, index):
        x_embedding = self.get_embedding_with_mask(x, index)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit

    def forward_with_random_mask(self, x):
        x_embedding = self.get_embedding_with_noise(x)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit


class FNN(BasicModel):

    def __init__(self, opt):
        super(FNN, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn = MultiLayerPerceptron(
            self.field_num * self.latent_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn
        )

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        logit = self.dnn(x_dnn)
        return logit

    def forward_with_mask(self, x, index):
        x_embedding = self.get_embedding_with_mask(x, index)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        logit = self.dnn(x_dnn)
        return logit

    def forward_with_random_mask(self, x):
        x_embedding = self.get_embedding_with_noise(x)
        x_dnn = x_embedding.view(-1, self.field_num * self.latent_dim)
        logit = self.dnn(x_dnn)
        return logit


class Wukong(BasicModel):

    def __init__(self, opt):
        super(Wukong, self).__init__(opt)
        self.num_emb_lcb = opt['num_embed_lcb']
        self.num_emb_fmb = opt['num_embed_fmb']
        rank_fmb = opt['rank_fmb']
        fm_mlp_dims = opt['fm_mlp_dims']
        mlp_dims = opt['mlp_dims']
        use_bn = opt['use_bn']
        dropout = opt['mlp_dropout']

        num_embed_in = self.field_num
        self.interaction_layers = nn.Sequential()
        for _ in range(opt['num_layers']):
            self.interaction_layers.append(
                WukongLayer(
                    num_embed_in, self.latent_dim, self.num_emb_lcb, self.num_emb_fmb, rank_fmb, fm_mlp_dims, dropout
                )
            )
            num_embed_in = self.num_emb_lcb + self.num_emb_fmb

        self.mlp = MultiLayerPerceptron(
            (self.num_emb_lcb + self.num_emb_fmb) * self.latent_dim,
            mlp_dims,
            output_layer=True,
            dropout=dropout,
            use_bn=use_bn
        )

    def forward(self, x):
        x_embedding = self.embedding(x)
        outputs = self.interaction_layers(x_embedding)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb) * self.latent_dim)
        outputs = self.mlp(outputs)
        return outputs


class RankMixer(BasicModel):

    def __init__(self, opt):
        super(RankMixer, self).__init__(opt)
        num_L = opt['num_L']
        expansion_rate = opt['expansion_rate']
        self.rank_mixer_layers = nn.Sequential()
        for _ in range(num_L):
            self.rank_mixer_layers.append(
                layer.RankMixerLayer(self.field_num, self.latent_dim, self.field_num, expansion_rate)
            )

        mlp_dims = opt['mlp_dims']
        use_bn = opt['use_bn']
        dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(
            self.field_num * self.latent_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn
        )

    def forward(self, x):
        x_embedding = self.embedding(x)
        outputs = self.rank_mixer_layers(x_embedding)
        outputs = outputs.view(-1, self.field_num * self.latent_dim)
        outputs = self.mlp(outputs)
        return outputs


class FinalMLP(BasicModel):

    def __init__(self, opt):
        super(FinalMLP, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.mlp_1 = MultiLayerPerceptron(
            self.field_num * self.latent_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn
        )
        self.mlp_2 = MultiLayerPerceptron(
            self.field_num * self.latent_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn
        )
        last_mlp_dim = mlp_dims[-1]
        self.fusion = InteractionAggregrationLayer(last_mlp_dim, last_mlp_dim)

    def forward(self, x):
        x_embedding = self.embedding(x)
        flattened_embedding = x_embedding.view(-1, self.field_num * self.latent_dim)
        x_1 = self.mlp_1(flattened_embedding)
        x_2 = self.mlp_2(flattened_embedding)
        x_dnn = self.fusion(x_1, x_2)
        return x_dnn


def build_model(model: str, opt):
    name = model.lower()
    if name == "fm":
        return FM(opt)
    elif name == "deepfm":
        return DeepFM(opt)
    elif name == "ipnn":
        return InnerProductNet(opt)
    elif name == "dcn":
        return DeepCrossNet(opt)
    elif name == "dcnv2":
        return DeepCrossNetV2(opt)
    elif name == "fnn":
        return FNN(opt)
    elif name == "wukong":
        return Wukong(opt)
    elif name == "rankmixer":
        return RankMixer(opt)
    elif name == "finalmlp":
        return FinalMLP(opt)
    else:
        raise ValueError(f"Invalid model type: {model}")
