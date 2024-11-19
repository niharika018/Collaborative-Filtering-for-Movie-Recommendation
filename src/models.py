import torch
import torch.nn as nn


class MatrixFactorizationRMSEModel(nn.Module):
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationRMSEModel, self).__init__()

        # MemoryEmbed
        self.user_memory = nn.Embedding(user_count, embed_size)
        self.user_memory.weight.data.uniform_(-0.005, 0.005)

        # ItemMemory
        self.item_memory = nn.Embedding(item_count, embed_size)
        self.item_memory.weight.data.uniform_(-0.005, 0.005)

    def _forward(self, userids, itemids):
        # [batch, embedding size]
        user_vec = self.user_memory(userids)

        # [batch, embedding size]
        item_vec = self.item_memory(itemids)

        pred_r = user_vec * item_vec

        return torch.sum(pred_r, dim=1)

    def criterion(self, batch, pred_r):
        """
        Calculate RMSE loss
        """
        ratings = batch[:, 2]
        return torch.sqrt(torch.mean((ratings - pred_r) ** 2))

    def forward(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)

    def run_eval(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)


class MatrixFactorizationBPRModel(nn.Module):
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationBPRModel, self).__init__()

        self.basemodel = MatrixFactorizationRMSEModel(
            user_count, item_count, embed_size
        )
        self.sig = nn.Sigmoid()

    def _forward(self, userids, pos_itemids, neg_itemids):
        pos_r = self.basemodel._forward(userids, pos_itemids)
        neg_r = self.basemodel._forward(userids, neg_itemids)

        diff = pos_r - neg_r

        return diff

    def criterion(self, _, vals):
        """
        Calculate BPR loss
        """
        return (1.0 - self.sig(vals)).pow(2).sum()

    def forward(self, batch):
        userids = batch[:, 0]
        pos_itemids = batch[:, 1]
        neg_itemids = batch[:, 2]

        return self._forward(userids, pos_itemids, neg_itemids)

    def run_eval(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self.basemodel._forward(userids, itemids)


class GMFBCEModel(MatrixFactorizationRMSEModel):
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationRMSEModel, self).__init__()

        self.user_memory = nn.Embedding(user_count, embed_size)
        self.item_memory = nn.Embedding(item_count, embed_size)
        self.output_layer = nn.Linear(embed_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.bce = nn.BCELoss()

        self.initialize_weights()

    def initialize_weights(self):
        for param in self.state_dict():
            self.state_dict()[param].uniform_(-0.01, 0.01)

    def _forward(self, userids, itemids):
        user_vec = self.user_memory(userids)
        item_vec = self.item_memory(itemids)

        return self.sigmoid(self.output_layer(user_vec * item_vec)).squeeze()

    def criterion(self, batch, pred_r):
        """
        Calculate BCE loss
        """
        ratings = batch[:, 2].float()
        return self.bce(pred_r, ratings)


class MLPBCEModel(MatrixFactorizationRMSEModel):
    def __init__(self, user_count, item_count, embed_size=40, layers=[20, 10]):
        super(MatrixFactorizationRMSEModel, self).__init__()

        self.user_memory = nn.Embedding(user_count, embed_size)
        self.item_memory = nn.Embedding(item_count, embed_size)

        self.mlp_layers = []
        in_size = 2 * embed_size
        for out_size in layers:
            self.mlp_layers.append(nn.Linear(in_size, out_size))
            self.mlp_layers.append(nn.ReLU())
            in_size = out_size
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
        self.output_layer = nn.Linear(layers[-1], 1)

        self.sigmoid = nn.Sigmoid()

        self.bce = nn.BCELoss()

        # self.initialize_weights()

    def initialize_weights(self):
        for param in self.state_dict():
            self.state_dict()[param].uniform_(-0.01, 0.01)

    def _forward(self, userids, itemids):
        user_vec = self.user_memory(userids)
        item_vec = self.item_memory(itemids)

        return self.sigmoid(
            self.output_layer(
                self.mlp_layers(torch.concat((user_vec, item_vec), dim=1))
            )
        ).squeeze()

    def criterion(self, batch, pred_r):
        """
        Calculate BCE loss
        """
        ratings = batch[:, 2].float()
        return self.bce(pred_r, ratings)


class NeuralMatrixFactorizationBCEModel(nn.Module):
    def __init__(
        self,
        user_count,
        item_count,
        gmf_embed_size=40,
        mlp_embed_size=40,
        layers=[20, 10],
        alpha=0.5,
    ):
        super(NeuralMatrixFactorizationBCEModel, self).__init__()

        self.alpha = alpha

        self.gmf_user_memory = nn.Embedding(user_count, gmf_embed_size)
        self.gmf_item_memory = nn.Embedding(item_count, gmf_embed_size)

        self.mlp_user_memory = nn.Embedding(user_count, mlp_embed_size)
        self.mlp_item_memory = nn.Embedding(item_count, mlp_embed_size)

        self.mlp_layers = []
        in_size = 2 * mlp_embed_size
        for out_size in layers:
            self.mlp_layers.append(nn.Linear(in_size, out_size))
            self.mlp_layers.append(nn.ReLU())
            in_size = out_size
        self.mlp_layers = nn.Sequential(*self.mlp_layers)

        self.neumf_layer = nn.Linear(gmf_embed_size + layers[-1], 1)

        self.initialize_weights()

        self.sigmoid = nn.Sigmoid()

        self.bce = nn.BCELoss()

    def initialize_weights(self):
        for param in self.state_dict():
            self.state_dict()[param].uniform_(-0.01, 0.01)

    def load_pretrained_weights(self, gmf_model, mlp_model):
        with torch.no_grad():
            self.gmf_user_memory.weight.copy_(gmf_model.user_memory.weight)
            self.gmf_item_memory.weight.copy_(gmf_model.item_memory.weight)

            self.mlp_user_memory.weight.copy_(mlp_model.user_memory.weight)
            self.mlp_item_memory.weight.copy_(mlp_model.item_memory.weight)

            for l_name in self.mlp_layers.state_dict().keys():
                self.mlp_layers.state_dict()[l_name].copy_(
                    mlp_model.mlp_layers.state_dict()[l_name]
                )

            neumf_weights = torch.concat(
                [
                    self.alpha * gmf_model.output_layer.weight,
                    (1 - self.alpha) * mlp_model.output_layer.weight,
                ],
                dim=1,
            )
            neumf_bias = self.alpha * gmf_model.output_layer.bias
            neumf_bias += (1 - self.alpha) * mlp_model.output_layer.bias

            self.neumf_layer.weight.copy_(neumf_weights)

    def _forward(self, userids, itemids):
        gmf_user_vec = self.gmf_user_memory(userids)
        gmf_item_vec = self.gmf_item_memory(itemids)

        mlp_user_vec = self.mlp_user_memory(userids)
        mlp_item_vec = self.mlp_item_memory(itemids)

        gmf_intermediate = gmf_user_vec * gmf_item_vec
        mlp_intermediate = self.mlp_layers(
            torch.concat([mlp_user_vec, mlp_item_vec], dim=1)
        )

        neumf_input = torch.concat([gmf_intermediate, mlp_intermediate], dim=1)
        neumf_intermediate = self.neumf_layer(neumf_input)

        return self.sigmoid(neumf_intermediate).squeeze()

    def criterion(self, batch, pred_r):
        """
        Calculate BCE loss
        """
        ratings = batch[:, 2].float()
        return self.bce(pred_r, ratings)

    def forward(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)

    def run_eval(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)


class GMFBPRModel(MatrixFactorizationBPRModel):
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationBPRModel, self).__init__()

        self.basemodel = GMFBCEModel(user_count, item_count, embed_size)

        self.sig = nn.Sigmoid()


class MLPBPRModel(MatrixFactorizationBPRModel):
    def __init__(self, user_count, item_count, embed_size=40, layers=[20, 10]):
        super(MatrixFactorizationBPRModel, self).__init__()

        self.basemodel = MLPBCEModel(user_count, item_count, embed_size, layers)

        self.sig = nn.Sigmoid()

    def initialize_weights(self):
        self.basemodel.initialize_weights()


class NeuralMatrixFactorizationBPRModel(MatrixFactorizationBPRModel):
    def __init__(
        self,
        user_count,
        item_count,
        gmf_embed_size=40,
        mlp_embed_size=40,
        layers=[20, 10],
        alpha=0.5,
    ):
        super(MatrixFactorizationBPRModel, self).__init__()

        self.basemodel = NeuralMatrixFactorizationBCEModel(
            user_count,
            item_count,
            gmf_embed_size=gmf_embed_size,
            mlp_embed_size=mlp_embed_size,
            layers=layers,
            alpha=alpha,
        )

        self.sig = nn.Sigmoid()

    def initialize_weights(self):
        self.basemodel.initialize_weights()

    def load_pretrained_weights(self, gmf_model, mlp_model):
        self.basemodel.load_pretrained_weights(gmf_model, mlp_model)
