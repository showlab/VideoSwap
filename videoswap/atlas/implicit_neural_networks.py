import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class IMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=256,
            use_positional=True,
            positional_dim=10,
            skip_layers=[4, 6],
            num_layers=8,  # includes the output layer
            verbose=True,
            use_tanh=True,
            apply_softmax=False):
        super(IMLP, self).__init__()
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)], requires_grad=False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f'Model has {count_parameters(self)} params')

    def forward(self, x):
        # print(x.shape) # torch.Size([99676, 3])

        if self.use_positional:
            if self.b.device != x.device:
                self.b = self.b.to(x.device)
            # print(x.shape) # torch.Size([99676, 3])
            pos = positionalEncoding_vec(x, self.b)
            x = pos
            # print(x.shape) # torch.Size([99676, 30])

        # nerf mlp

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)

        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x.to(torch.float32)


class IMLP_Hash(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        pe_type='none',  # ['none', 'hash_encoding', 'encoding']
        pe_dim=10,
        mlp_type='origin',  # ['origin', 'tcnn']
        skip_layers=[],
        mlp_layers=8,  # includes the output layer
        use_tanh=True,
        fp16=False
    ):
        super(IMLP_Hash, self).__init__()
        self.use_tanh = use_tanh

        self.pe_type = pe_type
        self.mlp_type = mlp_type

        if pe_type == 'hash_encoding':
            self.encoder = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': 16,
                    'n_features_per_level': 2,
                    'log2_hashmap_size': 19,
                    'base_resolution': 16,
                    'per_level_scale': 1.38
                },
                dtype=torch.float16 if fp16 else torch.float
            )
            encoding_dimensions = self.encoder.n_output_dims
        elif pe_type == 'encoding':
            encoding_dimensions = 2 * input_dim * pe_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(pe_dim)], requires_grad=False)
        else:
            encoding_dimensions = input_dim

        if mlp_type == 'origin':
            self.hidden = nn.ModuleList()
            for i in range(mlp_layers):
                if i == 0:
                    input_dims = encoding_dimensions
                elif i in skip_layers:
                    input_dims = hidden_dim + encoding_dimensions
                else:
                    input_dims = hidden_dim

                if i == mlp_layers - 1:
                    # last layer
                    self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
                else:
                    self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

            self.skip_layers = skip_layers
            self.mlp_layers = mlp_layers
        elif mlp_type == 'tcnn':
            self.decoder = tcnn.Network(encoding_dimensions, output_dim, network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Tanh',
                'n_neurons': hidden_dim,
                'n_hidden_layers': mlp_layers,
            })
        else:
            raise NotImplementedError

    def forward(self, x):

        if self.pe_type == 'hash_encoding':
            x = self.encoder(x)
        elif self.pe_type == 'encoding':
            if self.b.device != x.device:
                self.b = self.b.to(x.device)
            pos = positionalEncoding_vec(x, self.b)
            x = pos
        else:
            x = x

        if self.mlp_type == 'origin':
            input = x.detach().clone()
            for i, layer in enumerate(self.hidden):
                if i > 0:
                    x = F.relu(x)
                if i in self.skip_layers:
                    x = torch.cat((x, input), 1)
                x = layer(x)

            if self.use_tanh:
                x = torch.tanh(x)

        elif self.mlp_type == 'tcnn':
            x = self.decoder(x)
        else:
            raise NotImplementedError

        return x
