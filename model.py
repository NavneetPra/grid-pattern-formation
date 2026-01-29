# -*- coding: utf-8 -*-
from einops import rearrange
import torch
import torch.nn.functional as F

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Input weights
        if str(options.device) == 'mps':
            self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False, dtype=torch.float32)
            self.RNN = torch.nn.RNN(input_size=2,
                                    hidden_size=self.Ng,
                                    nonlinearity=options.activation,
                                    bias=False,
                                    dtype=torch.float32)
            self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False, dtype=torch.float32)
        else:
            self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
            self.RNN = torch.nn.RNN(input_size=2,
                                    hidden_size=self.Ng,
                                    nonlinearity=options.activation,
                                    bias=False)
            self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        yhat = torch.clamp(self.softmax(self.predict(inputs)), min=1e-8, max=1.0)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err
    
    def compute_loss_topo(self, inputs, pc_outputs, pos, smooth_lambda, distance_lambda):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        yhat = torch.clamp(self.softmax(self.predict(inputs)), min=1e-8, max=1.0)
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        loss += combined_biological_topo_loss(
            weight_matrix=self.RNN.weight_hh_l0,
            grid_height=64,
            grid_width=64,
            lambda_smoothness=smooth_lambda,
            lambda_distance=distance_lambda,
        )

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

def combined_biological_topo_loss(
    weight_matrix: torch.Tensor,
    grid_height: int,
    grid_width: int,
    factor_h: float = 4.0,
    factor_w: float = 4.0,
    lambda_smoothness: float = 1.0,
    lambda_distance: float = 1.0,
):
    """
    Combine TopoLoss smoothness with distance penalty.

    Smoothness: Neighboring neurons have similar connectivity patterns
    Distance: Neurons preferentially connect to nearby neighbors
    """
    n_neurons = weight_matrix.shape[0]
    assert n_neurons == grid_height * grid_width

    # Topoloss smoothness
    cortical_sheet = weight_matrix.reshape(grid_height, grid_width, n_neurons)
    grid = rearrange(cortical_sheet, "h w e -> e h w").unsqueeze(0)

    # Blur operation
    downscaled = F.interpolate(
        grid,
        scale_factor=(1 / factor_h, 1 / factor_w),
        mode='bilinear',
        align_corners=False
    )
    upscaled = F.interpolate(
        downscaled,
        size=grid.shape[2:],
        mode='bilinear',
        align_corners=False
    )

    # Smoothness loss: neighboring neurons should have similar connectivity
    grid_flat = rearrange(grid.squeeze(0), "e h w -> (h w) e")
    upscaled_flat = rearrange(upscaled.squeeze(0), "e h w -> (h w) e")
    smoothness_loss = 1 - F.cosine_similarity(grid_flat, upscaled_flat, dim=-1).mean()

    # Distance penalty
    # y_coords, x_coords = torch.meshgrid(
    #     torch.arange(grid_height, device=weight_matrix.device),
    #     torch.arange(grid_width, device=weight_matrix.device),
    #     indexing='ij'
    # )
    # coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1).float()
    # distances = torch.cdist(coords, coords, p=2)
    # distances = distances / distances.max()
    # connection_strength = torch.abs(weight_matrix)
    # wiring_cost = (connection_strength * distances).sum()
    # total_strength = connection_strength.sum() + 1e-8
    # distance_loss = wiring_cost / total_strength

    # Combined loss
    total_topo_loss = lambda_smoothness * smoothness_loss  # + lambda_distance * distance_loss
    return total_topo_loss
