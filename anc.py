import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
import json
from tqdm import tqdm
import time
import glob
import re 
import shutil
import os
"""
Code Summary: Matchgate Brick Wall QNN, Version 1

This code trains a QNN consisting of a brick wall of matchgates with mid-circuit measurements. All 6 pauli coefficients are included per matchgate. The input activations
are mapped to angles of the 1st layer. Each measurement site has the same probability of being measured. Measurements in the hidden layers are mapped to angles of the 
next layer. The output layer is real and classical. Backpropagation is performed via the effective activation function
                                                             d_back = 2*p(s=+1|previous layer measurements) -1
Only derivatives partial cost/partial activation for measured sites are included. Inference is done by majority vote over num_shots forward passes evaluated at the weights
from training.
"""

"""
Start Testing Functions
-----------------------
"""
@torch.no_grad()
def Deterministic_Brick_Wall_Layer_Forward(old_covariance_matrix,chi, measurement_sites,layer_index, measurement_results):
    """
    Same function as Brick_Wall_Layer_Forward except instead of randomly generating measurement reesults it takes the measurement results
    as input. measurement_results is assumed to be 1d torch tensor length num_qubits with measurement results placed in the appropriate positions.
    The function is only used for testing.
    """
    device = old_covariance_matrix.device
    dtype  = old_covariance_matrix.dtype
    # read off the number of two_qubit gates and angles per gate from chi
    num_gates = chi.shape[0]
    #construct a 3-d tensor for storing the covariance matrices, these will be used in the backwards pass
    covariance_size = old_covariance_matrix.shape[0]
    #construct the antisymmetric matrix that can be read off from the majorana representation of the matchgate hamiltonian
    #block_size is always 4, corresponding to size of individual two-qubit gates applied
    block_size = 4
    majorana_coefficients = torch.zeros(num_gates, block_size, block_size, device=device, dtype=dtype)
    #reminder: gate order is XX, XY, YX, YY, ZI, IZ, define the upper right triangle
    majorana_coefficients[:,0,1] = -0.5*chi[:,4] #ZI
    majorana_coefficients[:,0,2] = 0.5*chi[:,2]  #YX
    majorana_coefficients[:,0,3] = 0.5*chi[:,3]  #YY
    majorana_coefficients[:,1,2] = -0.5*chi[:,0] #XX
    majorana_coefficients[:,1,3] = -0.5*chi[:,1] #XY
    majorana_coefficients[:,2,3] = -0.5*chi[:,5] #IZ
    #define the lower right triangle
    majorana_coefficients -= majorana_coefficients.transpose(1,2).clone()
    #check whether we are on odd or even layer (counting from 1 putting even number two-qubit gates on the odd layer)
    is_odd_layer = (layer_index % 2 == 0)
    #construct the SO(2*num_qubits) update matrix for this layer
    orthogonal_blocks = torch.linalg.matrix_exp(4.0*majorana_coefficients)
    if is_odd_layer:
        update_matrix = torch.block_diag(*orthogonal_blocks)
    else:
        update_matrix = torch.block_diag(torch.eye(2, device=device, dtype=dtype),*orthogonal_blocks,torch.eye(2, device=device, dtype=dtype))
    #update the covariance matrix due to the unitary evolution from a layer matchgate
    covariance_matrix = update_matrix @ old_covariance_matrix @ update_matrix.t()
    #The method where we use expectation value in post-unitary/pre-measurement state requires the post-unitary/pre-measurement
    #and post-measurement covariance matrices be saved
    covariance_matrices = torch.zeros((2, covariance_size,covariance_size), device=device, dtype=dtype)
    #save post-unitary/pre-measurement covariance matrix for backwards pass
    covariance_matrices[0,:,:] = covariance_matrix
    #eps is just a small number used to ensure that we don't divide by zero (probability) in the projection update of the covariance matrix
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    for k in measurement_sites:
        k = int(k)  # ensure plain int for indexing
        #calculate measurement probability from the covariance matrix
        spin_up_probability = 0.5*(1-covariance_matrix[2*k,2*k+1])
        #get rid of small negative numbers that can occur due to finite numeric precision
        spin_up_probability = torch.clamp(spin_up_probability, min=0,max=1)
        #projective pauli-Z measurement, save result
        spin_outcome = measurement_results[k]
        # measurement update: Gamma' = Gamma + (s/(1 - s*Gamma[2k,2k+1])) * (Gamma A Gamma - A)
        col_a = covariance_matrix[:,2*k]
        col_b = covariance_matrix[:,2*k+1]
        temp  = torch.outer(col_a,col_b)-torch.outer(col_b,col_a)   # = Gamma A Gamma
        A = torch.zeros_like(covariance_matrix)
        A[2*k,   2*k+1] =  -1.0
        A[2*k+1, 2*k  ] = 1.0
        denom = 1.0 - spin_outcome*covariance_matrix[2*k,2*k+1]
        denom = torch.sign(denom) * torch.maximum(torch.abs(denom), eps)  # sign-preserving clamp
        covariance_matrix = covariance_matrix + (spin_outcome/denom)*(temp + A)
        # optional hard projection to kill residual drift, matters if you are forcing rare outcomes
        covariance_matrix[:, 2*k]   = 0
        covariance_matrix[2*k, :]   = 0
        covariance_matrix[:, 2*k+1] = 0
        covariance_matrix[2*k+1, :] = 0
        covariance_matrix[2*k, 2*k+1] = -spin_outcome
        covariance_matrix[2*k+1, 2*k] =  spin_outcome
        #explicitely antisymmeterize covariance matrix to mitigate numeric errors
        covariance_matrix = 0.5*(covariance_matrix - covariance_matrix.t())
    #save the post-measurement covariance matrix for next layer in forward pass
    covariance_matrices[1,:,:] = covariance_matrix
    #return the updated covariance matrix and the measurement results
    return covariance_matrices,update_matrix

class _StepCapture:
    """
    Captures activations from TOP-LEVEL modules in forward order (e.g., your nn.Sequential).
    For outputs that are tuples/lists (Brick_Wall, Brick_Wall_Last), we take the SECOND tensor.
    For others (e.g., Linear), we take the first tensor found.
    Saves ONE file per forward "step": step_000000.pt containing [t_layer0, t_layer1, ...].
    Only records when module.training == True (skips eval passes inside get_err()).
    """
    def __init__(self, model: nn.Module, save_dir: str, clear: bool = True, capture_eval: bool = False):
        self.model = model
        self.save_dir = save_dir
        self.capture_training_only = not capture_eval

        if clear:
            shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)

        # top-level children in forward order
        self.children = [m for m in model] if isinstance(model, nn.Sequential) else list(model.children())
        self.n = len(self.children)
        self._buf = [None] * self.n
        self._step = 0
        self._handles = []

        # cache layer names (optional; useful for debugging)
        self.layer_names = [f"layer_{i}_{m.__class__.__name__}" for i, m in enumerate(self.children)]
        torch.save(self.layer_names, os.path.join(save_dir, "layers.pt"))

        # register hooks
        for i, m in enumerate(self.children):
            self._handles.append(m.register_forward_hook(self._make_hook(i)))

    # ---- hook helpers ----
    def _first_tensor(self, x):
        if torch.is_tensor(x): return x
        if isinstance(x, (list, tuple)):
            for o in x:
                t = self._first_tensor(o)
                if t is not None: return t
        if isinstance(x, dict):
            for v in x.values():
                t = self._first_tensor(v)
                if t is not None: return t
        return None

    def _pick_activation(self, out):
        # Prefer the SECOND element for tuple/list outputs (your hidden-layer activations)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            y = out[1]
            if torch.is_tensor(y): return y
            t = self._first_tensor(y)
            if t is not None: return t
        # Fallback: first tensor anywhere
        return self._first_tensor(out)

    def _make_hook(self, idx):
        last = self.n - 1
        def hook(module, inputs, output):
            # skip eval forwards unless explicitly requested
            if self.capture_training_only and not module.training:
                return
            t = self._pick_activation(output)
            if t is None:
                return
            self._buf[idx] = t.detach().flatten().cpu()

            # finalize a step when the LAST top-level module fires
            if idx == last:
                if any(b is None for b in self._buf):
                    # likely an eval forward was skipped; reset partials
                    self._buf = [None] * self.n
                    return
                step_list = self._buf
                torch.save(step_list, os.path.join(self.save_dir, f"step_{self._step:06d}.pt"))
                self._step += 1
                self._buf = [None] * self.n
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

# ---- public API ----

def enable_step_activation_capture(model: nn.Module, save_dir: str = "training_logs",
                                   clear: bool = True, capture_eval: bool = False) -> _StepCapture:
    """
    One-liner #1: call this BEFORE train(...).
    Attaches hooks and starts writing per-step activations into save_dir/step_*.pt.
    Returns a handle you can ignore or .close() after training.
    """
    return _StepCapture(model, save_dir, clear=clear, capture_eval=capture_eval)

def load_step_activations(save_dir: str = "training_logs") -> list[list[torch.Tensor]]:
    """
    One-liner #2: call this AFTER train(...).
    Loads all step files into a list-of-lists:
      all_activations[step] = [act_layer0, act_layer1, ...]
    Every act_layer* is a 1-D torch.Tensor (flattened).
    """
    paths = sorted(glob.glob(os.path.join(save_dir, "step_*.pt")))
    return [torch.load(p, map_location="cpu") for p in paths]



"""
End Testing Functions
---------------------
"""

#Function that randomly chooses measurement sites. Each site has equal probability of being measured
@torch.no_grad()
def Generate_Measurement_Sites(num_qubits, measurement_probability) -> torch.Tensor:
    # create a mask of size num_qubits where each position is 1 with probability measurement_probability
    mask = torch.bernoulli(torch.full((num_qubits,),measurement_probability)).bool()
    # in the unlikely event none of the sites are measured, randomly choose one to be measured. we need at least one
    if not mask.any():
        random_integer = torch.randint(num_qubits,(1,))
        mask[random_integer] = True
    # return a torch tensor of integers corresponding to measurement site locations for this layer
    return torch.arange(num_qubits)[mask]

@torch.no_grad()
def Brick_Wall_Layer_Forward(old_covariance_matrix,chi, measurement_sites,layer_index):
    """
    Simulates one layer of matchgate brickwall circuit. The method involves JW transformation and updating covariance matrices
    in the majorana basis. The Unitary of a brickwall layer corresponds to applying an SO(2xnum_qubits) matrix. There is also
    an update rule that makes the corresponding projection in the covariance matrix and normalization.

    Parameters
    ----------
    old_covariance_matrix: torch.Tensor 
        covariance matrix of state coming into the layer
    chi: torch.Tensor 
        2-qubit angles for the layer (6, num_gates). Assumes order XX,XY,YX,YY,ZI,IZ
    measurement_sites: torch.Tensor 
        list of measurement sites for layer (0 to num_qubits-1)
    layer_index: int
        index of layer (counting from 0)

    Returns
    -------
        covariance_matrices: torch.Tensor 
            3d torch.Tensor (2,2*num_qubits,2*num_qubits). It is covariance matrix from post-unitary/pre-measurement(for backpass)
            and post-measurement (to continue forward pass)
        update_matrix: torch.Tensor 
            The SO(2*num_qubits) update matrix. It is saved for the backpass
        measurement_results: torch.Tensor 
            tensor with measurement results for each measurement site +-1
    """
    device = old_covariance_matrix.device
    dtype  = old_covariance_matrix.dtype
    # read off the number of two_qubit gates and angles per gate from chi
    num_gates = chi.shape[0]
    #construct a 3-d tensor for storing the covariance matrices, these will be used in the backwards pass
    covariance_size = old_covariance_matrix.shape[0]
    #construct the antisymmetric matrix that can be read off from the majorana representation of the matchgate hamiltonian
    #block_size is always 4, corresponding to size of individual two-qubit gates applied
    block_size = 4
    majorana_coefficients = torch.zeros(num_gates, block_size, block_size, device=device, dtype=dtype)
    #reminder: gate order is XX, XY, YX, YY, ZI, IZ, define the upper right triangle
    majorana_coefficients[:,0,1] = -0.5*chi[:,4] #ZI
    majorana_coefficients[:,0,2] = 0.5*chi[:,2]  #YX
    majorana_coefficients[:,0,3] = 0.5*chi[:,3]  #YY
    majorana_coefficients[:,1,2] = -0.5*chi[:,0] #XX
    majorana_coefficients[:,1,3] = -0.5*chi[:,1] #XY
    majorana_coefficients[:,2,3] = -0.5*chi[:,5] #IZ
    #define the lower right triangle
    majorana_coefficients -= majorana_coefficients.transpose(1,2).clone()
    #check whether we are on odd or even layer (counting from 1 putting even number two-qubit gates on the odd layer)
    is_odd_layer = (layer_index % 2 == 0)
    #construct the SO(2*num_qubits) update matrix for this layer
    orthogonal_blocks = torch.linalg.matrix_exp(4.0*majorana_coefficients)
    if is_odd_layer:
        update_matrix = torch.block_diag(*orthogonal_blocks)
    else:
        update_matrix = torch.block_diag(torch.eye(2, device=device, dtype=dtype),*orthogonal_blocks,torch.eye(2, device=device, dtype=dtype))
    #update the covariance matrix due to the unitary evolution from a layer matchgate
    covariance_matrix = update_matrix @ old_covariance_matrix @ update_matrix.t()
    #The method where we use expectation value in post-unitary/pre-measurement state requires the post-unitary/pre-measurement
    #and post-measurement covariance matrices be saved
    covariance_matrices = torch.zeros((2, covariance_size,covariance_size), device=device, dtype=dtype)
    #save post-unitary/pre-measurement covariance matrix for backwards pass
    covariance_matrices[0,:,:] = covariance_matrix
    #simulate the mid-circuit measurements
    measurement_results = []
    #eps is just a small number used to ensure that we don't divide by zero (probability) in the projection update of the covariance matrix
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    for k in measurement_sites:
        k = int(k)  # ensure plain int for indexing
        #calculate measurement probability from the covariance matrix
        spin_up_probability = 0.5*(1-covariance_matrix[2*k,2*k+1])
        #get rid of small negative numbers that can occur due to finite numeric precision
        spin_up_probability = torch.clamp(spin_up_probability, min=0,max=1)
        #projective pauli-Z measurement, save result
        spin_outcome = 2*torch.bernoulli(spin_up_probability)-1
        #deterministic version commented out, can be used for testing
        measurement_results.append(spin_outcome)
        # measurement update: Gamma' = Gamma + (s/(1 - s*Gamma[2k,2k+1])) * (Gamma A Gamma - A)
        col_a = covariance_matrix[:,2*k]
        col_b = covariance_matrix[:,2*k+1]
        temp  = torch.outer(col_a,col_b)-torch.outer(col_b,col_a)   # = Gamma A Gamma
        A = torch.zeros_like(covariance_matrix)
        A[2*k,   2*k+1] =  -1.0
        A[2*k+1, 2*k  ] = 1.0
        denom = 1.0 - spin_outcome*covariance_matrix[2*k,2*k+1]
        denom = torch.sign(denom) * torch.maximum(torch.abs(denom), eps)  # sign-preserving clamp
        covariance_matrix = covariance_matrix + (spin_outcome/denom)*(temp + A)
        # optional hard projection to kill residual drift, matters if you are forcing rare outcomes
        covariance_matrix[:, 2*k]   = 0
        covariance_matrix[2*k, :]   = 0
        covariance_matrix[:, 2*k+1] = 0
        covariance_matrix[2*k+1, :] = 0
        covariance_matrix[2*k, 2*k+1] = -spin_outcome
        covariance_matrix[2*k+1, 2*k] =  spin_outcome
        #explicitely antisymmeterize covariance matrix to mitigate numeric errors
        covariance_matrix = 0.5*(covariance_matrix - covariance_matrix.t())
    #save the post-measurement covariance matrix for next layer in forward pass
    covariance_matrices[1,:,:] = covariance_matrix
    #return the updated covariance matrix and the measurement results
    return covariance_matrices,update_matrix, torch.stack(measurement_results).to(device=device, dtype=dtype)

@torch.no_grad()
def diag_blocks(A: torch.Tensor) -> torch.Tensor:
    """
    Extract all non-overlapping 4x4 diagonal blocks from a square (2N x 2N) matrix A.
    Returns a tensor of shape (N/2, 4, 4) with the diagonal index as the FIRST dimension.
    Uses a zero-copy as_strided view (requires A to be contiguous).
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    S = A.shape[0]
    assert S % 4 == 0, "Side length must be divisible by 4"
    k = S // 4
    A = A.contiguous()
    return A.as_strided(size=(k, 4, 4),
                        stride=(4 * S + 4, S, 1))

@torch.no_grad()
def middle_diag_blocks(B: torch.Tensor) -> torch.Tensor:
    """
    Leave the 2x2 blocks in the corners of a (2N x 2N) matrix B,
    and extract the (N/2 - 1) 4x4 diagonal blocks from the middle.
    Returns a tensor of shape (N/2 - 1, 4, 4) with the diagonal index first.
    """
    assert B.ndim == 2 and B.shape[0] == B.shape[1], "B must be square"
    S = B.shape[0]
    assert (S - 4) % 4 == 0 and S >= 8, "Side length must be >= 8 and S-4 divisible by 4"
    inner = B[2:-2, 2:-2].contiguous()
    return diag_blocks(inner)

@torch.no_grad()
def Brick_Wall_Layer_Back(partial_cost_partial_activation,
                          covariance_matrix,
                          update_matrix,
                          chi,
                          measurement_sites,
                          measurement_results,
                          layer_index):
    """
    This function treats measurements like activations in the neural network. It calculates partial cost/ partial chi from partial cost/partial activation
    using the following effective activation function on the backpass
                                                      d_back = 2*p(s=+1|previous layer measurements)-1
    Parameters
    ----------
        partial_cost_partial_activation: torch.Tensor 
            From previous layer
        covariance_matrix: torch.Tensor 
            covariance matrix from post-unitary/pre-measurement state
        update_matrix: torch.Tensor 
            The SO(2*num_qubits) update matrix
        chi: torch.Tensor 
            2-qubit angles for the layer (6, num_gates). Assumes order XX,XY,YX,YY,ZI,IZ
        measurement_sites: torch.Tensor 
            list of measurement sites for layer (0 to num_qubits-1)
        measurement_results: torch.Tensor
            tensor with measurement results for each measurement site +-1
        layer_index: int 
            index of layer (counting from 0)
    Outputs:
        partial_cost_partial_chi: list 
            list of 6 tensors with the derivative of the cost function with respect to each chi in the layer
    """
    device = covariance_matrix.device
    dtype  = covariance_matrix.dtype
    # check whether we have N/2 or N/2-1 two-qubit gates in this layer
    num_qubits = covariance_matrix.shape[0] // 2
    even_layer = (layer_index % 2 == 0)
    if even_layer:
        covariance_blocks = diag_blocks(covariance_matrix)
        orthogonal_blocks = diag_blocks(update_matrix)
    else:
        covariance_blocks = middle_diag_blocks(covariance_matrix)
        orthogonal_blocks = middle_diag_blocks(update_matrix)
    num_gates = covariance_blocks.shape[0]  # robust for both parities
    # Define antisymmetric majorana_coefficients as 3d tensor (num_gatesx4x4)
    block_size = 4
    majorana_coefficients = torch.zeros(num_gates, block_size, block_size, device=device, dtype=dtype)
    # gate order: XX, XY, YX, YY, ZI, IZ (fill upper triangle)
    majorana_coefficients[:,0,1] = -0.5*chi[:,4] # ZI
    majorana_coefficients[:,0,2] =  0.5*chi[:,2] # YX
    majorana_coefficients[:,0,3] =  0.5*chi[:,3] # YY
    majorana_coefficients[:,1,2] = -0.5*chi[:,0] # XX
    majorana_coefficients[:,1,3] = -0.5*chi[:,1] # XY
    majorana_coefficients[:,2,3] = -0.5*chi[:,5] # IZ
    majorana_coefficients = majorana_coefficients - majorana_coefficients.transpose(1, 2)
    # Define partial h/partial chi where h is the antisymmetric matrix defined by mapping the qubit hamiltonian to majorana basis
    num_matchgates = 6
    partial_h_partial_chi = torch.zeros(num_matchgates, block_size, block_size, device=device, dtype=dtype)
    # XX
    partial_h_partial_chi[0, 1, 2] = -0.5; partial_h_partial_chi[0, 2, 1] =  0.5
    # XY
    partial_h_partial_chi[1, 1, 3] = -0.5; partial_h_partial_chi[1, 3, 1] =  0.5
    # YX
    partial_h_partial_chi[2, 0, 2] =  0.5; partial_h_partial_chi[2, 2, 0] = -0.5
    # YY
    partial_h_partial_chi[3, 0, 3] =  0.5; partial_h_partial_chi[3, 3, 0] = -0.5
    # ZI
    partial_h_partial_chi[4, 0, 1] = -0.5; partial_h_partial_chi[4, 1, 0] =  0.5
    # IZ
    partial_h_partial_chi[5, 2, 3] = -0.5; partial_h_partial_chi[5, 3, 2] =  0.5
    #Calculating the derivative of the orthogonal 2-qubit matrices with respect to each chi
    #This can be viewed as computing the Frechet derivative along direction 4*partial_h_partial_chi
    #The "block-matrix trick" is used which involves calculating fretchet derivative by constructing a certain larger matrix (8x8 here) from exponent
    #and perturbation direction and exponentiating it. Then the Frenchet derivative can be extracted from the upper right hand block
    exponent_blocks = 4.0 * majorana_coefficients                              #dimension (num_gates,4,4)
    exponent_tiled  = exponent_blocks.unsqueeze(0).expand(6, -1, -1, -1)       # repeat data to get dimension (6,num_gates,4,4)
    direction_blocks = 4.0 * partial_h_partial_chi.unsqueeze(1).expand(-1, num_gates, -1, -1)  # dimension (6,num_gates,4,4)
    zeros = torch.zeros_like(exponent_tiled)                                #dimension (6,num_gates,4,4)
    top = torch.cat([exponent_tiled, direction_blocks], dim=-1)                   # (6,num_gates,4,8)
    bottom = torch.cat([zeros,   exponent_tiled ], dim=-1)                        # (6,num_gates,4,8)
    block_trick_exponent   = torch.cat([top, bottom], dim=-2)                     # (6,num_gates,8,8)
    block_trick_matrix = torch.linalg.matrix_exp(block_trick_exponent)            # (6,num_gates,8,8)
    partial_R_partial_chi = block_trick_matrix[..., 0:4, 4:8]                     # (6,num_gates,4,4) == Dexp(exponent)[4*partial_h_partial_chi]
    # Multiply by R^T
    R = orthogonal_blocks                                                # (num_gates,4,4)
    F = partial_R_partial_chi @ R.transpose(-1, -2)                      # (6,num_gates,4,4)
    # Commutators [F, covariance_blocks]
    cov_blocks = covariance_blocks.unsqueeze(0)                          # (1,num_gates,4,4)
    commutators = F @ cov_blocks - cov_blocks @ F                        # (6,num_gates,4,4)
    # gate indices and reduction to your two entries
    if even_layer:
        gates_index = torch.arange(num_qubits // 2, device=device)
    else:
        gates_index = torch.arange(num_qubits // 2 - 1, device=device)
    partial_activation_partial_chi = torch.stack([
        commutators[..., 1, 0],  # (6,num_gates)
        commutators[..., 3, 2]   # (6,num_gates)
    ], dim=1)  # (6,2,num_gates)
    partials = (
        partial_activation_partial_chi[:, 0, :] * partial_cost_partial_activation[2 * gates_index] +
        partial_activation_partial_chi[:, 1, :] * partial_cost_partial_activation[2 * gates_index + 1]
    )  # (6,num_gates)
    # Unpack in the required order
    partial_cost_partial_chi_xx = partials[0]
    partial_cost_partial_chi_xy = partials[1]
    partial_cost_partial_chi_yx = partials[2]
    partial_cost_partial_chi_yy = partials[3]
    partial_cost_partial_chi_zi = partials[4]
    partial_cost_partial_chi_iz = partials[5]
    partial_cost_partial_chi_list = [
        partial_cost_partial_chi_xx,
        partial_cost_partial_chi_xy,
        partial_cost_partial_chi_yx,
        partial_cost_partial_chi_yy,
        partial_cost_partial_chi_zi,
        partial_cost_partial_chi_iz
    ]
    return partial_cost_partial_chi_list

#htanh(x/a)
@torch.no_grad()
def phia(input: torch.Tensor, a: float):
    return torch.clamp(input / a, min=-1.0, max=1.0)

#d/dx htanh(x/a)
@torch.no_grad()
def phia_prime(input:torch.Tensor, a:float):
    alist = torch.ones_like(input)*a
    return torch.logical_and(input>-alist,input<alist).float()*(1/a)

#mapping chi(input)
@torch.no_grad()
def phi_chi(input:torch.Tensor, a:float):
    return torch.pi/4*(torch.ones_like(input)-phia(input,a))

#derivative of mapping chi(input)
@torch.no_grad()
def phi_chi_prime(input:torch.Tensor, a:float):
    return -torch.pi/4*phia_prime(input,a)

#Standard drop out function. Its purpose is to make it so partial cost/partial activation only includes sites that were measured
def dropout_keep(activations_in: torch.Tensor,
                 keep_idx: torch.Tensor,
                 dim: int = -1) -> torch.Tensor:
    """
    Drop all features except those at keep_idx.

    Parameters
    ----------
    activations_in : torch.Tensor
        Input activations (any shape, but must have a feature dimension).
    keep_idx : torch.Tensor
        1-D LongTensor of feature indices to KEEP.
    dim : int, default=-1
        The dimension along which features lie.

    Returns
    -------
    torch.Tensor
        activations with the un-kept features zeroed and scaled so the
        expected value is unchanged.
    """
    feature_size = activations_in.size(dim)
    if keep_idx.numel() == 0:
        raise ValueError("keep_idx must contain at least one index")
    if keep_idx.max().item() >= feature_size:
        raise IndexError(
            f"keep_idx contains out-of-range indices for dim of size {feature_size}"
        )
    # 1. 1-D mask of 0/1 for features
    mask1d = torch.zeros(feature_size, device=activations_in.device,
                         dtype=activations_in.dtype)
    mask1d[keep_idx] = 1.0
    # 2. keep probability and broadcastable mask
    keep_prob = mask1d.mean().clamp(min=1e-12)      # avoid divide-by-zero
    view_shape = [1] * activations_in.ndim
    view_shape[dim] = feature_size
    mask = mask1d.view(*view_shape)
    # 3. Apply mask. I commented out scaled version which I don't think is desirable here, but can be used.
    #x_dropped = activations_in * mask / keep_prob
    x_dropped = activations_in * mask
    return x_dropped


#run this function for only one image at a time. That way you don't hold more copies of covariance matrix in memory than necessary.
class Brick_Wall_Function(autograd.Function):
    @staticmethod
    def forward(ctx:torch.Tensor,
                covariance_matrix:torch.Tensor,
                a:float,
                layer_index:int,
                zero_qubit:bool,
                measurement_sites,
                *chi_input_tuple
                ) -> torch.Tensor:
        """
        Custom autograd function for implementing forward pass/back pass in matchgate brickwall circuit (one layer). The covariance matrix is updated, measurements are simulated.
        Post measurement covariance matrix and measurement results are passed onto next layer. The  backpass is done via differentiating the following backwards activation function
                                                             d_back = 2*p(s=+1|previous layer measurements)-1
        forward                                                     
        Parameters
        ----------
        ctx : torch.Tensor
            usual context tensor for saving tensors from forward pass for back pass
        covariance_matrix : torch.Tensor
            the covariance matrix coming into the layer
        a : float
            hyper-parameter associated with mapping input activations to chi, determines how much to stretch/squeeze that function horizontally
        layer_index : int
            index for the layer counting from 0
        zero_qubit : bool
            whether to zero all the measured qubits once the layer measurements are complete. The measurement values still map to the rotation angles of the next layer
        measurement_sites : torch.Tensor
            list of measurement sites 0 to num_qubits-1
        *chi_input_tuple : tuple
            tuple containing the inputs to chi for different basis in order XX,XY,YX,YY,ZI,IZ

        Returns
        -------
        covariance_matrix: torch.Tensor
            post-measurement covariance matrix to be used next layer
        activations : torch.Tensor
            activations is a 1d tensor of length num_qubits. It is 0 except at in its positions correponding to the measurement sites where it has the 
                correspoding measurement value (+-1)
                
        backward
        Parameters
        ----------
        ctx : torch.Tensor
            usual context tensor for saving tensors from forward pass for back pass
        old_partial_cost_partial_covariance: torch.Tensor
            this is only None, not used
        partial_cost_partial_activation: torch.Tensor
            partial_cost_partial_activation coming from next layer, 1d length num_qubits
        
        Returns
        -------
        None for everything except the updated partial_cost_parial_chi_input. This is a list of 6 1d tensors length num_gates
        
        """
        #lets reformat this to be a num_gates x 6 2d tensor since this is the form those forward and backwards functions expect for chi
        chi_input = torch.stack(list(chi_input_tuple),dim=0).squeeze(1).t()
        if chi_input.dim() ==1:
            chi_input = chi_input.view(1,-1)
        chi = phi_chi(chi_input,a)
        covariance_matrices,update_matrix, measurement_results = Brick_Wall_Layer_Forward(covariance_matrix,chi, measurement_sites,layer_index)
        ctx.save_for_backward(covariance_matrices[0,:,:], chi, measurement_sites, measurement_results, update_matrix, chi_input, torch.tensor(a),torch.tensor(layer_index))
        #activations is zero at unmeasured sites and the appropriate
        num_qubits = covariance_matrices.shape[1]//2
        activations = torch.zeros(num_qubits)
        activations.scatter_(0,measurement_sites.long(),measurement_results)
        out_covariance_matrix = covariance_matrices[-1,:,:]
        #If zero_qubit then reset measured qubits to zero
        if zero_qubit:
            odd_indices = 2*measurement_sites
            even_indices = 2*measurement_sites + 1
            out_covariance_matrix[even_indices,odd_indices] = 1.0
            out_covariance_matrix[odd_indices, even_indices] = -1.0
        return (out_covariance_matrix, activations)
    @staticmethod
    def backward(ctx:torch.Tensor,old_partial_cost_partial_covariance, partial_cost_partial_activation) -> torch.Tensor:
      #Note this backward takes the two parts from the forward, but only partial_cost_partial_covariance is used in this method
      covariance_matrix, chi, measurement_sites, measurement_results, update_matrix,chi_input, a, layer_index = ctx.saved_tensors
      partial_cost_partial_chi = Brick_Wall_Layer_Back(partial_cost_partial_activation,covariance_matrix, update_matrix, chi, measurement_sites, measurement_results, layer_index)
      partial_cost_partial_chi_input = [(partial_cost_partial_chi[i]*phi_chi_prime(chi_input[:,i],a)).view(1,-1) for i in range(6)]
      # return None for all the parameters of the forward method we don't want to calculate gradients for
      return None, None, None, None, None, *partial_cost_partial_chi_input

    
class Brick_Wall(nn.Module):
    def __init__(self,in_size:int, num_qubits:int, layer_index:int, a:float, measurement_probability:float, bias:bool, zero_qubit:bool):
        super(Brick_Wall,self).__init__()
        """
        Implements forward pass for a hidden layer, but not the last hidden layer
        
        Attributes
        ----------
        num_qubits(int): number qubits in the network
        chi_linear(ModuleList): module list of the linear mappings to the angles
        a(float): hyper-parameter associated with mapping from activations to angles of next layer, measure of the width of the linear part of the mapping functions
        in_size(int): size of previous layer activations
        out_size(int): number of two-qubit gates in the layer
        zero_qubit(bool): whether to zero measured qubits after measuring that layer
        layer_index(int): index layer counting from 0
        measurement_probability(float): all sites have same measurement_probability
        initial_covariance_matrix(torch.Tensor): covariance matrix of the all |0> state
        bias(bool): whether to include bias
        
        Parameters
        ----------
        in_size(int): size of previous layer activations
        num_qubits(int): number qubits in the network
        layer_index(int): index layer counting from 0
        a(float): hyper-parameter associated with mapping from activations to angles of next layer, measure of the width of the linear part of the mapping functions
        measurement_probability(float): all sites have same measurement_probability
        bias(bool): whether to include bias
        zero_qubit(bool): whether to zero measured qubits after measuring that layer
        
        Methods
        -------
        forward(self,input_tuple) -> torch.Tensor
        
        """        
        if layer_index % 2 ==0:
            out_size = num_qubits//2
        else:
            out_size = num_qubits//2 - 1
        self.num_qubits = num_qubits
        self.chi_linear = nn.ModuleList([nn.Linear(in_size, out_size,bias = bias) for i in range(6)])                 
        self.a = a
        self.bias = bias
        self.in_size = in_size
        self.zero_qubit = zero_qubit
        self.layer_index = layer_index
        self.measurement_probability = measurement_probability
        self.initial_covariance_matrix = torch.zeros((2*self.num_qubits, 2*self.num_qubits))
        #odd and even indices if you count from 1 (so the reverse in python)
        odd_indices = 2*torch.arange(self.num_qubits)
        even_indices = 2*torch.arange(self.num_qubits) + 1
        #the covariance matrix for the initial spin |0> state
        self.initial_covariance_matrix[even_indices,odd_indices] = 1
        self.initial_covariance_matrix[odd_indices,even_indices] = -1     
    def forward(self,input_tuple) -> torch.Tensor:
        measurement_sites = Generate_Measurement_Sites(self.num_qubits, self.measurement_probability)
        if self.layer_index ==0:
            activations_in = input_tuple
            covariance_matrix = self.initial_covariance_matrix
        else:
            (covariance_matrix, activations_in)=input_tuple
            previous_measurement_sites = activations_in.nonzero(as_tuple=True)[0]
            #We need to apply mask so that partial_cost_partial_activation does include contributions of unmeasured sites on backpass
            activations_in = dropout_keep(activations_in, previous_measurement_sites, dim=-1)
        chi_input = [lin(activations_in) for lin in self.chi_linear]
        return Brick_Wall_Function.apply(covariance_matrix ,self.a, self.layer_index,self.zero_qubit, measurement_sites, *chi_input)
    
"""Code for last hidden layer activation function. There are a couple reasons to define this separately,
1. It only takes partial_cost_partial_activations (derivative with respect to measurement_results) and not partial_cost_partial_covariance
2. The drop out function needs to be applied to activations after the forward pass (if dropout is desired on the last hidden layer)
"""    
class Brick_Wall_Last_Function(autograd.Function):
    @staticmethod
    def forward(ctx:torch.Tensor,
                covariance_matrix:torch.Tensor,
                a:float,
                layer_index:int,
                zero_qubit:bool,
                measurement_sites,
                *chi_input_tuple
                ) -> torch.Tensor:
        """
        See Brick_Wall_Function. The only difference is this verion only passes forward/ recieves backward activations/partial_cost_partial_activation
        and not covariance matrix/partial_cost_partial_covariance
        """
        #lets reformat this to be a num_gates x 6 2d tensor since this is the form those forward and backwards functions expect for chi
        chi_input = torch.stack(list(chi_input_tuple),dim=0).squeeze(1).t()
        if chi_input.dim() == 1:
            chi_input = chi_input.view(1,-1)
        chi = phi_chi(chi_input,a)
        covariance_matrices,update_matrix, measurement_results = Brick_Wall_Layer_Forward(covariance_matrix,chi, measurement_sites,layer_index)
        ctx.save_for_backward(covariance_matrices[0,:,:], chi, measurement_sites, measurement_results, update_matrix, chi_input, torch.tensor(a),torch.tensor(layer_index))
        #activations is zero at unmeasured sites and the appropriate
        num_qubits = covariance_matrices.shape[1]//2
        activations = torch.zeros(num_qubits)
        activations.scatter_(0,measurement_sites.long(),measurement_results)
        return activations
    @staticmethod
    def backward(ctx:torch.Tensor,partial_cost_partial_activation:torch.Tensor) -> torch.Tensor:
      covariance_matrix, chi, measurement_sites, measurement_results, update_matrix,chi_input, a, layer_index = ctx.saved_tensors
      partial_cost_partial_chi = Brick_Wall_Layer_Back(partial_cost_partial_activation,covariance_matrix, update_matrix, chi, measurement_sites, measurement_results, layer_index)
      partial_cost_partial_chi_input = [(partial_cost_partial_chi[i]*phi_chi_prime(chi_input[:,i],a)).view(1,-1) for i in range(6)]
      # return None for all the parameters of the forward method we don't want to calculate gradients for
      return None, None, None, None, None, *partial_cost_partial_chi_input
  
    
class Brick_Wall_Last(nn.Module):
    def __init__(self,in_size:int, num_qubits:int, layer_index:int, a:float, measurement_probability:float, bias:bool, zero_qubit:bool):
        super(Brick_Wall_Last,self).__init__()
        """
        See Brick_Wall. Two differences between this and that.
        1. No initial step with initial covariance matrix
        2. dropout is applied after as well as before forward propagation. This ensures terms partial_cost_partial_activation only include contributions from 
        sites that were measured
        """        
        if layer_index % 2 ==0:
            out_size = num_qubits//2
        else:
            out_size = num_qubits//2 - 1
        self.num_qubits = num_qubits
        self.bias = bias
        self.chi_linear = nn.ModuleList([nn.Linear(in_size, out_size,bias = bias) for i in range(6)])                 
        self.a = a
        self.in_size = in_size
        self.zero_qubit = zero_qubit
        self.layer_index = layer_index
        self.measurement_probability = measurement_probability
    def forward(self,input_tuple) -> torch.Tensor:
        measurement_sites = Generate_Measurement_Sites(self.num_qubits, self.measurement_probability)
        (covariance_matrix, activations_in)=input_tuple
        previous_measurement_sites = activations_in.nonzero(as_tuple=True)[0]
        #We need to apply mask so that partial_cost_partial_activation does include contributions of unmeasured sites on backpass
        activations_in = dropout_keep(activations_in, previous_measurement_sites, dim=-1)
        chi_input = [lin(activations_in) for lin in self.chi_linear]
        last_activations = Brick_Wall_Last_Function.apply(covariance_matrix ,self.a, self.layer_index,self.zero_qubit, measurement_sites, *chi_input) 
        last_activations = dropout_keep(last_activations,measurement_sites,dim=-1)
        return last_activations

  

class Quantum_Network(nn.Module):
    def __init__(self,
                 num_qubits:int,
                 num_hidden_layers:int,
                 in_size:int,
                 out_size:int,
                 bias:bool=False,
                 a:float=0.5,
                 measurement_probability:float = 0.5,
                 zero_qubit:bool = False
                 ) -> "Quantum_Network":
        super(Quantum_Network,self).__init__()
        """
        Attributes
        ----------
        num_qubits(int): number qubits in the network
        a(float): hyper-parameter associated with mapping from activations to angles of next layer, measure of the width of the linear part of the mapping functions
        in_size(int): size of previous layer activations
        out_size(int): number of two-qubit gates in the layer
        zero_qubit(bool): whether to zero measured qubits after measuring that layer
        measurement_probability(float): all sites have same measurement_probability
        bias(bool): whether to include bias
        num_hidden_layers(int): number of hidden(quantum) layers
        layers(nn.Sequential): layers of the network using nn.Sequential
        
        Parameters
        ----------
        in_size(int): size of input layer
        out_size(int): size of output layer
        num_qubits(int): number qubits in the network
        num_hidden_layers(int): number of hidden(quantum) layers
        a(float): hyper-parameter associated with mapping from activations to angles of next layer, measure of the width of the linear part of the mapping functions
        measurement_probability(float): all sites have same measurement_probability
        bias(bool): whether to include bias
        zero_qubit(bool): whether to zero measured qubits after measuring that layer
        
        Methods
        -------
        forward(input:torch.Tensor) -> torch.Tensor
        
        """
        self.num_qubits = num_qubits
        self.num_hidden_layers = num_hidden_layers
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.a = a
        self.zero_qubit = zero_qubit
        self.measurement_probability = measurement_probability
        # build the layers of the network
        layers=[Brick_Wall(in_size, num_qubits,0, a, measurement_probability, bias,zero_qubit)]
        for layer_index in range(1,num_hidden_layers-1):
            layers.append(Brick_Wall(num_qubits, num_qubits,layer_index, a, measurement_probability, bias,zero_qubit))
        #Use different custom function for the last hidden layer
        layers.append(Brick_Wall_Last(num_qubits, num_qubits,num_hidden_layers-1, a, measurement_probability, bias,zero_qubit))
        layers.append(nn.Linear(num_qubits,out_size,bias=bias))
        self.net=nn.Sequential(*layers)
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        return self.net(input)

def init_weights(model:torch.nn.Module):
    # Kaiming initializes all weights of the model
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight,nonlinearity='linear')

def prep_data(train_size:int,
              batch_size_train:int,
              test_size:int,
              batch_size_test:int
              ) -> tuple[torch.utils.data.dataloader.DataLoader]:
    """
    Prepares training and testing data

    Parameters:
        train_size (int): the number of training data to prepare
        batch_size_train (int): the batch size for the training data
        test_size (int): the number of testing data to prepare
        batch_size_test (int): the batch size for the testing data

    Returns:
        tuple[torch.utils.data.dataloader.DataLoader]: contains 2 dataloaders, the first for training, the second for testing
    """
    # prepare training data
    train_set = datasets.MNIST('data/mnist_data', train=True, download=True, 
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                                torch.flatten
                             ]))
    train_set=torch.utils.data.Subset(train_set,range(0,train_size))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size_train, shuffle=True)

    # prepare testing data
    test_set = datasets.MNIST('data/mnist_data', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                                 torch.flatten
                             ]))
    test_set=torch.utils.data.Subset(test_set,range(0,test_size))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)

    return train_loader,test_loader
@torch.no_grad()
def _with_mode(model: nn.Module, mode: bool):
    """
    Context manager that temporarily switches model.train(mode) and restores it.
    """
    class _Ctx:
        def __enter__(self_nonlocal):
            _Ctx.prev = model.training
            model.train(mode)
        def __exit__(self_nonlocal, exc_type, exc, tb):
            model.train(_Ctx.prev)
    return _Ctx()


@torch.no_grad()
def mc_logits(model: nn.Module, x: torch.Tensor, num_shots: int) -> torch.Tensor:
    """
    Run num_shots stochastic forward passes and stack logits: [S, C].
    We deliberately put the model in TRAIN mode for the duration of the MC passes
    so any stochastic layers/masks fire, then restore the original mode.
    """
    outs = []
    with _with_mode(model, True):  # enable stochastic paths, then restore
        for _ in range(num_shots):
            y = model(x)                      # [C] or [1, C]
            if y.dim() == 1:
                y = y.unsqueeze(0)            # -> [1, C]
            outs.append(y.squeeze(0))         # -> [C]
    return torch.stack(outs, dim=0)           # [S, C]


@torch.no_grad()
def mc_vote_class(model: nn.Module, x: torch.Tensor, num_shots: int) -> torch.Tensor:
    shots = mc_logits(model, x, num_shots)    # [S, C]
    shot_preds = shots.argmax(dim=-1)         # [S]
    return torch.mode(shot_preds).values      # scalar LongTensor


@torch.no_grad()
def get_err(loader: torch.utils.data.dataloader.DataLoader,
            model: torch.nn.Module,
            num_shots: int = 1) -> float:
    """
    Evaluation with true MC-dropout voting. We run eval overall (no grads),
    and temporarily flip to train-mode only during the S stochastic passes
    for each example; the model’s mode is restored afterwards.
    """
    # Keep outer eval context for layers like BatchNorm stats, etc.
    # Your code doesn’t use BatchNorm, but this keeps semantics standard.
    prev_mode = model.training
    model.eval()
    try:
        batch_errs = []
        for imgs, labels in loader:
            preds = []
            for i in range(imgs.size(0)):
                pred_i = mc_vote_class(model, imgs[i:i+1], num_shots)  # scalar
                preds.append(pred_i)
            preds = torch.stack(preds, dim=0)                           # [B]
            batch_errs.append((preds != labels).float().mean().item())
        return float(np.mean(batch_errs))
    finally:
        model.train(prev_mode)

@torch.no_grad()
def get_err_from_out(output: torch.Tensor,
                     labels: torch.Tensor,
                     num_shots: int = 1) -> float:
    """
    Works for outputs that are either [C] or [B, C].
    Returns the error rate over the (logical) batch.
    """
    if output.dim() == 1:
        output = output.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    # One forward already happened; repeating argmax num_shots times
    # just reproduces the same prediction
    preds_list = [torch.argmax(output, dim=-1) for _ in range(num_shots)]
    preds = torch.mode(torch.stack(preds_list, dim=0), dim=0).values

    wrong = (preds != labels).float()
    return float(wrong.mean().item())


@torch.no_grad()
def get_loss(loader: torch.utils.data.dataloader.DataLoader,
             net: torch.nn.Module,
             loss_fn: torch.nn.modules.loss._Loss) -> float:
    """
    Computes average loss over the dataset when the network processes one image at a time.
    Returns the mean of per-batch average losses (matching your original API).
    """
    net.eval()
    batch_losses: list[float] = []

    for imgs, labels in loader:
        # Accumulate per-sample loss and average over the logical batch
        running = 0.0
        for i in range(imgs.size(0)):
            preds = net(imgs[i:i+1])                 # [C] or [1, C]
            if preds.dim() == 1:
                preds = preds.unsqueeze(0)           # [1, C]
            lbl = labels[i:i+1]                      # [1]
            running += float(loss_fn(preds, lbl).item())
        batch_losses.append(running / imgs.size(0))

    return float(np.mean(batch_losses))


def train(net: torch.nn.Module,
          train_loader: torch.utils.data.dataloader.DataLoader,
          test_loader: torch.utils.data.dataloader.DataLoader,
          learning_rate: float,
          momentum: float,
          num_shots: int,
          num_epochs: int,
          step: int) -> dict:
    """
    Train the network using batch gradient descent while running
    the forward/backward pass on one image at a time.
    """
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    device = next(net.parameters()).device
    net.train()

    first_params = [param.detach().tolist() for param in net.parameters()]
    losses: list[float] = []
    train_err_rates: list[float] = []
    err_rates = [get_err(test_loader, net, num_shots)]  # initial test error

    is_first_batch = True
    for epoch in tqdm(range(num_epochs)):
        should_record = (epoch % step == step - 1) or (epoch == num_epochs - 1)

        for images_batch, labels_batch in train_loader:
            net.train()
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            batch_size = images_batch.size(0)
            running_loss = 0.0

            # --- process one image at a time, accumulate grads ---
            for sample_index in range(batch_size):
                image = images_batch[sample_index:sample_index + 1]   # [1, ...]
                label = labels_batch[sample_index:sample_index + 1]   # [1]

                logits = net(image)                    # [C] or [1, C]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)       # -> [1, C]
                label = label.view(-1).long()          # ensure [1], Long

                loss_value = loss_fn(logits, label) / batch_size
                loss_value.backward()
                running_loss += float(loss_value.item())

                last_prediction = logits.detach()      # keep last for logging
                last_label = label.detach()

                if is_first_batch:
                    # log with batched shapes
                    train_err_rates.append(get_err_from_out(last_prediction, last_label, 1))
                    losses.append(float(loss_value.detach().item() * batch_size))  # un-averaged per your original behavior
                    is_first_batch = False

            optimizer.step()
            batch_loss_value = running_loss

        # --- record epoch-level stats if requested ---
        if should_record:
            err_rates.append(float(get_err(test_loader, net, num_shots)))
            train_err_rates.append(get_err_from_out(last_prediction, last_label, 1))
            losses.append(float(batch_loss_value))

    last_params = [param.detach().tolist() for param in net.parameters()]
    accuracy = 1 - get_err(test_loader, net, num_shots)

    return {
        "test_accuracy": accuracy,
        "losses": losses,
        "err_rates": err_rates,
        "train_err_rates": train_err_rates,
        "first_params": first_params,
        "last_params": last_params,
    }



def record(out_dict:dict,directory:str,filename:str):
    """
    Writes contents of a dictionary to a json file

    Parameters:
        out_dict (dict): dictionary to write to file
        directory (str)
        filename (str)
    """
    os.makedirs(directory,exist_ok=True)
    with open(directory+'/'+filename,'w') as outfile:
        json.dump(out_dict,outfile)