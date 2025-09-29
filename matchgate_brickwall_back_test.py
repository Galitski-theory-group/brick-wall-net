import numpy as np
import torch
import time
from scipy.linalg import expm


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

    Inputs:
        old_covariance_matrix: covariance matrix of state coming into the layer
        chi: 2-qubit angles for the layer (6, num_gates). Assumes order XX,XY,YX,YY,ZI,IZ
        measurement_sites: list of measurement sites for layer (0 to num_qubits-1)
        layer_index: index of layer (counting from 0)
    
    Outputs:
        covariance_matrices: 3d torch.Tensor (2,2*num_qubits,2*num_qubits). It is covariance matrix from post-unitary/pre-measurement(for backpass)
            and post-measurement (to continue forward pass)
        update_matrix: The SO(2*num_qubits) update matrix. It is saved for the backpass
        measurement_results: tensor with measurement results for each measurement site +-1
    """
    device = old_covariance_matrix.device
    dtype  = old_covariance_matrix.dtype
    # read off the number of two_qubit gates and angles per gate from chi
    num_gates = chi.shape[0]
    #construct a 3-d tensor for storing the covariance matrices, these will be used in the backwards pass
    covariance_size = old_covariance_matrix.shape[0]
    num_measurements = len(measurement_sites)
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
    Inputs:
        partial_cost_partial_activation: From previous layer
        covariance_matrix: covariance matrix from post-unitary/pre-measurement state
        update_matrix: The SO(2*num_qubits) update matrix
        chi: 2-qubit angles for the layer (6, num_gates). Assumes order XX,XY,YX,YY,ZI,IZ
        measurement_sites: list of measurement sites for layer (0 to num_qubits-1)
        measurement_results: tensor with measurement results for each measurement site +-1
        layer_index: index of layer (counting from 0)
    Outputs:
        partial_cost_partial_chi: list of 6 tensors with the derivative of the cost function with respect to each chi in the layer
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

#number of qubits
num_qubits=4
#number of hidden layers
num_hidden_layers = 3

initial_covariance_matrix = torch.zeros((2*num_qubits, 2*num_qubits))
#odd and even indices if you count from 1 (so the reverse in python)
odd_indices = 2*torch.arange(num_qubits)
even_indices = 2*torch.arange(num_qubits) + 1
#the covariance matrix for the initial spin |0> state

initial_covariance_matrix[even_indices,odd_indices] = 1
initial_covariance_matrix[odd_indices,even_indices] = -1

measurement_sites_list = [Generate_Measurement_Sites(num_qubits, 0.5) for layer_index in range(num_hidden_layers)]

"""Simulate a layer for each case where only one of the 6 basis in the 1st gate is nonzero"""
#define the chi for each case
chi_xx = torch.zeros((num_qubits//2,6))
chi_xy = torch.zeros((num_qubits//2,6))
chi_yx = torch.zeros((num_qubits//2,6))
chi_yy = torch.zeros((num_qubits//2,6))
chi_zi = torch.zeros((num_qubits//2,6))
chi_iz = torch.zeros((num_qubits//2,6))

chi_val = 2*torch.pi*torch.rand(1)
chi_xx[0,0] = chi_val
chi_xy[0,1] = chi_val
chi_yx[0,2] = chi_val
chi_yy[0,3] = chi_val
chi_zi[0,4] = chi_val
chi_iz[0,5] = chi_val

#Apply two-qubit gate for each case, the measurement_sites are irrelevant here
cov_xx, R_xx, meas_res_xx = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_xx, measurement_sites_list[0],0)
cov_xy, R_xy, meas_res_xy = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_xy, measurement_sites_list[0],0)
cov_yx, R_yx, meas_res_yx = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_yx, measurement_sites_list[0],0)
cov_yy, R_yy, meas_res_yy = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_yy, measurement_sites_list[0],0)
cov_zi, R_zi, meas_res_zi = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_zi, measurement_sites_list[0],0)
cov_iz, R_iz, meas_res_iz = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi_iz, measurement_sites_list[0],0)


#take cost = probability for 1st qubit spin up measurement after the two-qubit gate
initial_partial_cost_partial_activation = torch.zeros(2*num_qubits)
initial_partial_cost_partial_activation[0] = 1.0
#Call the backwards function
partial_chi_xx = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_xx[0,:,:], R_xx,chi_xx, measurement_sites_list[0],meas_res_xx,0)
partial_chi_xy = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_xy[0,:,:], R_xy,chi_xy, measurement_sites_list[0],meas_res_xy,0)
partial_chi_yx = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_yx[0,:,:], R_yx,chi_yx, measurement_sites_list[0],meas_res_yx,0)
partial_chi_yy = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_yy[0,:,:], R_yy,chi_yy, measurement_sites_list[0],meas_res_yy,0)
partial_chi_zi = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_zi[0,:,:], R_zi,chi_zi, measurement_sites_list[0],meas_res_zi,0)
partial_chi_iz = Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,cov_iz[0,:,:], R_iz,chi_iz, measurement_sites_list[0],meas_res_iz,0)

print("Single Gate, Single Basis Test: Backpass Code")
print("backpass code partial p(1st qubit up)/ partial chi_xx: ", partial_chi_xx[0][0])
print("backpass code partial p(1st qubit up)/ partial chi_xy: ",partial_chi_xy[1][0])
print("backpass code partial p(1st qubit up)/ partial chi_yx: ",partial_chi_yx[2][0])
print("backpass code partial p(1st qubit up)/ partial chi_yy: ",partial_chi_yy[3][0])
print("backpass code partial p(1st qubit up)/ partial chi_zi: ",partial_chi_zi[4][0])
print("backpass code partial p(1st qubit up)/ partial chi_iz: ",partial_chi_iz[5][0])

partial_chi_transverse = -2*torch.sin(2*chi_val)
print("Analytic Single Gate, Single Basis")
print("analytic partial p(1st qubit up)/ partial chi_xx: ",partial_chi_transverse)
print("analytic partial p(1st qubit up)/ partial chi_xy: ",partial_chi_transverse)
print("analytic partial p(1st qubit up)/ partial chi_yx: ",partial_chi_transverse)
print("analytic partial p(1st qubit up)/ partial chi_yy: ",partial_chi_transverse)
print("analytic partial p(1st qubit up)/ partial chi_zi: ",0.0)
print("analytic partial p(1st qubit up)/ partial chi_iz: ",0.0)

def _kron_chain(mats):
    """Kronecker product of a list of 2x2 matrices."""
    M = mats[0]
    for A in mats[1:]:
        M = np.kron(M, A)
    return M

def X(k: int, n: int) -> np.ndarray:
    """Pauli X acting on qubit k (0-based), identity elsewhere."""
    assert 0 <= k < n
    ops = [_I]*k + [_X] + [_I]*(n-1-k)
    return _kron_chain(ops)

def Y(k: int, n: int) -> np.ndarray:
    """Pauli Y acting on qubit k (0-based), identity elsewhere."""
    assert 0 <= k < n
    ops = [_I]*k + [_Y] + [_I]*(n-1-k)
    return _kron_chain(ops)

def Z(k: int, n: int) -> np.ndarray:
    """Pauli Z acting on qubit k (0-based), identity elsewhere."""
    assert 0 <= k < n
    ops = [_I]*k + [_Z] + [_I]*(n-1-k)
    return _kron_chain(ops)
def P(k,n,s):
  return 0.5*(np.eye(2**n, dtype=complex)+s*Z(k, n))

def Brute_Force(initial_density_matrix,num_qubits,measurement_sites_list, chi_tensor, measurement_results_tensor):
  density_matrix = initial_density_matrix
  #measurement_results = [np.zeros(len(measurement_sites_list[n])) for n in range(len(measurement_sites_list))]
  chi = chi_tensor.numpy().astype(np.float64)
  #measurement_sites_list = [measurement_sites_list_tensor[n] for n in range(len(measurement_sites_list_tensor))]
  measurement_results  = measurement_results_tensor.numpy().astype(np.float64)
  Id =np.eye(2**num_qubits)
  U = np.eye(2**num_qubits)
  #Unitary update
  num_gates = num_qubits//2
  start = 0
  end = num_qubits-1
  for k in range(start,end,2):
      i = k//2
      H = chi[i,0]*X(k,num_qubits) @ X(k+1,num_qubits) +chi[i,1]*X(k,num_qubits) @ Y(k+1,num_qubits) +chi[i,2]*Y(k,num_qubits) @ X(k+1,num_qubits)
      H+= chi[i,3]*Y(k,num_qubits) @ Y(k+1,num_qubits) +chi[i,4]*Z(k,num_qubits) @ Id +chi[i,5]*Id @ Z(k+1,num_qubits)
      U = expm(-1j*H) @ U
  density_matrix = U @ density_matrix @ U.conj().T
  p_up= np.clip(np.trace(density_matrix @ P(0,num_qubits,1)).real,0,1)
  return p_up


#define chi for one layer randomly
chi = 2*torch.pi*torch.rand(num_qubits//2,6)
#Forward prop to get new covariance matrix and measurement_results
covariance_matrices, update_matrix, measurement_results = Brick_Wall_Layer_Forward(initial_covariance_matrix,chi, measurement_sites_list[0],0)

#randomly choose a basis gate to differetiate with respect to, could do all, but just doing one randomly is good enough
gate_index = 0
basis_index = torch.randint(0,6,(1,))

#calculate derivative of 1st qubit probability using backprop
initial_partial_cost_partial_activation = torch.zeros(2*num_qubits)
initial_partial_cost_partial_activation[0] = 1.0
partial_cost_partial_chi =Brick_Wall_Layer_Back(initial_partial_cost_partial_activation,covariance_matrices[0,:,:], update_matrix,chi, measurement_sites_list[0],measurement_results,0)
print("backprop code all 6 matchgates: ", partial_cost_partial_chi[basis_index][gate_index])

#Now I will do finite difference. I will use brute force code since the forward code is not double precision, double precision desired for finite difference
chi_plus = chi.clone().detach()
chi_minus = chi.clone().detach()
epsilon = 10**(-3)
chi_plus[gate_index][basis_index] += epsilon
chi_minus[gate_index][basis_index] -= epsilon


#zero state density matrix
density_matrix = np.zeros((2**num_qubits,2**num_qubits),dtype = complex)
density_matrix[0,0] = 1.0
# Single-qubit Paulis (complex dtype to cover Y)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1],
               [1, 0]], dtype=complex)
_Y = np.array([[0, -1j],
               [1j,  0]], dtype=complex)
_Z = np.array([[1,  0],
               [0, -1]], dtype=complex)

prob_plus = Brute_Force(density_matrix,num_qubits,measurement_sites_list[0],chi_plus,measurement_results)
prob_minus = Brute_Force(density_matrix,num_qubits,measurement_sites_list[0],chi_minus,measurement_results)
print("finite difference all 6 matchgates: ", 2*(prob_plus-prob_minus)/(2*epsilon))