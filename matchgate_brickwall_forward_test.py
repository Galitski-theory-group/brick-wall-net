import numpy as np
import torch
import time
from scipy.linalg import expm



#Function that randomly chooses measurement sites. Each site has equal probability of being measured
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
        print("Covariance Matrix Method Spin Up Probability: ",spin_up_probability)
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

def Brick_Wall_Brute_Force(initial_density_matrix,num_hidden_layers,num_qubits,measurement_sites_list, chi_list_tensor, measurement_results_tensor):
  density_matrix = initial_density_matrix
  #measurement_results = [np.zeros(len(measurement_sites_list[n])) for n in range(len(measurement_sites_list))]
  chi_list = [chi_list_tensor[n].numpy() for n in range(len(chi_list_tensor))]
  #measurement_sites_list = [measurement_sites_list_tensor[n] for n in range(len(measurement_sites_list_tensor))]
  measurement_results  = [measurement_results_tensor[n].numpy().astype(np.float64) for n in range(len(measurement_results_tensor))]
  Id =np.eye(2**num_qubits)
  for l in range(num_hidden_layers):
    U = np.eye(2**num_qubits)
    #Unitary update
    if l % 2 ==0:
      num_gates = num_qubits//2
      start = 0
      end = num_qubits-1
      is_odd = True
    else:
      num_gates = num_qubits//2-1
      start = 1
      end = num_qubits -2
      is_odd = False
    for k in range(start,end,2):
      if is_odd:
        i = k//2
      else:
        i = (k-1)//2
      H = chi_list[l][i,0]*X(k,num_qubits) @ X(k+1,num_qubits) +chi_list[l][i,1]*X(k,num_qubits) @ Y(k+1,num_qubits) +chi_list[l][i,2]*Y(k,num_qubits) @ X(k+1,num_qubits)
      H+= chi_list[l][i,3]*Y(k,num_qubits) @ Y(k+1,num_qubits) +chi_list[l][i,4]*Z(k,num_qubits) @ Id +chi_list[l][i,5]*Id @ Z(k+1,num_qubits)
      U = expm(-1j*H) @ U
    density_matrix = U @ density_matrix @ U.conj().T
    index = 0
    for k in measurement_sites_list[l]:
      p_up= np.clip(np.trace(density_matrix @ P(k,num_qubits,1)).real,0,1)
      s = measurement_results[l][index]
      #s = 2*np.random.binomial(1, p_up, size=1)-1
      print("Brute Force Spin Up Probability: ",p_up)
      density_matrix = P(k,num_qubits,s) @ density_matrix @ P(k,num_qubits,s)/(np.trace(density_matrix @ P(k,num_qubits,s)))
      measurement_results[l][index] = np.asarray(s).real.item()
      index+=1
  return None

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
chi_list = [2*torch.pi*torch.rand(num_qubits//2-layer_index % 2,6) for layer_index in range(num_hidden_layers)]

covariance_matrix = initial_covariance_matrix
num_sims = 1
for sim in range(num_sims):
  measurement_results =[]
  covariance_matrix = initial_covariance_matrix
  for layer in range(num_hidden_layers):
    covariance_matrices,blocks, layer_results = Brick_Wall_Layer_Forward(covariance_matrix, chi_list[layer], measurement_sites_list[layer],layer)
    covariance_matrix = covariance_matrices[-1,:,:]
    measurement_results.append(layer_results)

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
Brick_Wall_Brute_Force(density_matrix,num_hidden_layers,num_qubits,measurement_sites_list,chi_list,measurement_results)


