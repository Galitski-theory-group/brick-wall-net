from main import *


def test_phia():
    assert phia(torch.tensor(1.2),0.5) == torch.tensor(1.0)
    assert phia(torch.tensor(0.2),0.5) ==torch.tensor(0.2/0.5)
    assert phia(torch.tensor(-4),0.5) == torch.tensor(-1.0)

def test_phia_prime():
    assert phia_prime(torch.tensor(0.3),0.5) == torch.tensor(1.0/0.5)
    assert phia_prime(torch.tensor(1.2),0.5) == torch.tensor(0.0)


def test_dropout_keep():
    test_activations = torch.tensor([0.0,1.0,2.0,3.0,4.0])
    keep_index = torch.tensor([1,3])
    dropped_activation = dropout_keep(test_activations,keep_index,dim=-1)
    assert torch.equal(dropped_activation, torch.tensor([0.0,1.0,0.0,3.0,0.0]))
    
def test_sgd_step():
    """
    Function does full SGD step (forward & backward). It create an instance of Quantum_Network
    and initializes the weights. It does forward and backpass on an image using the model
    in Quantum_Network and updates the weights.It saves all measurement results and weights. 
    Using these a SGD step update is done using matrix operations, the Brick_Wall_Layer_Back function,
    and a deterministic version of Brick_Wall_Layer_Forward that accepts measurement results as inputs.
    It is tested if this more manual update gives the same result for the new weights within
    expected numeric precision.
    """
    num_qubits = 16
    num_hidden_layers = 2
    input_size = 784
    output_size = 10
    measurement_probability = 0.75
    zero_qubit = False
    train_size=1
    train_batch_size = 1
    test_size=1
    test_batch_size=1
    a=0.5
    #initialize train and test loaders
    train_loader,test_loader=prep_data(train_size,train_batch_size,test_size,test_batch_size)
    #create instance of Quantum_Network
    network=Quantum_Network(num_qubits,num_hidden_layers,input_size,output_size,bias = False ,a=a, measurement_probability = measurement_probability, zero_qubit=zero_qubit)
    #Initialize weights
    network.apply(init_weights)
    model = network.net
    #Extract the weights
    chi_weights= [[model[i].chi_linear[j].weight.clone().detach() for j in range(6)] for i in range(num_hidden_layers)]
    last_weights = model[num_hidden_layers].weight.clone().detach()
    #Register hooks to save the activations during the forward pass
    _ = enable_step_activation_capture(model, save_dir="training_logs", clear=True)
    #"train" the network, just one SGD step
    learning_rate = 0.01
    momentum = 0.9
    num_shots = 1
    num_epochs = 1
    step=1
    res_dict=train(model,train_loader,test_loader,learning_rate,momentum,num_shots,num_epochs,step)
    new_chi_weights= [[model[i].chi_linear[j].weight.clone().detach() for j in range(6)] for i in range(num_hidden_layers)]
    new_last_weights = model[num_hidden_layers].weight.clone().detach()
    #get the activations
    all_activations = load_step_activations("training_logs")
    #extract the activations from the training step
    activations = all_activations[1]
    # get the input activation for that image
    device = next(model.parameters()).device
    imgs, labels = next(iter(train_loader))   # tensors after your transforms
    x = imgs[0].to(device).reshape(-1).contiguous()  # shape [784] for MNIST
    activations.insert(0,x)
    """
    Manual SGD step. The forward propagation uses a deterministic version of the code so we get the same measurement results.
    """
    #Manual Forward Pass
    #initialize covariance matrix
    initial_covariance_matrix = torch.zeros((2*num_qubits, 2*num_qubits))
    #odd and even indices if you count from 1 (so the reverse in python)
    odd_indices = 2*torch.arange(num_qubits)
    even_indices = 2*torch.arange(num_qubits) + 1
    #the covariance matrix for the initial spin |0> state
    initial_covariance_matrix[even_indices,odd_indices] = 1
    initial_covariance_matrix[odd_indices,even_indices] = -1
    covariance_matrix = initial_covariance_matrix
    #post-unitary/pre-measurement covariance matrices used in backprop
    covariance_matrices_back = torch.zeros(num_hidden_layers,2*num_qubits,2*num_qubits)
    #SO(2N) update matrices for backpass
    update_matrices = torch.zeros(num_hidden_layers,2*num_qubits,2*num_qubits)
    chi_layers = []
    measurement_sites_layers = []
    chi_input_layers = []
    for layer_index in range(num_hidden_layers):
        #calculate chi
        chi_input =[chi_weights[layer_index][i] @ activations[layer_index] for i in range(6)]
        chi = torch.stack([phi_chi(chi_input[i],a) for i in range(6)]).t()
        measurement_sites = torch.nonzero(activations[layer_index+1], as_tuple=False).squeeze()
        cov_temp, R_temp = Deterministic_Brick_Wall_Layer_Forward(covariance_matrix,chi, measurement_sites,layer_index, activations[layer_index+1])
        covariance_matrix[:,:] = cov_temp[-1,:,:]
        covariance_matrices_back[layer_index,:,:] = cov_temp[0,:,:]
        update_matrices[layer_index,:,:] = R_temp
        chi_layers.append(chi)
        chi_input_layers.append(chi_input)
        measurement_sites_layers.append(measurement_sites)
    #check that the last layer is what we expect
    assert torch.equal(activations[-1],last_weights @ activations[-2])
    #Manual Backwards Pass
    #backprop through last layer
    partial_cost_partial_activation = 1/train_batch_size * torch.exp(activations[-1])/(torch.exp(activations[-1]).sum())
    partial_cost_partial_activation[labels] = partial_cost_partial_activation[labels] - 1/train_batch_size
    partial_cost_partial_last_weights = torch.outer(partial_cost_partial_activation,activations[-2])
    partial_cost_partial_activation = last_weights.t() @ partial_cost_partial_activation
    temp = torch.zeros_like(partial_cost_partial_activation)
    temp[measurement_sites_layers[-1]] = partial_cost_partial_activation[measurement_sites_layers[-1]]
    partial_cost_partial_activation = temp
    #backprop through the hidden layers
    partial_cost_partial_chi_weights=[]
    for layer_index in range(num_hidden_layers-1,-1,-1):
        measurement_results = activations[layer_index+1][activations[layer_index+1] !=0]
        partial_cost_partial_chi = Brick_Wall_Layer_Back(partial_cost_partial_activation,covariance_matrices_back[layer_index,:,:],update_matrices[layer_index,:,:],chi_layers[layer_index], measurement_sites_layers[layer_index], measurement_results,layer_index)
        partial_cost_partial_chi_input = [partial_cost_partial_chi[i]*phi_chi_prime(chi_input_layers[layer_index][i],a) for i in range(6)]
        partial_cost_partial_chi_weights.append([torch.outer(partial_cost_partial_chi_input[i],activations[layer_index]) for i in range(6)])
        partial_cost_partial_activation = ( chi_weights[layer_index][0].t() @ partial_cost_partial_chi_input[0] +chi_weights[layer_index][1].t() @ partial_cost_partial_chi_input[1] 
        +chi_weights[layer_index][2].t() @ partial_cost_partial_chi_input[2] +chi_weights[layer_index][3].t() @ partial_cost_partial_chi_input[3]
        +chi_weights[layer_index][4].t() @ partial_cost_partial_chi_input[4] +chi_weights[layer_index][5].t() @ partial_cost_partial_chi_input[5])
        if layer_index !=0:
            temp = torch.zeros_like(partial_cost_partial_activation)
            temp[measurement_sites_layers[layer_index-1]] = partial_cost_partial_activation[measurement_sites_layers[layer_index-1]]
            partial_cost_partial_activation = temp
        
    partial_cost_partial_chi_weights.reverse()
    #Now to compare with the training step using Quantum_Network 
    last_step1 = new_last_weights - last_weights
    last_step2 = -learning_rate*partial_cost_partial_last_weights
    assert torch.allclose(last_step1,last_step2,rtol=1e-5,atol=1e-06)
    #new_chi_weights= [[model[i].chi_linear[i].weight.clone().detach() for j in range(6)] for i in range(num_hidden_layers)]
    step1_list = []
    step2_list = []
    for i in range(num_hidden_layers):
        for j in range(6):
            step1 = new_chi_weights[i][j] - chi_weights[i][j]
            step2 = -learning_rate*partial_cost_partial_chi_weights[i][j]
            step1_list.append(step1.sum())
            step2_list.append(step2.sum())
            assert torch.allclose(step1,step2,rtol=1e-5,atol=1e-06)

    
