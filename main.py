from anc import *
from sys import argv
# from quantum_nn import anc

#Matchgate Brickwall with Mid-Circuit Measurements Version 1
#
def main():
    # Note: one arguments is passed by the system, a.
    train_size=5000
    train_batch_size = 64
    test_size=1000
    test_batch_size=1000
    bias = False

    num_qubits = 24 #num_qubits should be an even integer
    num_hidden_layers = 2
    input_size=784
    output_size=10
    a = float(argv[1])

    learning_rate=0.01
    momentum=0.9
    num_shots=10
    num_epochs=1
    step=1000
    zero_qubit = True #whether to zero qubits after measuring a layer or just keep the original measurement value
    measurement_probability = 0.75

    hp_dict={
        "train_size": train_size,
        "batch_size": train_batch_size,
        "test_size": test_size,
        "num_qubits": num_qubits,
        "num_hidden_layers": num_hidden_layers,
        "a": a,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "num_shots": num_shots,
        "epochs": num_epochs,
        "step": step,
        "zero_qubit":zero_qubit        
    }

    train_loader,test_loader=prep_data(train_size,train_batch_size,test_size,test_batch_size) # prepare training and test data
    net=Quantum_Network(num_qubits,num_hidden_layers,input_size,output_size,bias = bias ,a=a, measurement_probability = measurement_probability, zero_qubit=zero_qubit) # create quantum network

    net.apply(init_weights) # apply Kaiming intialization to weights
    res_dict=train(net,train_loader,test_loader,learning_rate,momentum,num_shots,num_epochs,step) # train the network
    record(hp_dict|res_dict,'data/res_data','a'+str(a)+'.json') # write the data to a file
    
    print(res_dict["test_accuracy"])

if __name__=='__main__':
    main()