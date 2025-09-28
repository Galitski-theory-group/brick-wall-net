## Natural Quantization 

This repository stores code and coding documentation on an implementation of a quantum neural network whose design and simulation are outlined [here](https://arxiv.org/abs/2503.15482). This code implements the quantum neural network above on actual ibm devices. 

### Design of the Network 

Methods

Overview
	•	In prior work, a partially binarized multilayer perceptron (PBMLP) was quantized by replacing neurons with qubits and non-linear activations with a rotate-measure operator.
	•	A classical MLP has layers l_0, l_1, \dots, l_n. The input vector d^(0) of size m₀ is fed layer-by-layer through an activation function φ to produce final output f of size mₙ.
	•	Between layer l-1 and layer l, activations d^(l-1) are multiplied by weight matrix W^(l) and added to bias b^(l) to form logits z^(l), then passed through φ to get d^(l). The weight matrices and bias vectors are tunable parameters, initialized and then optimized during training.
	•	Training uses a dataset of input/output pairs and stochastic gradient descent with backpropagation to minimize a cost function J(W, b). Classification maps the output vector fᵢ to a category, e.g. by choosing the index of the maximum component.

Notation and Feedforward in Plain Form
	•	Layers:
	•	Input layer l_0 holds vector d^(0) of length m₀.
	•	Hidden layers l_1, l_2, \dots, l_{n-1} hold vectors d^(1), d^(2), … of sizes m₁, m₂, … respectively.
	•	Output layer l_n produces final vector f of length mₙ.
	•	Between layer l-1 and layer l:
	•	Compute logits:

    z^(l) = W^(l) · d^(l-1) + b^(l)

### Structure of "Package" 

#### src 

This is where actual code is contained.

The code is structured as a python package. To actually perform the experiments users work with the __main__.py application which is a simple typer application. 

The code pipeline is very simple. 

app -> main -> run_experiment -> 

```mermaid
flowchart LR
    TA[Typer Application]
    MF[main.py]
    QNN[QuantumNeuralNetwork Class]
    P[predict()]
    FF[feedforward()]

    TA --> MF
    MF --> QNN
    QNN --> P
    P --> FF
```

```mermaid
    classDiagram
    class TyperApp {
        +run()
    }
    class Main {
        +main()
    }
    class QuantumNeuralNetwork {
        +predict(input)
        -feedforward(input)
    }

    TyperApp --> Main : invokes
    Main --> QuantumNeuralNetwork : calls \n predict()
    QuantumNeuralNetwork --> predict : defines
    predict --> feedforward : calls
    ```