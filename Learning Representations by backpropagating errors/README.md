# Overview

Backpropagation was first introduced in this paper and described as such:

'The procedure repeatedly adjusts the weights of the connections in the network so as to minimize a measure of the difference between the actual output vector of the net and the desired output vector. '

There are a few important things to note here. There's the **weights** and a **measure of the difference** between the actual output vector and the desired output vector. So basically they are saying that based on some way of **measuring the difference** the weights will be adjusted but till now it's not clear how.

As a result of the weight adjustments, hidden units start to represent certain important features of the task domain.

Then there is a comparison between backpropagation and perceptron-convergence procedure. Perceptron-convergence procedure was the predecessor of backpropagation. It's very simple and I wanted to look into what the logic was. Basically, the idea was that the error was used to update the weights but this only applied to linearly separable problems (i.e. AND gate). This method would always get the right weights in a finite number of steps for linearly separable problems.

'The simplest form of the learning procedure is for layered
networks which have a layer of input units at the bottom; any
number of intermediate layers; and a layer of output units at
the top. Connections within a layer or from higher to lower
layers are forbidden, but connections can skip intermediate
layers.'

The above is the explanation of the network connections for backpropagation. The idea of layers describes an MLP but then they also say that connections can skip intermediate layers which is an interesting idea. 

During a single forward pass the inputs are first set. Then they are multiplied by their weights, added to a bias and run through the activation function to determine the 'state' of the one of the neurons in the next layer. The idea of representing the bias as an additional value equal to 1 is also explained.

At this point there are some key terminology that I want to highlight that are different to what I have seen, probably because they are discussing at a low-level:
 - state: the value of the unit, for the inputs its the given value, for the hidden units its the value after the linear combination of inputs and the non-linear transformation.
 - threshold: it's the value above which the neuron activates (e.g. outputs 1)
 - bias: a term added to the linear combination of the inputs. b = -T, since sum(x_i * w_i) > T
 - unit: represents a neuron which is the output of linear combination + non-linear transformation

The computing process is explained mathematically as follows. Each unit gets a linear combination of the i inputs, y_i is all the inputs and the output for the linear combination is sum(x_i * w_i) + b. Then the output undergoes a non-linear transformation based on the sigmoid equation ($\sigma(x) = \frac{1}{1 + e^{-x}}$) in this case.

Input (y_i) -> Linear Combination (x_j) -> Non-Linear Transformation (y_j)
y is the input for the unit and x is the output. i is the ith input and j is the jth unit (neuron).


