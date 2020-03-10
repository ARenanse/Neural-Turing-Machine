# Neural-Turing-Machine
A modular implementation of the Neural Turing Machine introduced by Alex Graves et al.

Currently, two tasks have been implemented, Copy Task and Associative Recall Task as tf.keras.Model wrapper, available in the NTM_Model.py

Use them as showed in the Training Notebooks

## Architecture Implemented

Since the paper only provides the mathematical operations for the generation and use of the Heads' Weighings, not the full architecture, thus the complete architecture becomes an open ended problem, where I've used the following architecture:

![Architecture](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/Picture1.png)




## Task Results

### 1. Copy Task

**Training the above NTM on randomized sequence length between 1 and 20 yields the following results.**

#### 1.1. Till 10,000 epochs on Cross Entropy Loss.

##### Test 1:-

##### Input:

Sequence Length = 9, including the Start Of File and End Of File delimeters.

![Input seq_len = 9](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/input.png)

##### Output:

![Output](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/output.png)


##### Test 2:-

##### Input:

Sequence Length = 33, including the Start Of File and End Of File delimeters.

Note that it is more than what the above NTM is trained upon.

![Input seq_len = 33](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_input.png)

##### Output:

![Output](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_output.png)

##### Test 3:-

##### Input:

Sequence Length = 73, including the Start Of File and End Of File delimeters.

![Input](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_input_2.png)

##### Output:

![Output](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_output_2.png)

#### 1.2. Till 20,000 epochs on Cross Entropy Loss.

##### Input:

Sequence Length = 90, including the Start Of File and End Of File delimeters.

![Input](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_input_3.png)

##### Output:

![Output](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_output_3.png)

##### Error incurred

![Error](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_3_error.png)

##### Memory Matrix for this input after last timestep:

![Memory Matrix](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/COPY%20TASK/Till%2010000%20epochs/test_output_3_Memory_Matrix.png)

#### Results on more tasks to follow soon...

### 2. Associative Recall Task

Training the Associative Recall Model for 158,000 episodes on randomized item numbers between 2 and 6 yields the following results:

#### Input 

![Input](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/input_2.png)

#### Output from NTM

![Output](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/output_2.png)

#### Write Weighing while Reading over time

![RWeights](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/WriteHead_Reading_1.png)

#### Memory Matrix compared with Read Vectors while Writing

![MM](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/MMduringWriting.png) 
![RV while W](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/ReadVectorsWhileWriting.png)

#### Memory Matrix Evolution over Time

![MM Evolution](https://github.com/WhenDustSettles/Neural-Turing-Machine/blob/master/RESULTS/ASSOCIATIVE%20RECALL%20TASK/MM_Evo.gif)

## Progress Timeline:

1. Wed, Jan 15:-
+ Completed the NTMCell Implementation along with various Vector Generation Tasks.
+ Also tested it's result with dynamic_RNN, observed some NaN values in the result, was fixed by initializing states by a considerably low (0.5 in this case) value.

2. Sun, Jan 19:-
+ Added *sigmoid* layer on *Heads_w_t* which produced much better results on one time step passes (not the training)
+ Random Initialization works well now too
+ In the process of finalizing the training schedule.

3. Fri, Jan 31:-
+ First Complete version, added Inputs Generator for Copy Task and some minor bug fixes.
+ One still needs to train this though, there maybe some problems during training which one needs to solve.

4. Sun, Feb 2:-
+ Training with Cross Entropy Loss Function proved to be difficult as loss seem to be stuck somewhere between 0.4 - 0.55
+ *Using Huber Loss Function seem to generate much better results, as loss seem to decrease linearly from 1.2 to 0.6 on max sequence length in about 10,000 epochs, after 1 injection of randomized initial states while preserving the weights.

5. Wed, Feb 6:-
+ More careful analysis brought some more subtle bugs, which were holding back the generalisation of the model, removing those increases generalization much better with Cross Entropy Loss now.

6. Sun, Feb 16:-
+ Added Associative Recall Task.

