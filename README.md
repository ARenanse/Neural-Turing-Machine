# Neural-Turing-Machine
An attempt at Understanding and Implementation of the corresponding Paper by Alex Graves et al.

Progress Timeline:

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



