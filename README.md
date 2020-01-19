# Neural-Turing-Machine
An attempt at Understanding and Implementation of the corresponding Paper by Alex Graves et al.

Progress Timeline:

1. Wed, Jan 15:-
+ Completed the NTMCell Implementation along with various Vector Generation Tasks.
+ __Also tested it's result with dynamic_RNN, observed some NaN values in the result, was fixed by Initializing      States by a considerably low (0.5 in this case) value.__

2. Sun, Jan 19:-
+ Added *sigmoid* layer on *Heads_w_t* which produced much better results on one time step passes (not the training)
+ In the process of finalizing the training schedule.
