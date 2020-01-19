import tensorflow as tf
import numpy as np

import Batch_Focusing
import Batch_RWV_Generation

class NTMCell(tf.keras.layers.AbstractRNNCell):
    
    def __init__(self, rnn_size, memory_rows, memory_columns, num_read_heads, num_write_heads, num_bits_per_output_vector, addressing_type = 'LOC', shift_range = tf.range(-1,2), **kwargs
                 ):
        
        super().__init__(**kwargs)
        
        
        self.rnn_size = rnn_size
        self.memory_rows = memory_rows       #The "N" or "size of memory" in Literature
        self.memory_columns = memory_columns      #The "M" or "memory vector's dimension" in Literature
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.num_bits_per_output_vector = num_bits_per_output_vector
        self.addressing_type = addressing_type
        
        if ((self.addressing_type != 'LOC') and (self.addressing_type != 'CONT')):
            raise ValueError('Incorrect Addressing Type: Allowed values are "LOC" for Location based Focusing and "CONT" for Content based Focusing.')

        
        self.shift_range = shift_range
        
        self.total_num_heads = self.num_read_heads + self.num_write_heads
        
        self.controller = tf.keras.layers.LSTMCell(self.rnn_size)
        
        self.output_dim = self.num_bits_per_output_vector #vector_dim
        
        self.total_parameters = ( 3 * self.memory_columns + 3 + len(self.shift_range) )*(self.num_write_heads + self.num_read_heads)
        
        self.PMG_Layer = tf.keras.layers.Dense(units= self.total_parameters, use_bias=True,) #PMG_Layer = Parameter Matrix GeneratingLayer CHECK INTITIALISATION OF PARAMETERS FOR IMPROVEMENT
        
        self.NTM_ouput_gen_layer = tf.keras.layers.Dense(units= self.output_dim,use_bias = True)
        
            
    def call(self, inputs, previous_states):

        '''
        inputs: shape = (Batch_size, input_features) where input_features is equal to num_bits_per_output_vector.
        previous_states: dictionary, contains: 1. controller_state (list of two matrices, one for Memory and one for Carry, both of size [Batch_size, RNN_size]),
                                               2. All_Read_vectors of size [Num_Read_Heads, Batch_size, Memory_dim(M)]
                                               3. All_Weights of size [Num_ALL_Heads, Batch_size, Memory_size(N)]
                                               4. Memory_Matrix of size (Batch_size, Memory_size(N), Memory_dim(M))

        '''

        #Since controller itself is a LSTMCell, thus it would demand a input of shape [Batch_size, features].
        #We construct a controller whose input will be of size [Batch_size, features_for_controller]
        #where features_for_controller is Num_Read_Heads * Memory_dim(M) + input_features

        All_prev_read_vectors = previous_states['All_Read_vectors']

        prev_controller_state = previous_states['controller_state']

        M_prev = previous_states['Memory_Matrix']

        w_prev = previous_states['All_Weight_vectors']
        #^Of shape [num_total_heads, batch_size, N]

        assert inputs.shape[1] == self.num_bits_per_output_vector

        controller_input = [All_prev_read_vectors[i] for i in range(All_prev_read_vectors.shape[0])]
       
        controller_input.append(inputs)

        controller_input = tf.concat(controller_input, axis = 1)

        assert controller_input.shape[1] == self.num_read_heads * self.memory_columns + inputs.shape[1]

        controller_output, controller_state = self.controller(controller_input, prev_controller_state)
        #controller_output is of the same shape as the controller_input

        Parameter_Matrix = self.PMG_Layer(controller_output)
        #Parameter_Matrix is of shape [Batch_size, self.total_parameters]

        Each_Heads_PM_list = tf.split(Parameter_Matrix, self.num_read_heads + self.num_write_heads,axis = 1)
        #Contains Each Head's Parameter matrix; is of total length self.num_read_heads + self.num_write_heads.

        All_Heads_W_list = []
        All_Heads_R_list = []

        #To get the weights for each Head in the whole Batch
        #To get the Read Vectors and Updated Memory Matrix, we assume first self.num_read_heads to be READ Heads and rest to be WRITE Heads
        for i,Head_PM in enumerate(Each_Heads_PM_list):

            k_t, beta_t, g_t, s_t, gamma_t, a_t, e_t = tf.split(Head_PM, [self.memory_columns, 1, 1, len(self.shift_range), 1, self.memory_columns, self.memory_columns], axis = 1)

            #EXPERIMENT WITH OTHER VALID COMBINATIONS OF THE BELOW USED ACTIVATIONS
            
            #For k_t:-
            k_t = tf.tanh(k_t + 1e-6)
            #For beta_t:-
            beta_t = tf.sigmoid(beta_t + 1e-6)
            #For g_t:-
            g_t = tf.sigmoid(g_t + 1e-6)
            #For s_t:-
            s_t = tf.nn.softmax(s_t + 1e-6)
            #The above s_t is one of the points where we can improve
            #For gamma_t:-
            gamma_t = tf.math.log(tf.exp(gamma_t + 1e-6) + 1 + 1e-6) + 1
            #For a_t:-
            a_t = tf.tanh(a_t + 1e-6)
            #For e_t:-
            e_t = tf.sigmoid(e_t + 1e-6)

            if self.addressing_type == 'LOC':
                Heads_w_t = tf.nn.sigmoid(Batch_Focusing.LocationFocusing( k_t, M_prev, beta_t,    g_t, w_prev[i], s_t, gamma_t,   K = None))
            elif self.addressing_type == 'CONT':
                Heads_w_t = tf.nn.sigmoid(Batch_Focusing.ContentFocusing( k_t, M_prev, beta_t, K = None))
                #^Should be of shape [batch_size,N]

            if i<self.num_read_heads:
                r_t = Batch_RWV_Generation.ReadVector(M_prev,Heads_w_t)
                All_Heads_R_list.append(r_t)
            elif i>=self.num_read_heads:
                M_prev = Batch_RWV_Generation.WriteOnMemory(M_prev,Heads_w_t,e_t,a_t)


            All_Heads_W_list.append(Heads_w_t)

        #Please Note that at this point M_prev has been updated to the new weight Matrix

        All_W_Matrix = tf.convert_to_tensor(All_Heads_W_list) #W for Weights
        #^Of shape [num_total_heads, batch_size, N]

        All_R_Matrix = tf.convert_to_tensor(All_Heads_R_list)  #R for Read         
        #^Of Shape [num_Read_Heads, batch_size, M]

        #TODO:: COMPLETE THE CONVOLUTION OPERATION IN FOCUSING AND THEN COMPLETE THIS CLASS

        NTM_output = self.NTM_ouput_gen_layer(controller_output)
        
        current_states = {
                            'All_Read_vectors' : All_R_Matrix,
                            'controller_state' : controller_state,
                            'Memory_Matrix' : M_prev,
                            'All_Weight_vectors' : All_W_Matrix
                         }
        
        return NTM_output, current_states

    
    @property
    def state_size(self):
        return {
            'controller_state' : self.controller.state_size,
            'All_Read_vectors' : tf.TensorShape((self.num_read_heads,None,self.memory_columns)),
            'All_Weight_vectors' : tf.TensorShape(((self.total_num_heads, None, self.memory_rows))),
            'Memory_Matrix' : tf.TensorShape([None,self.memory_rows,self.memory_columns])
        }

    @property
    def output_size(self):
        return self.output_dim
    
    #CHANGE INITIAL STATES TO SOME OTHER VALUES AND OBSERVE WHETHER THE MODEL IMPROVES OR NOT
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_state = {
            'controller_state': [0.5 * tf.ones((batch_size,self.rnn_size)), 2.53 * tf.ones((batch_size,self.rnn_size))],
            'All_Read_vectors': 0.5 * tf.ones((self.num_read_heads,batch_size,self.memory_columns)),
           'All_Weight_vectors': 0.5 * tf.ones((self.total_num_heads, batch_size, self.memory_rows)),
            'Memory_Matrix': 0.5 * tf.ones((batch_size,self.memory_rows,self.memory_columns))
        }
        
        #initial_state = {
        #    'controller_state': [tf.compat.v1.get_variable(name = 'controller_state_memory',shape=[batch_size, self.rnn_size], dtype = tf.float32,                                       #                         initializer=tf.random_normal_initializer(stddev = 0.5)), tf.compat.v1.get_variable(name = 'controller_state_carry',shape=[batch_size,                   #                                                                          self.rnn_size], dtype = tf.float32, initializer=tf.random_normal_initializer(stddev = 0.5))],
        #    
        #    'All_Read_vectors': tf.compat.v1.get_variable(name = 'All_Read_vectors',shape=[self.num_read_heads, batch_size, self.memory_columns], dtype = tf.float32,                   #                                                                              initializer=tf.random_normal_initializer(stddev = 0.5)),
        #    
        #    'All_Weight_vectors': tf.compat.v1.get_variable(name = 'All_Weight_vectors',shape=[self.total_num_heads, batch_size, self.memory_rows], dtype = tf.float32,                 #                                                                                  initializer=tf.random_normal_initializer(stddev = 0.5)),
        #    'Memory_Matrix': tf.compat.v1.get_variable(name = 'Memory_Matrix',shape=[batch_size, self.memory_rows, self.memory_columns], dtype = tf.float32,                             #                                                                       initializer=tf.random_normal_initializer(stddev = 0.5))
        #}
        return initial_state
