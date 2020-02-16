import tensorflow as tf
import NTMCell

class NTM_CopyTask_Model(tf.keras.Model):
    
    def __init__(self, batch_size, output_dim,     rnn_size, memory_rows, memory_columns, num_read_heads, num_write_heads, addressing_type='LOC',
                 shift_range=tf.range(-1,2), return_all_states = False, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.cell = NTMCell.NTMCell(rnn_size, memory_rows, memory_columns, num_read_heads, num_write_heads, output_dim, addressing_type='LOC',
                 shift_range=tf.range(-1,2))
        
        self.init_state = self.cell.get_initial_state(batch_size = batch_size)
        self.return_all_states = return_all_states
    @tf.function
    def call(self,inputs):
        
        
        
        
        timesteps = inputs.shape[1]
        
        #state_list = []
        #op_list = []
        #for t in range(timesteps):
        #    op, state = self.cell(inputs[:,t,:], state)
        #    op = tf.nn.sigmoid(op)
        #    
        #    state_list.append(state)
        #    op_list.append(op)

            
        #outputs = tf.concat([tf.expand_dims(op_list[i],1) for i in range(timesteps)], axis = 1)
        
        #if self.return_all_states:
        #    return outputs, state_list
        #timesteps = inputs.shape[1]
        
        #states = self.cell.get_initial_state(batch_size = self.batch_size)
        
        #for time in range(timesteps):
        #   outputs, states = self.cell(inputs[:,time,:],states)  #inputs contain the sof and eof delimeters
        
        
        #response_sheet = tf.zeros([self.batch_size,timesteps-2,self.output_dim]) #timesteps -2 because inputs contains 1 sof and 1 eof, which we do not want
        
        
        #final_response = []
        #for time in range(timesteps - 2):
        #    outputs, states = self.cell(response_sheet[:,time,:], states)
        #    final_response.append(outputs)
            
        #final_response = tf.nn.sigmoid(tf.stack(final_response, axis =1))  
        
        if self.return_all_states:
            outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,inputs,initial_state=self.init_state,)
            response_sheet = tf.zeros([self.batch_size,timesteps-2,self.output_dim]) #timesteps -2 because inputs contains 1 sof and 1 eof, which we do not want
            outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,response_sheet,initial_state=states)
        
        outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,inputs,initial_state=self.init_state)
        response_sheet = tf.zeros([self.batch_size,timesteps-2,self.output_dim]) #timesteps -2 because inputs contains 1 sof and 1 eof, which we do not want
        outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,response_sheet,initial_state=states)
        
        return tf.nn.sigmoid(outputs), states
    
    

    
class NTM_Associative_Recall_Model(tf.keras.Model):
    
    def __init__(self, batch_size, output_dim, item_len ,rnn_size, memory_rows, memory_columns, num_read_heads, num_write_heads, controller = tf.keras.layers.LSTMCell, addressing_type='LOC',
                 shift_range=tf.range(-1,2), **kwargs):
        
        #output_dim should include the two delimiters' rows
        #item_len: no. of vectors in each item
        
        super().__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.item_len = item_len
        self.controller = controller
        self.cell = NTMCell.NTMCell(rnn_size, memory_rows, memory_columns, num_read_heads, num_write_heads, output_dim, controller = self.controller ,addressing_type='LOC',
                 shift_range=tf.range(-1,2))
        
        self.init_state = self.cell.get_initial_state(batch_size = batch_size)
        
    @tf.function
    def call(self, inputs):
        
        #We'll make sure this time that we will include the history of each timesteps' states, as using tf.nn.dynamic_rnn only returns the last timesteps' states
        
        '''
        inputs whould be of shape [batch_size, timesteps, output_dim]
        '''
        
        timesteps = inputs.shape[1]
        state = self.init_state
        
        #Reading the inputs
        read_states = []
        read_outputs = []
        
        for time in range(timesteps):
            
            output, state = self.cell(inputs[:,time,:], state)
            
            read_states.append(state)
            read_outputs.append(output)
            
    
        #read_states = tf.nn.sigmoid(tf.convert_to_tensor(read_states))
        #read_outputs = tf.nn.sigmoid(tf.convert_to_tensor(read_outputs))
        
        #Response Strip for the NTM
        response_sheet = tf.zeros([self.batch_size, self.item_len, self.output_dim])
            
        #Writing the answer
        write_states = []
        write_outputs = []
        
        print(response_sheet.shape)
        for time in range(self.item_len):
            
            
            output, state = self.cell(response_sheet[:,time,:], state)
            
            write_states.append(state)
            write_outputs.append(output)
            
        #write_states = tf.nn.sigmoid(tf.convert_to_tensor(write_states))
        #write_outputs = tf.nn.sigmoid(tf.convert_to_tensor(write_outputs))
        
        cached_stuff = {
                        'While_Reading' : (read_outputs, read_states),
                        'While_Writing' : (write_outputs, write_states)
                       }
        write_outputs = tf.nn.sigmoid(tf.stack(write_outputs, axis = 1))
        
        return write_outputs, cached_stuff
        
            

    
    
class CE_Loss_Function(tf.keras.losses.Loss):

    def call(self,y_true, y_pred):
        '''
        y_true: (batch_size, timesteps*2 + 2, output_dim + 1), The input sequence copied to the right side of the 

        y_pred: (batch_size, timesteps*2 + 2, output_dim + 1), The output of the model
        '''
        return -tf.reduce_mean(y_true*tf.math.log(y_pred + 1e-8) + (1-y_true)*tf.math.log(1-y_pred + 1e-8))


class CustomLoss(tf.keras.losses.Loss):
    
    def call(self,y_true, y_pred):
        
        return tf.reduce_mean(tf.sqrt(tf.abs(y_true - y_pred)))
    
    
    

    
def HuberLoss(y_true, y_pred, delta):
    return 10*tf.reduce_mean(tf.where(tf.abs(y_true-y_pred) < delta,.5*(y_true-y_pred)**2 , delta*(tf.abs(y_true-y_pred)-0.5*delta)))
    #Result enhanced by a factor of 10 for easy interpretability
