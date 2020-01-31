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
        
        
        
        #timesteps = inputs.shape[1]
        
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
        timesteps = inputs.shape[1]
        outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,inputs,initial_state=self.init_state)
        response_sheet = tf.zeros([self.batch_size,timesteps-2,self.output_dim]) #timesteps -2 because inputs contains 1 sof and 1 eof, which we do not want
        outputs, states = tf.compat.v1.nn.dynamic_rnn(self.cell,response_sheet,initial_state=states)
        
        return tf.nn.sigmoid(outputs), states
    
    
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