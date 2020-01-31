import tensorflow as tf
import numpy as np

def Input_Generator(output_dim = 8, timesteps = 4, batch_size = 40,):
    
    '''
    Creates a whole sheet of shape [batch_size, 2*timesteps + 1, output_dim + 1] where op is cut and pasted part of ip to the other side
    '''
    
    inputs = (np.random.uniform(size=[batch_size,timesteps,output_dim]) > 0.5).astype(int)
    ip_with_bottom_delim = tf.concat([inputs, tf.zeros([batch_size,timesteps,2])], axis = 2)
    a = np.zeros([batch_size,1,output_dim+2])
    b = np.zeros([batch_size,1,output_dim+2])
    b[:,:,-2] = np.ones_like(a[:,:,0])
    a[:,:,-1] = np.ones_like(a[:,:,0])
    ip_with_eof = tf.concat([tf.convert_to_tensor(b, dtype = tf.float32),ip_with_bottom_delim,tf.convert_to_tensor(a, dtype = tf.float32)], axis = 1)
    final_input = ip_with_eof
    final_output = ip_with_bottom_delim
    #final_input = tf.concat([ip_with_eof,tf.zeros([batch_size,timesteps,output_dim+2])], axis = 1)
    #copied_output = tf.concat([tf.zeros([batch_size,timesteps + 2, output_dim + 2]), ip_with_bottom_delim], axis = 1)

    return final_input, final_output