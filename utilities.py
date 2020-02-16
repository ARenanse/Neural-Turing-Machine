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


def Associative_Recall_InpGen(batch_size = 20, item_len = 3, output_dim = 8, num_items = 2):
    
    '''
    item_len = No.of vectors in each item
    output_dim = length of each vector including the two delimiting rows
    num_items = How many no. of items to include in each input, this is expected to be a random number between 2 and 6
    
    '''
    
    #num_items amount of delimiters + 2 for query + item_len*num_items + item_len for query  
    
    query_delim = np.zeros([batch_size, 1,output_dim], dtype = np.int32)
    norm_delim = np.zeros([batch_size, 1,output_dim], dtype = np.int32)
    norm_delim[:,:,-2] = np.ones_like(norm_delim[:,:,0])
    query_delim[:,:,-1] = np.ones_like(norm_delim[:,:,0])
    
    inp = (np.random.uniform(size=[batch_size,item_len*num_items,output_dim-2]) > 0.5).astype(int)
    inp = tf.concat([inp, tf.zeros([batch_size, item_len*num_items, 2])], axis = 2)
    rand_seq_list = tf.split(inp, [item_len]*num_items, axis = 1)

    res = tf.concat([tf.concat( [norm_delim, tf.cast(rand_seq_list[i], tf.int32)] , axis = 1 ) for i in range(num_items) ], axis = 1 ) 
    
    
    #rand_ind = a list of random numbers denoting the first vector's index of the item whose next item will be recalled from memory.
    rand_ind = np.random.randint(0,num_items-1, size = [batch_size])
    ind  = ((item_len + 1)*(rand_ind) + 1)
    
    labels = []
    query = []
    for i in range(batch_size):

        index = ind[i]

        query.append(res[i, index:index + item_len, :])
        labels.append(res[i, index + item_len+1 : index + 2*item_len + 1, :])

    queries = tf.convert_to_tensor(query)
    labels = tf.convert_to_tensor(labels)
    
    queries_w_delim = tf.concat([query_delim, queries, query_delim], axis = 1)
    final = tf.cast(tf.concat([res, queries_w_delim], axis = 1),dtype=tf.float32)
    
    
    return final, labels
