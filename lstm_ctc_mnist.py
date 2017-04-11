
# Import packages
from six.moves import xrange 
import tensorflow as tf
import numpy as np
import load_data
import matplotlib.pyplot as plt  

print ("Packages imported")

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where 
                         each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*(seq[0].shape[0]), xrange((seq[0].shape[0]))))
        #print "length is :   ",seq[0].shape[0],"  seq is:   ",seq[0][0]," seq type is: ",type(seq[0][0])
        #print "seq[0][0] is :   ",seq[0][0]
        #print "seq[0][1] is :   ",seq[0][1]
        values.extend([seq[0][0]])
        values.extend([seq[0][1]])
        values.extend([seq[0][2]])
        values.extend([seq[0][3]]) 

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    #print "indices: ",indices
    #print "values : ",values
    #print "shape  : ",shape

    return indices, values, shape

def _RNN(_X,batch_size, _W, _b,_nsteps, _name):

    X = tf.reshape(_X, [-1, diminput])

    X_in = tf.matmul(X, weights['hidden']) + biases['hidden']

    X_in = tf.reshape(X_in, [-1, nsteps, dimhidden])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(dimhidden)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

 

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results=[]
    for i in range(len(outputs)) : 
        result = tf.matmul(outputs[i], weights['out']) + biases['out']    # shape = (128, 10)
        results.append(result)
    return results
    
  

# Load MNIST, our beloved friend
mnist = load_data.read_data_sets("1",\
                         "2",one_hot=False)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images,\
                                               mnist.train.labels,\
                                               mnist.test.images,\
                                               mnist.test.labels

nclasses = 36

print ("MNIST loaded")

# Training params
training_epochs =  4000
batch_size      =  1
display_step    =  1
#learning_rate   =  0.001
learning_rate   = 0.001


# Recurrent neural network params
diminput = 27
dimhidden = 100
# here we add the blank label
dimoutput = nclasses+1
nsteps = 72

graph = tf.Graph()
with graph.as_default():
    weights = {
        'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
        'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([dimhidden])),
        'out': tf.Variable(tf.random_normal([dimoutput]))
    }


    #**************************************************
    # will be used in CTC_LOSS
    #x = tf.placeholder(tf.float32, [None, nsteps, diminput])
    x = tf.placeholder(tf.float32, [batch_size, nsteps, diminput])
    istate = tf.placeholder(tf.float32, [batch_size, 2*dimhidden]) #state & cell => 2x n_hidden
    #istate = tf.placeholder(tf.float32, [None, 2*dimhidden]) #state & cell => 2x n_hidden
    #y  = tf.placeholder("float",[None,dimoutput])
    y = tf.sparse_placeholder(tf.int32)
    # 1d array of size [batch_size]
    # Seq len indicates the quantity of true data in the input, since when working with batches we have to pad with zeros to fit the input in a matrix
    seq_len = tf.placeholder(tf.int32, [None])

    myrnn = _RNN(x,batch_size, weights, biases,nsteps, 'basic')
    pred = myrnn
    #**************************************************
    # we add ctc module

    loss = tf.nn.ctc_loss(y,pred, seq_len)

    cost = tf.reduce_mean(loss)
    #cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

    # Adam Optimizer
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #Decode the best path
    decoded, log_prob = tf.nn.ctc_greedy_decoder(pred, seq_len)
    accr = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))
    #accr  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
    init  = tf.global_variables_initializer()
    print ("Network Ready!")


with tf.Session(graph=graph) as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, '1/w.data')
    print ("Start optimization")
    
    for epoch in range(1):
#        avg_cost = 0.
#        total_batch = int(mnist.train.num_examples/batch_size)+20
#        # Loop over all batches
#        for i in range(total_batch):
#            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
#            # Fit training using batch data
#            #feed_dict={x: batch_xs, y: sparse_tuple_from([[value] for value in batch_ys]),\
#            #                             istate: np.zeros((batch_size, 2*dimhidden)), \
#            #                             seq_len: [nsteps for _ in xrange(batch_size)]}
#            feed_dict={x: batch_xs, y: sparse_tuple_from([[value] for value in batch_ys]),\
#                                         seq_len: [nsteps for _ in xrange(batch_size)]} 
#
#            a = sess.run(pred, feed_dict=feed_dict)
#                              
#            _  = sess.run(optm, feed_dict=feed_dict)                              
#            batch_cost = sess.run(cost, feed_dict=feed_dict)
#            # Compute average loss
#            avg_cost += batch_cost*batch_size
#            #print "COST_pred shape is :",pred.shapecmap=plt.cm.jet
#        avg_cost /= len(trainimgs)
#        # Display logs per epoch step
#        if epoch % display_step == 0:
#            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
#
#
#            train_acc = sess.run(accr, feed_dict=feed_dict)
#            print ("    Training    label    error   rate:   %.3f" % (train_acc))
#            #testimgs = testimgs.reshape((ntest, nsteps, diminput))
            batch_txs,batch_tys = mnist.test.next_batch(batch_size)
            batch_txs = batch_txs.reshape((batch_size,nsteps,diminput))

            feed_dict={x:batch_txs, y: sparse_tuple_from([[value] for value in batch_tys]), \
                                 seq_len: [nsteps for _ in xrange(batch_size)]}
            test_acc = sess.run(accr, feed_dict=feed_dict)
            print (" Test label error rate: %.3f" % (test_acc))
#            print("                   batch_ys is :  "+str(batch_tys[0])+"  "\
#                                    +str(batch_tys[1])\
#                                +"  "+str(batch_tys[2]))
            p = sess.run(decoded[0],feed_dict=feed_dict)
            print("prediction is :            ")
            print(len(p[1])) 
            print(p[1]) 
            
            
            i = 0
            predict = p[1][i*4:i*4+4]
            print(predict)
            for k in range(4):
                if(predict[k]>=10):
                    print(chr(predict[k]-10+ord('a')))
                else:
                    print(predict[k])
            
            a = batch_txs[i] 

            a = a.transpose(1,0)
            plt.imshow(a)
            
print ("Optimization Finished.")
