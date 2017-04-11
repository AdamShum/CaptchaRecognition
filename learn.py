# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:29:58 2017

@author: cilab_2
"""

#a = 'a'
#a = ord(a)
#b = a+15
#print(chr(b))
import os
mnist = load_data.read_data_sets("1",\
                         "2",one_hot=False)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images,\
                                               mnist.train.labels,\
                                               mnist.test.images,\
                                               mnist.test.labels
with tf.Session(graph=graph) as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, '1/w.data')
    print ("Start optimization")
    
    for epoch in range(1):
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
            name =''
            predict = p[1][i*4:i*4+len(p[1])]
            print(predict)
            for k in range(len(p[1])):
                if(predict[k]>=10):
                    print(chr(predict[k]-10+ord('a')))
                    name = name +str(chr(predict[k]-10+ord('a')))
                else:
                    print(predict[k])
                    name = name + str(predict[k])
            name = name +'.gif'
            os.rename('1.gif',name)
            a = batch_txs[i] 

            a = a.transpose(1,0)
            plt.imshow(a)
            
print ("Optimization Finished.")
                                        