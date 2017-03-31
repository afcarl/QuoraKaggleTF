import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.python.ops.losses.util import get_regularization_losses
from tensorflow.python.ops.nn_ops import atrous_conv2d
import numpy as np
from bytenet import ops
from bytenet.model_config import translator_config as config
from arg_getter import FLAGS
from model import BasicQuoraModel
ln = tf.contrib.layers.layer_norm
class BytenetQuora():
    def __init__(self,gs):
        self.s1 = tf.placeholder(shape=[None,None],dtype=tf.int32,name="s1_pl")
        self.s2 = tf.placeholder(shape=[None, None], dtype=tf.int32,name="s2_pl")
        self.l1 = tf.placeholder(shape=[None], dtype=tf.int32,name="l1_pl")
        self.l2 = tf.placeholder(shape=[None], dtype=tf.int32,name="l2_pl")
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32,name="labels_pl")
        self.dropout_pl = tf.placeholder(shape=[],name="dropout_pl",dtype=tf.float32)
        concat_lens = tf.stack([self.l1, self.l2], 1)
        greater_len = tf.reduce_max(concat_lens, 1)
        max_len = tf.reduce_max(greater_len)
        batch_size = tf.unstack(tf.shape(self.s1))[0]
        s1 = tf.slice(self.s1,[0,0],[batch_size,max_len])
        s2 = tf.slice(self.s2, [0,0],[batch_size,max_len])
        mask,mask1,mask2 = self.build_mask_per_batch(self.l1,self.l2)
        self.logits_op= self.quick_encode(s1, s2,mask,mask1,mask2)
        self.loss_op = BasicQuoraModel.loss(self.logits_op,self.labels)
        reg_loss =sum(get_regularization_losses())
        self.train_op = BasicQuoraModel.optimizer(self.loss_op+reg_loss,gs)
        self.probs = tf.unstack(tf.nn.softmax(self.logits_op), axis=1)[1]
        gradient_summaries = BasicQuoraModel.make_gradient_summaries(self.loss_op)
        metrics,self.metrics_update_op =BasicQuoraModel.metrics(logits=self.logits_op,labels=self.labels)
        metrics.append(tf.summary.scalar("loss", self.loss_op))
        metrics.append(tf.summary.scalar("reg_loss", reg_loss))
        metrics.append(tf.summary.scalar("total_loss", self.loss_op+reg_loss))
        self.val_summaries = tf.summary.merge(metrics)
        self.train_summaries = tf.summary.merge(metrics+gradient_summaries)

        self.gs = gs

    @staticmethod
    def build_mask_per_batch(l1,l2):
        '''
        given lengths of two sentances l1 and l2 returns a mask "tensor" of shape
                 [batch_size,2,sequence_length,1] which can be broadcasted with
                 tf.multiply
                 c = tf.multiply(mask,sentances)

                 actually, returns 3 masks, a mask of the larger sequence lengths and one for each sequence.
                 We zero the intiial embedding idnivudallu and zero the convolutions with the longer of the two
                 to get the dynamics when one is longer
        '''
        concat_lens = tf.stack([l1, l2], 1)
        greater_len= tf.reduce_max(concat_lens, 1)
        max_len = tf.reduce_max(greater_len)
        mask1 = tf.expand_dims(tf.sequence_mask(l1,maxlen=max_len,  dtype=tf.float32),axis=2)
        mask2 = tf.expand_dims(tf.sequence_mask(l2,maxlen=max_len, dtype=tf.float32),axis=2)
        big_mask = tf.sequence_mask(greater_len, dtype=tf.float32,)
        mask = tf.expand_dims(tf.stack([big_mask, big_mask], axis=1), axis=3)
        return mask,mask1,mask2
    def quick_encode(self,s1,s2,mask,mask1,mask2):
        with tf.variable_scope("model", initializer=xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(0.2)):
            combined, s1_emb = self.emebdd_and_stack_inputs(s1, s2,mask1,mask2)
            combined = tf.multiply(mask, combined) # Zero out what should be zero
            height =2
            width =2
            size = s1_emb.shape[2]
            next_input = combined
            next_input = self.build_dilations(height, next_input, size, width,mask)
            next_input = self.convolve_reduce(height,next_input,size)
            logits = self.to_logits(next_input)
        return logits

    def emebdd_and_stack_inputs(self, s1, s2,mask1,mask2):
        embedding_size = int(np.log10(FLAGS.vocab_size)+1)
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, embedding_size],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s1_emb = tf.multiply(mask1,s1_emb)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
        s2_emb = tf.multiply(mask1, s2_emb)
        filters =tf.get_variable(name="input_conv",shape=[3,embedding_size,FLAGS.hidden1,])
        s1_emb = tf.nn.conv1d(s1_emb,filters=filters,stride=1,padding="SAME")
        s2_emb = tf.nn.conv1d(s2_emb, filters=filters, stride=1, padding="SAME")
        combined = tf.stack([s1_emb, s2_emb], 1)  # [batch_size,2,seq_len,hidden]
        return combined, s1_emb

    def to_logits(self, next_input):
        logits = tf.contrib.layers.linear(next_input, num_outputs=2)
        return logits

    def convolve_reduce(self, height, next_input, size):
        i = 0
        width = 5
        #TODO need to take means only up to last length
        batch_size,_,sequence_length,hidden_size = tf.unstack(tf.shape(next_input))
        next_input = tf.reshape(next_input,shape=[batch_size,FLAGS.hidden2,-1])
        means = tf.reduce_max(next_input,axis=2) #mean is [batch_size,2,size]
        next_input = ln(tf.nn.relu(means))
        # i =0
        # next_input = tf.expand_dims(means,axis=1)
        # next_input = tf.expand_dims(next_input, axis=1)
        # activations =[]
        # while next_input.shape.as_list()[2] >1:
        #
        #     with tf.name_scope("reduce_conv_{}".format(i)):
        #         filters =tf.get_variable(name="reduce_filter_{}".format(i),shape=[1,FLAGS.hidden2//10,FLAGS.hidden2,FLAGS.hidden2,])
        #         next_input = tf.nn.conv2d(next_input,filter=filters,strides=[1,1,1,1],padding="VALID")
        #         next_input =ln(tf.nn.relu(next_input))
        #         activations.append(next_input)
        #
        #     i+=1
        # for num,activ in enumerate(activations):
        #     tf.summary.histogram("reduce_conv_{}".format(num),activ)

        return next_input

    def build_dilations(self, height, next_input, size, width,mask):
        inputs = [next_input]

        i =0
        rates = [1,1,2,2,1,4,1,2,8,1,2,1,4,2,1,2,4,1]
        for dilation_rate in rates:

            with tf.variable_scope("dilated_{}".format(i)):
                if dilation_rate <=2:
                    height =2
                else:
                    height =1
                width = 3 + (i%2)
                filter_ = tf.get_variable(name="conv_filter_{}".format(i), shape=[height, width, size, size],)
                res = atrous_conv2d(next_input, filters=filter_, rate=dilation_rate, padding="SAME")
                res =tf.multiply(mask, res)
                inputs.append(res)
                next_input = ln(tf.nn.relu(sum(inputs)))
                tf.nn.dropout(next_input,self.dropout_pl)
                tf.summary.histogram(name="activation", values=next_input)
                i+=1
        return next_input
