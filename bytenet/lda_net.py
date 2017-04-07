'''
Idea is to enocde each sentance with dilated convolutions as before.
Then have it predict both if they are the same and the lda topic distribution.
Reasoning is that the topic distribution is a decent proxy to the actual problem and encourages richer gradients
'''

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.ops.nn_ops import atrous_conv2d
import numpy as np
from arg_getter import FLAGS
from model import BasicQuoraModel
ln = tf.contrib.layers.layer_norm
class LDANet():
    def __init__(self,gs):
        self.s1 = tf.placeholder(shape=[None,FLAGS.max_len],dtype=tf.int32,name="s1_pl")
        self.s2 = tf.placeholder(shape=[None, FLAGS.max_len], dtype=tf.int32,name="s2_pl")
        self.l1 = tf.placeholder(shape=[None], dtype=tf.int32,name="l1_pl")
        self.l2 = tf.placeholder(shape=[None], dtype=tf.int32,name="l2_pl")
        self.lda1 = tf.placeholder(shape=[None,FLAGS.num_topics], dtype=tf.float32, name="lda1_pl")
        self.lda2 = tf.placeholder(shape=[None,FLAGS.num_topics], dtype=tf.float32, name="lda2_pl")
        self.use_lda_loss = tf.placeholder(shape=[],dtype=tf.float32,name="use_lda_pl")
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32,name="labels_pl")
        self.dropout_pl = tf.placeholder(shape=[],name="dropout_pl",dtype=tf.float32)

        mask, mask1, mask2, s1, s2 = self.prepare_sentances_and_masks()
        combined,size = self.emebdd_and_stack_inputs(s1,s2,mask,mask1,mask2)
        encoded_sentances = self.encode_sentances(combined,size,)
        #sim_target,lda_target = self.convolve_reduce(encoded_sentances)
        sim_target, l1,l2 = self.reshape_reduce(encoded_sentances)
        sim_probs = tf.unstack(tf.nn.softmax(sim_target), axis=1)[1]
        #sim_loss, sim_probs,prelogits =self.similarity_flow(sim_target,labels=self.labels)
        sim_loss = self.similarity_loss(sim_target,self.labels)
        lda_loss = self.lda_loss(l1,l2,self.lda1,self.lda2)
        #lda_loss = self.lda_flow(lda_target, self.lda1, self.lda2)
        summaries = []
        total_loss = sim_loss+self.use_lda_loss*lda_loss
        tf.summary.scalar("total_loss",total_loss)
        tf.summary.scalar("sim_loss", sim_loss)
        tf.summary.scalar("lda_loss", lda_loss)


        self.loss_op = total_loss
        self.train_op = BasicQuoraModel.optimizer(self.loss_op,gs)
        #self.probs = sim_probs
        metrics, self.metrics_update_op = BasicQuoraModel.metrics(probs=sim_probs, labels=self.labels)
        stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'metrics']
        self.reset_op = [tf.initialize_variables(stream_vars)]
        gradient_summaries = BasicQuoraModel.make_gradient_summaries(self.loss_op)

        self.val_summaries = tf.summary.merge_all()#tf.summary.merge(summaries)
        self.train_summaries = tf.summary.merge_all()

        self.gs = gs
    def edit_loss(self,logits,s1,s2):
        dist = tf.edit_distance(tf.SparseTensor(s1),tf.SparseTensor(s2))
        dist_pred = tf.contrib.layers.fully_conntext(logits,num_outputs=1,activation_fn=tf.nn.sigmoid)
        loss = tf.nn.l2_loss(dist-dist_pred)
        return loss
    def get_losses(self, encoded_sentances, labels, lda1, lda2,l1,l2):
        s1,s2= tf.unstack(encoded_sentances,axis=1)
        sim_logits, (lda1_pred,lda2_pred)= self.lstm_sentances(s1,s2,l1,l2)
        lda_loss = self.lda_loss(lda1_pred,lda2_pred,lda1,lda2)
        sim_loss = self.similarity_loss(sim_logits,labels=labels)
        sim_probs =tf.unstack(tf.nn.softmax(sim_logits), axis=1)[1]
        return sim_loss,lda_loss,sim_probs

        
        total_loss = self.total_loss(sim_loss, lda_loss, use_lda_loss)
        summaries.append(tf.summary.scalar("total_loss", total_loss))
        summaries.append(tf.summary.scalar("sim_loss", sim_loss))
        summaries.append(tf.summary.scalar("lda_loss", lda_loss))
        return sim_probs, total_loss,summaries

    def prepare_sentances_and_masks(self):
        concat_lens = tf.stack([self.l1, self.l2], 1)
        greater_len = tf.reduce_max(concat_lens, 1)
        max_len = tf.reduce_max(greater_len)
        batch_size = tf.unstack(tf.shape(self.s1))[0]
        s1 = tf.slice(self.s1, [0, 0], [batch_size, max_len])
        s2 = tf.slice(self.s2, [0, 0], [batch_size, max_len])
        mask, mask1, mask2 = self.build_mask_per_batch(self.l1, self.l2)
        return mask, mask1, mask2, self.s1, self.s2

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
        mask1 = tf.expand_dims(tf.sequence_mask(l1,maxlen=FLAGS.max_len,  dtype=tf.float32),axis=2)
        mask2 = tf.expand_dims(tf.sequence_mask(l2,maxlen=FLAGS.max_len, dtype=tf.float32),axis=2)
        big_mask = tf.sequence_mask(greater_len,maxlen=FLAGS.max_len, dtype=tf.float32,)
        mask = tf.expand_dims(tf.stack([big_mask, big_mask], axis=1), axis=3)
        return mask,mask1,mask2

    def emebdd_and_stack_inputs(self, s1, s2,mask,mask1,mask2):
        embedding_size = int(np.log10(FLAGS.vocab_size)+1)
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, embedding_size],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s1_emb = tf.multiply(mask1,s1_emb)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
        s2_emb = tf.multiply(mask2, s2_emb)
        filters =tf.get_variable(name="input_conv",shape=[2,embedding_size,FLAGS.hidden1,])
        s1_emb = tf.nn.conv1d(s1_emb,filters=filters,stride=1,padding="SAME")
        s2_emb = tf.nn.conv1d(s2_emb, filters=filters, stride=1, padding="SAME")
        combined = tf.stack([s1_emb, s2_emb], 1)  # [batch_size,2,seq_len,hidden]
        combined = tf.multiply(mask, combined)  # Zero out what should be zero
        size = s1_emb.shape[2]
        return combined, size

    def to_logits(self, next_input):
        logits = tf.contrib.layers.linear(next_input, num_outputs=2)
        return logits

    def reshape_reduce(self,next_input):
        next_input = tf.reshape(next_input, shape=[-1, 2, FLAGS.max_len * FLAGS.hidden1])
        with tf.variable_scope("lda_reduce",initializer=xavier_initializer()) as scope:
            l1, l2 = tf.unstack(next_input, axis=1)
            l1 = ln(tf.contrib.layers.fully_connected(l1, num_outputs=FLAGS.hidden1),
                            activation_fn=tf.nn.softsign,scope=scope,)
            l1 = tf.contrib.layers.linear(l1, num_outputs=FLAGS.num_topics, scope=scope,)
            l2 = ln(tf.contrib.layers.fully_connected(l2, num_outputs=FLAGS.hidden1),
                    activation_fn=tf.nn.softsign, scope=scope, reuse=True)
            l2 = tf.contrib.layers.linear(l2, num_outputs=FLAGS.num_topics, scope=scope,reuse=True)

        with tf.variable_scope("comp_reduce",initializer=xavier_initializer()) as scope:
            next_input = tf.reshape(next_input, shape=[-1, 2*FLAGS.max_len * FLAGS.hidden1])
            sims =ln(tf.contrib.layers.fully_connected(next_input, num_outputs=FLAGS.hidden1//2),
                            activation_fn=tf.nn.softsign,scope=scope,)
            sims =tf.contrib.layers.linear(sims, num_outputs=FLAGS.num_topics, scope=scope)
        return sims,l1,l2




    def convolve_reduce(self, next_input,):
        width =4
        lda_target_size = FLAGS.num_topics
        lda_targets = None
        with tf.variable_scope("reduce",initializer=xavier_initializer()):
            layer_num =0
            while next_input.shape[2] > 1:
                if lda_targets is not None and next_input.shape[1] >1:
                    height =2
                else:
                    height = 1
                stride_width = min(width//2,next_input.shape[2])
                filter_ = tf.get_variable("reduce_filter_{}".format(layer_num),shape=[height,width,FLAGS.hidden1,FLAGS.hidden1])
                next_input =tf.nn.conv2d(next_input,filter=filter_,strides=[1,1,1,1],padding="SAME")
                next_input = tf.nn.avg_pool(next_input,[1,height,stride_width,1],strides=[1,1,stride_width,1],padding="VALID")
                next_input = ln(tf.nn.softsign(next_input))
                tf.summary.histogram(name="reduce_activation_{}".format(layer_num), values=next_input)
                if next_input.shape[2] == lda_target_size:
                    lda_targets = next_input
                layer_num +=1
        sim_targets = next_input
        return sim_targets,lda_targets


    def encode_sentances(self, next_input, size, ):
        '''
        Takes two sentances as a 2d "image" and runs dilated convolutions on each of them seperatly
        '''

        rates = [1,2,1,4,1,8,1,4,1,2,1]
        with tf.variable_scope("encoding_scope",initializer=xavier_initializer()) as scope:
            comp_sentance_conv = self.run_dilations("single", next_input, rates, size)
        return comp_sentance_conv

    def compare_sentances(self, next_input,  ):
        '''
        Takes two sentances as a 2d "image" and runs dilated convolutions on each of them seperatly
        '''
        next_input = tf.reshape(next_input,shape=[-1,FLAGS.hidden1])
        next_input =ln(tf.contrib.layers.fully_connected(next_input, num_outputs=FLAGS.hidden2),activation_fn=tf.nn.softsign)
        next_input = ln(tf.contrib.layers.fully_connected(next_input, num_outputs=FLAGS.hidden2),activation_fn=tf.nn.softsign)
        logits = tf.contrib.layers.linear(next_input, num_outputs=2)
        return logits,next_input



    def run_dilations(self, mode, next_input, rates, size):
        inputs = [next_input]
        batch_size, height_v, width_v, size = next_input.shape.as_list()
        for layer_num, dilation_rate in enumerate(rates):



            with tf.variable_scope("dilated_{}".format(layer_num),initializer=xavier_initializer()):
                if dilation_rate <=2 and mode !="single":
                    height =2
                else:
                    height =1
                width = 3
                filter_ = tf.get_variable(name="conv_filter_{}".format(layer_num), shape=[height, width, size, size],)
                res = atrous_conv2d(next_input, filters=filter_, rate=dilation_rate, padding="SAME")
                inputs.append(res)
                next_input = tf.nn.softsign(ln(sum(inputs)))
                tf.summary.histogram(name="activation", values=next_input)
                next_input =tf.nn.dropout(next_input,self.dropout_pl)
        next_input= tf.reshape(next_input, shape=[-1, height_v, width_v, size])  # Keep tensor size constant(atrous returns ?
        return next_input

    def sentances_to_lda(self,encoded_sentances):
        '''

        :param encoded_sentances: [batch_size,2,None,hidden1]
        :return:
        '''
        with tf.variable_scope("sentance_to_lda",initializer=xavier_initializer()) as scope:
            next_input = tf.reshape(encoded_sentances,shape=[-1,2,FLAGS.num_topics*FLAGS.hidden1])
            s1,s2 = tf.unstack(next_input,axis=1)
            s1 = tf.contrib.layers.fully_connected(s1,num_outputs=FLAGS.num_topics*2,scope=scope)
            s2 = tf.contrib.layers.fully_connected(s2, num_outputs=FLAGS.num_topics * 2, scope=scope,reuse=True)
        with tf.variable_scope("sentance_to_lda_logits",initializer=xavier_initializer()) as scope:
            scope.reuse_variables()
            s1 = tf.contrib.layers.linear(s1, num_outputs=FLAGS.num_topics, scope=scope)
            s2 = tf.contrib.layers.linear(s2, num_outputs=FLAGS.num_topics , scope=scope,reuse=True)
        return s1,s2
    def lda_loss(self,s1,s2,lda1,lda2):
        l1= tf.nn.softmax_cross_entropy_with_logits(labels=lda1, logits=s1)
        l2 = tf.nn.softmax_cross_entropy_with_logits(labels=lda2, logits=s2)
        loss = tf.reduce_mean(tf.stack((l1,l2)))
        return loss

    def lda_flow(self,encoded_sentances,lda1,lda2):
        s1,s2 = self.sentances_to_lda(encoded_sentances)
        lda_loss = self.lda_loss(s1,s2,lda1,lda2)
        return lda_loss
    def similarity_flow(self,encoded_sentances,labels):
        logits,pre_logits= self.compare_sentances(encoded_sentances)

        sim_loss = self.similarity_loss(logits,labels=labels)
        sim_probs =tf.unstack(tf.nn.softmax(logits), axis=1)[1]
        return sim_loss,sim_probs,pre_logits


    def similarity_loss(self,logits,labels):
        sim_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return sim_loss

    def total_loss(self,sim_loss,lda_loss,use_lda):

        total_loss = sim_loss+use_lda*lda_loss
        return total_loss


