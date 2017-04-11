import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.python.ops.losses.util import get_regularization_losses
from tensorflow.python.ops.nn_ops import atrous_conv2d
import numpy as np
from bytenet import ops
from bytenet.model_config import translator_config as config
from arg_getter import FLAGS
from model import BasicQuoraModel
ln = tf.contrib.layers.layer_norm
from densenet.densenet_encoder import DenseNetEncoder
class DenseNetQuora():
    def __init__(self,gs):
        self.s1 = tf.placeholder(shape=[None,FLAGS.max_len],dtype=tf.int32,name="s1_pl")
        self.s2 = tf.placeholder(shape=[None, FLAGS.max_len], dtype=tf.int32,name="s2_pl")
        self.l1 = tf.placeholder(shape=[None], dtype=tf.int32,name="l1_pl")
        self.l2 = tf.placeholder(shape=[None], dtype=tf.int32,name="l2_pl")
        self.labels = tf.placeholder(shape=[None], dtype=tf.int32,name="labels_pl")
        self.dropout_pl = tf.placeholder(shape=[],name="dropout_pl",dtype=tf.float32)
        concat_lens = tf.stack([self.l1, self.l2], 1)
        greater_len = tf.reduce_max(concat_lens, 1)
        s1 = self.s1
        s2 = self.s2
        mask,mask1,mask2 = self.build_mask_per_batch(self.l1,self.l2)
        stacked_sentances = self.emebdd_and_stack_inputs(s1,s2,mask1,mask2)
        encoded = DenseNetEncoder(stacked_sentances,num_blocks=3,layers_per_batch=10,growth_rate=8)
        self.logits_op = self.to_logits(encoded)
        self.loss_op = BasicQuoraModel.loss(self.logits_op,self.labels)
        metrics,self.metrics_update_op =BasicQuoraModel.metrics(logits=self.logits_op,labels=self.labels)
        metrics.append(tf.summary.scalar("loss", self.loss_op))
        self.train_op = BasicQuoraModel.optimizer(self.loss_op,gs)
        self.probs = tf.unstack(tf.nn.softmax(self.logits_op), axis=1)[1]
        gradient_summaries = BasicQuoraModel.make_gradient_summaries(self.loss_op)
        self.val_summaries = tf.summary.merge(metrics)
        self.train_summaries = tf.summary.merge_all()
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
        mask1 = tf.expand_dims(tf.sequence_mask(l1,maxlen=FLAGS.max_len,  dtype=tf.float32),axis=2)
        mask2 = tf.expand_dims(tf.sequence_mask(l2,maxlen=FLAGS.max_len, dtype=tf.float32),axis=2)
        big_mask = tf.sequence_mask(greater_len, dtype=tf.float32,)
        mask = tf.expand_dims(tf.stack([big_mask, big_mask], axis=1), axis=3)
        return mask,mask1,mask2

    def emebdd_and_stack_inputs(self, s1, s2,mask1,mask2):
        embedding_size = 8#int(np.log10(FLAGS.vocab_size)+1)
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, embedding_size],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s1_emb = tf.multiply(mask1,s1_emb)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
        s2_emb = tf.multiply(mask2, s2_emb)
        combined = tf.stack([s1_emb, s2_emb], 1)  # [batch_size,2,seq_len,hidden]
        return combined

    def to_logits(self, next_input):
        logits = tf.contrib.layers.linear(next_input, num_outputs=2)
        return logits


