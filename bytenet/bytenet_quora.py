import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.python.ops.nn_ops import atrous_conv2d

from bytenet import ops
from bytenet.model_config import translator_config as config
from arg_getter import FLAGS
from model import BasicQuoraModel
ln = tf.contrib.layers.layer_norm
class BytenetQuora():
    def __init__(self,s1,s2,labels,gs):
        self.logits_op= self.quick_encode(s1, s2)
        # s1_enc,s2_enc = self.encode_sentances(s1,s2)
        #
        # self.logits_op = self.get_logits(s1_enc,s2_enc)
        self.loss_op = BasicQuoraModel.loss(self.logits_op,labels)
        self.train_op = BasicQuoraModel.optimizer(self.loss_op,gs)
        BasicQuoraModel.make_gradient_summaries(self.loss_op)
        self.metrics_op =BasicQuoraModel.metrics(logits=self.logits_op,labels=labels)
        self.summaries = tf.summary.merge_all()
        

    def quick_encode(self,s1,s2):
        with tf.variable_scope("model", initializer=xavier_initializer()):
            combined, s1_emb = self.emebdd_and_stack_inputs(s1, s2)
            height =2
            width =2
            size = s1_emb.shape[2]
            next_input = combined
            next_input = self.build_dilations(height, next_input, size, width)
            next_input = self.convolve_reduce(height,next_input,size)
            logits = self.reduceded_vons_to_logits(next_input)
        return logits

    def emebdd_and_stack_inputs(self, s1, s2):
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, FLAGS.hidden1],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
        combined = tf.stack([s1_emb, s2_emb], 1)  # [batch_size,2,seq_len,hidden]
        return combined, s1_emb

    def reduceded_vons_to_logits(self, next_input):
        final = next_input
        final = tf.squeeze(final)
        final = tf.reshape(final, [FLAGS.batch_size, -1])
        logits = ln(tf.nn.sigmoid(final))
        logits = tf.contrib.layers.linear(logits, num_outputs=2)
        return logits

    def convolve_reduce(self, height, next_input, size):
        i = 0
        width = 5
        while next_input.shape[2] > width:
            if i > 0:
                height =1
            filter_ = tf.get_variable(name="conv_shrink_filter_{}".format(i), shape=[height, width, size, size])
            conv1 = tf.nn.conv2d(next_input, filter=filter_, strides=[1, 1, width, 1], padding="VALID")

            sigmoidd = ln(tf.nn.sigmoid(conv1, name="sigmoid_{}".format(i)))

            next_input = tf.nn.max_pool(sigmoidd, ksize=[1, height, 2, 1], strides=[1, height, width, 1],
                                        padding="SAME")

            i += 1
        next_input = tf.squeeze(next_input)
        return next_input

    def build_dilations(self, height, next_input, size, width):
        inputs = [next_input]
        height=1
        i =0
        for i in range(5): #5 dilation si receptive field of 2^7 = 128
            with tf.variable_scope("dilated_{}".format(i)):
                if i== 0:
                    height =2
                else:
                    height =1
                dilation_rate=2**i
                filter_ = tf.get_variable(name="conv_filter_{}".format(i), shape=[height, width, size, size])
                res = atrous_conv2d(next_input, filters=filter_, rate=dilation_rate, padding="SAME")
                inputs.append(res)
                next_input = ln(tf.nn.relu(sum(inputs)))
                tf.nn.dropout(next_input,FLAGS.dropout_keep_prob)
                tf.summary.histogram(name="activation", values=next_input)
        return next_input
