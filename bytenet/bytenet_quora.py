import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.python.ops.nn_ops import atrous_conv2d

from bytenet import ops
from bytenet.model_config import translator_config as config
from arg_getter import FLAGS
from model import BasicQuoraModel
ln = tf.contrib.layers.layer_norm
class BytenetQuora():
    def __init__(self,s1,s2,l1,l2,labels,gs):
        mask,mask1,mask2 = self.build_mask_per_batch(l1,l2)
        self.logits_op= self.quick_encode(s1, s2,mask,mask1,mask2)
        # s1_enc,s2_enc = self.encode_sentances(s1,s2)
        #
        # self.logits_op = self.get_logits(s1_enc,s2_enc)
        self.loss_op = BasicQuoraModel.loss(self.logits_op,labels)
        self.train_op = BasicQuoraModel.optimizer(self.loss_op,gs)
        BasicQuoraModel.make_gradient_summaries(self.loss_op)
        self.metrics_op =BasicQuoraModel.metrics(logits=self.logits_op,labels=labels)
        self.summaries = tf.summary.merge_all()

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
        mask1 = tf.expand_dims(tf.sequence_mask(l1, FLAGS.max_len, dtype=tf.float32),axis=2)
        mask2 = tf.expand_dims(tf.sequence_mask(l2, FLAGS.max_len, dtype=tf.float32),axis=2)
        big_mask = tf.sequence_mask(greater_len, FLAGS.max_len, dtype=tf.float32,)
        mask = tf.expand_dims(tf.stack([big_mask, big_mask], axis=1), axis=3)
        return mask,mask1,mask2
    def quick_encode(self,s1,s2,mask,mask1,mask2):
        with tf.variable_scope("model", initializer=xavier_initializer()):
            combined, s1_emb = self.emebdd_and_stack_inputs(s1, s2,mask1,mask2)
            combined = tf.multiply(mask, combined) # Zero out what should be zero
            height =2
            width =2
            size = s1_emb.shape[2]
            next_input = combined
            next_input = self.build_dilations(height, next_input, size, width,mask)
            next_input = self.convolve_reduce(height,next_input,size)
            logits = self.reduceded_vons_to_logits(next_input)
        return logits

    def emebdd_and_stack_inputs(self, s1, s2,mask1,mask2):
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, FLAGS.hidden1],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s1_emb = tf.multiply(mask1,s1_emb)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
        s2_emb = tf.multiply(mask1, s2_emb)
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

    def build_dilations(self, height, next_input, size, width,mask):
        inputs = [next_input]

        i =0
        while 2**i < FLAGS.max_len: #9 dilation si receptive field of 2^10 = 1024
            dilation_rate = 2 ** i
            with tf.variable_scope("dilated_{}".format(i)):
                if dilation_rate <=2:
                    height =2
                else:
                    height =1

                filter_ = tf.get_variable(name="conv_filter_{}".format(i), shape=[height, width, size, size])
                res = atrous_conv2d(next_input, filters=filter_, rate=dilation_rate, padding="SAME")
                res =tf.multiply(mask, res)
                inputs.append(res)
                layer_weights = tf.nn.softmax([tf.get_variable("layer_weight_{}".format(i),shape=[],dtype=tf.float32) for i in range(len(inputs))])
                tf.summary.histogram("layer_weight_{i}",layer_weights)
                layer_weights = tf.unstack(layer_weights)
                next_input = ln(tf.nn.relu(sum([w*x  for w,x in zip(layer_weights,inputs)])))
                tf.nn.dropout(next_input,FLAGS.dropout_keep_prob)
                tf.summary.histogram(name="activation", values=next_input)
                i+=1
        return next_input
