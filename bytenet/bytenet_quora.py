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
        self.options = get_model_options()
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
        w_source_embedding = tf.get_variable('z_source_embedding',
                                             [FLAGS.vocab_size, FLAGS.hidden1],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
        s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)

        height =2
        width =8
        i =1
        size = s1_emb.shape[2]
        batch,sequence,hidden = s1_emb.get_shape()
        combined = tf.stack([s1_emb, s2_emb], 1)  # [batch_size,2,seq_len,hidden]
        next_input = combined
        for i in range(3):
            j =i
            filter_ = tf.get_variable(name="conv_filter_{}".format(i), shape=[height, width, size, size])
            conv1 = tf.nn.conv2d(next_input, filter=filter_, strides=[1,1,1,1], padding="SAME")
            sigmoid2 = ln(tf.nn.sigmoid(conv1, name='enc_sigmoid2_layer{}'.format(i)))
            res =atrous_conv2d(sigmoid2,filters=filter_,rate=2**j,padding="SAME")
            res  = ln(tf.nn.sigmoid(res, name='enc_sigmoid2_layer{}'.format(i)))
            next_input = res+next_input
        i =0
        while next_input.shape[2] >width:

            filter_ = tf.get_variable(name="conv_shrink_filter_{}".format(i), shape=[1, width, size, size])
            conv1 = tf.nn.conv2d(next_input, filter=filter_, strides=[1, 1, 1, 1], padding="VALID")
            sigmoidd = ln(tf.nn.sigmoid(conv1, name="sigmoid_{}".format(i)))
            next_input = tf.nn.max_pool(sigmoidd, ksize=[1, height, width, 1], strides=[1, height, width, 1], padding="SAME")
            i+=1
        final = next_input
        final = tf.squeeze(final)
        final = tf.reshape(final,[FLAGS.batch_size,-1])
        logits = ln(tf.nn.sigmoid(final))
        logits = tf.contrib.layers.linear(logits, num_outputs=2)
        return logits
    def encode_sentances(self,s1,s2):

        with tf.variable_scope("model", initializer=xavier_initializer()):
            w_source_embedding = tf.get_variable('w_source_embedding',
                                                 [self.options['n_source_quant'], 2 * self.options['residual_channels']],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            s1_emb = tf.nn.embedding_lookup(w_source_embedding, s1)
            s2_emb = tf.nn.embedding_lookup(w_source_embedding, s2)
            with tf.variable_scope("encoder") as scope:
                s1_enc =self.encoder(s1_emb,)
                scope.reuse_variables()
                s2_enc = self.encoder(s2_emb)
        return s1_enc,s2_enc
    def get_logits(self,s1,s2):
        with tf.variable_scope("model", initializer=xavier_initializer()):
            combined = tf.stack([s1,s2],1) #[batch_size,2,seq_len,hidden]
            next_input = combined
            size = combined.shape[3]
            strides =[1,1,1,1]
            i=0
            width = 4
            while next_input.shape[2] > width :
                with tf.variable_scope("squeeze_layer_{}".format(i)):
                    height = 2 if i==0 else 1
                    filter_ =tf.get_variable(name="conv_filter_{}".format(i), shape=[height,width,size,size])
                    conv = tf.nn.conv2d(input=next_input,filter=filter_,strides=strides,name="conv_op_{}".format(i),padding="VALID")
                    sigmoidd = tf.nn.sigmoid(conv,name="sigmoid_{}".format(i))
                    sigmoidd = tf.nn.max_pool(sigmoidd,ksize=[1,height,width,1],strides=[1,height,width,1],padding="SAME")
                    sigmoidd = tf.contrib.layers.layer_norm(sigmoidd)
                    next_input = sigmoidd
                    i+=1
            logits = tf.squeeze(next_input)
            logits =tf.contrib.layers.linear(logits,num_outputs=2)
        return logits






    def encode_layer(self,input_, dilation, layer_no, last_layer=False):
        with tf.variable_scope("enc_layer_{}".format(layer_no)):
            options = get_model_options()
            sigmoid1 = ln(tf.nn.sigmoid(input_, name='enc_sigmoid1_layer{}'.format(layer_no)))
            conv1 = ops.conv1d(sigmoid1, options['residual_channels'], name='enc_conv1d_1_layer{}'.format(layer_no))
            #conv1 = tf.matmul(conv1, self.source_masked_d)
            sigmoid2 = ln(tf.nn.sigmoid(conv1, name='enc_sigmoid2_layer{}'.format(layer_no)))
            dilated_conv = ops.dilated_conv1d(sigmoid2, options['residual_channels'],
                                              dilation, options['encoder_filter_width'],
                                              causal=False,
                                              name="enc_dilated_conv_layer{}".format(layer_no)
                                              )
            #dilated_conv = tf.matmul(dilated_conv, self.source_masked_d)
            sigmoid3 = ln(tf.nn.sigmoid(dilated_conv, name='enc_sigmoid3_layer{}'.format(layer_no)))
            conv2 = ops.conv1d(sigmoid3, 2 * options['residual_channels'], name='enc_conv1d_2_layer{}'.format(layer_no))
            return input_ + conv2
    def encoder(self, input_):
        options = get_model_options()
        curr_input = input_
        for layer_no, dilation in enumerate(options['encoder_dilations']):
            layer_output = self.encode_layer(curr_input, dilation, layer_no)

            # ENCODE ONLY TILL THE INPUT LENGTH, conditioning should be 0 beyond that
            #layer_output = tf.matmul(layer_output, self.source_masked, name='layer_{}_output'.format(layer_no))

            curr_input = layer_output

        # TO BE CONCATENATED WITH TARGET EMBEDDING
        processed_output = tf.nn.sigmoid(ops.conv1d(tf.nn.sigmoid(layer_output),
                                                 options['residual_channels'],
                                                 name='encoder_post_processing'))
        return processed_output

def get_model_options():
    model_options = {
        'n_source_quant': FLAGS.vocab_size,
        'n_target_quant': FLAGS.vocab_size,
        'residual_channels': config['residual_channels'],
        'decoder_dilations': config['decoder_dilations'],
        'encoder_dilations': config['encoder_dilations'],
        'sample_size': 10,
        'decoder_filter_width': config['decoder_filter_width'],
        'encoder_filter_width': config['encoder_filter_width'],
        'source_mask_chars': 0,
        'target_mask_chars': 0,
    }
    return model_options

# s = tf.ones(shape=[3,150],dtype=tf.int32)
# BasicQuoraModel(s,s,s)
# total_parameters = 0
# for variable in tf.trainable_variables():
#     # shape is an array of tf.Dimension
#     shape = variable.get_shape()
#     print(shape)
#     print(len(shape))
#     variable_parametes = 1
#     for dim in shape:
#         print(dim)
#         variable_parametes *= dim.value
#     print(variable_parametes)
#     total_parameters += variable_parametes
# print(total_parameters)
