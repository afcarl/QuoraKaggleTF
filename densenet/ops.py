'''
DenseNet for text with 1dconvolutions
'''
import tensorflow as tf
from tensorflow.contrib.layers import linear, fully_connected,layer_norm as ln
from arg_getter import FLAGS


def avg_pool1d(_input, k,height=1):
    ksize = [1, height, k, 1]
    strides = [1, 1, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output


def composite_function(_input,outSize,width):
    with tf.variable_scope("composite_function"):
        output = ln(_input)
        output = tf.nn.relu(output)
        output = conv2d(output, outSize, width=width)
        output = tf.nn.dropout(output,keep_prob=FLAGS.dropout_keep_prob)
    return output

def conv2d(x, outSize, width):
    inputSize = x.get_shape()[-1]
    filter_ = tf.get_variable("conv_filter", shape=[2,width, inputSize, outSize])
    convolved = tf.nn.conv2d(x,filter=filter_,strides=[1,1,1,1],padding="SAME")
    convolved =ln(convolved)
    return convolved
def bottleneck(_input, growthRate):
    '''
    Per the paper, each bottlneck outputs 4k feature size where k is the growth rate of the network.
    :return:
    '''
    output = ln(_input)
    output =tf.nn.relu(output)
    outSize = growthRate * 4
    output = conv2d(output, outSize, width=1)
    return output

def addInternalLayer(_input,growth_rate):

    compOut = composite_function(_input,growth_rate,width=3)
    output = tf.concat([_input,compOut],axis=3)
    return output

def makeBlock(_input,growth_rate,num_layers,bottle=True):
    output = _input
    for layer in range(num_layers):
        with tf.variable_scope("layer_{}".format(layer)):
            output = addInternalLayer(output,growth_rate)
    if bottle:
        output= bottleneck(output, growth_rate)
    return output

def transition_layer(_input,reduction=1):
    outSize = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(_input,outSize,width=1)
    output = avg_pool1d(output,2)
    return output

def transition_to_vector(_input):
    '''
    Transforms the last block into a single vector by avg_pooling
    '''
    output =ln(_input)
    output = tf.nn.relu(output)
    last_pool_kernel = int(output.get_shape()[-2])
    output = avg_pool1d(output,last_pool_kernel,height=2)
    output = tf.squeeze(output,axis=1)
    output = tf.squeeze(output, axis=1)
    return output



