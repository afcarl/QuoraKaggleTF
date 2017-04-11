import tensorflow as tf
import pickle
with open('./data/vocab.pkl','rb') as f:
    voc = pickle.load(f)
    vocab_size = len(voc)

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_float('dropout_keep_prob', 0.9, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('max_len', 256, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('vocab_size', vocab_size, 'Vocab size.')
flags.DEFINE_string('data_dir', './data/','Where to get data')
flags.DEFINE_string('save_dir', './chkpoint/','Where to save checkpoints')
flags.DEFINE_string('mode', 'normal','set to test to check flow')
FLAGS = tf.app.flags.FLAGS
