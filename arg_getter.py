import tensorflow as tf
import pickle
with open('/home/tal/dev/recover/data/vocab.pkl','rb') as f:
    voc = pickle.load(f)
    vocab_size = len(voc["vocab"])

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('vocab_size', vocab_size, 'Vocab size.')
flags.DEFINE_string('data_dir', './data/','Where to get data')
flags.DEFINE_string('save_dir', './chkpoint/','Where to save checkpoints')
FLAGS = tf.app.flags.FLAGS
