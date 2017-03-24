import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from arg_getter import FLAGS
from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormBasicLSTMCell




class BasicQuoraModel():
    def __init__(self,gs,inputs,lengths,label):
        self.s1,self.s2 = inputs
        self.l1,self.l2 = lengths
        self.label = label
        self.gs = gs
        self.logits_op = BasicQuoraModel.inference(self.s1,self.s2,self.l1,self.l2)
        self.loss_op = BasicQuoraModel.loss(self.logits_op,self.label)
        self.train_op = BasicQuoraModel.optimizer(self.loss_op,self.gs)
        BasicQuoraModel.make_gradient_summaries(self.loss_op)
        self.metrics_op =BasicQuoraModel.metrics(logits=self.logits_op,labels=label)
        self.summaries = tf.summary.merge_all()
    @staticmethod
    def metrics(logits,labels):
        with tf.name_scope("metrics"):
            probs = tf.nn.softmax(logits)
            preds = tf.arg_max(probs,1)
            precision,precision_update_op = tf.metrics.precision(labels,preds)
            tf.summary.scalar(name="precision", tensor=precision)
            recall,recall_update_op = tf.metrics.recall(labels,preds)
            tf.summary.scalar(name="recall", tensor=recall)
            accuracy,accuracy_update_op =tf.metrics.accuracy(labels,preds)
            tf.summary.scalar(name="accuracy", tensor=accuracy)
            metrics_ops = tf.group(recall_update_op,precision_update_op,accuracy_update_op)
            return metrics_ops


    @staticmethod
    def metrics_at_thresh(logits,labels):
        with tf.name_scope("metrics"):
            thresholds = [0.5,0.75,0.9,0.95]
            probs = tf.nn.softmax(logits)
            with tf.name_scope("precision"):
                precisions,update_op = tf.metrics.precision_at_thresholds(labels,probs,thresholds)
                precisions = tf.unstack(precisions)
                for prec,thresh in zip(precisions,thresholds):
                    tf.summary.scalar(name="precision@{}%".format(thresh*100),tensor=prec)
            with tf.name_scope("recall"):
                recalls,update_op = tf.metrics.recall_at_thresholds(labels,probs,thresholds)
                recalls = tf.unstack(recalls)
                for rec,thresh in zip(recalls,thresholds):
                    tf.summary.scalar(name="precision@{}%".format(thresh*100),tensor=rec)





    @staticmethod
    def make_input_summaries(l1,l2):
        with tf.name_scope("lengths"):
            tf.summary.histogram("lengths_q1",l1)
            tf.summary.histogram("lengths_q2", l2)
            tf.summary.histogram("lengths_dif", tf.abs(l1-l2))
    @staticmethod
    def make_gradient_summaries(loss):
        with tf.name_scope("gradients"):
            grads = tf.gradients(loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                if grad is None:
                    print(var.name)
                else:
                    tf.summary.histogram(var.name + '/gradient', grad)

    @staticmethod
    def prepare_sentance(sent,lengths,cell,):
        outputs, s1_sate = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=sent, dtype=tf.float32,
                                                              sequence_length=lengths)
        return tf.concat(outputs,2)

    @staticmethod
    def inference(s1,s2,l1,l2):
        with tf.variable_scope("model",initializer=xavier_initializer()):
            l1,l2,s1_lstmed, s2_lstmed = BasicQuoraModel.rnn_sentances( l1, l2, s1, s2)
            difs_state = BasicQuoraModel.rnn_compare_sentances(l1, l2, s1_lstmed, s2_lstmed)
            logits = tf.contrib.layers.linear(difs_state,num_outputs=2)
        return logits

    @staticmethod
    def rnn_compare_sentances( l1, l2, s1_lstmed, s2_lstmed):
        concat_lens = tf.stack([l1, l2], 1)
        dif_len = tf.reduce_max(concat_lens, 1)
        difs = tf.concat([s1_lstmed, s2_lstmed], 1)
        difs = tf.contrib.layers.fully_connected(difs, num_outputs=FLAGS.hidden2)
        cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=FLAGS.hidden2)
        difs_output, difs_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell2, cell_bw=cell2, inputs=difs,
                                                                  dtype=tf.float32, sequence_length=dif_len)
        difs_state = tf.concat([difs_state[0].c, difs_state[0].c], 1)
        return difs_state
    @staticmethod
    def convolve_embedded_sentances(l1, l2, s1, s2,embedding_size,stride=2):
        batch_size,seqeunce_length,_ = s1.shape
        filters_1 = tf.get_variable(name="filters_1",shape=[stride*2,embedding_size,2*embedding_size])
        filters_2 = tf.get_variable(name="filters_2", shape=[stride*2, 2*embedding_size, 4 * embedding_size])
        s1_conv = tf.nn.conv1d(s1,filters_1,stride,"VALID")
        s1_conv = tf.nn.conv1d(s1_conv, filters_2, stride, "VALID")

        s2_conv = tf.nn.conv1d(s2, filters_1, stride, "VALID")
        s2_conv = tf.nn.conv1d(s2_conv, filters_2, stride, "VALID")

        l1 = tf.clip_by_value(l1//(stride**2),1,s1_conv.shape[1])
        l2 = tf.clip_by_value(l2//(stride**2),1,s2_conv.shape[1])

        return l1,l2, s1_conv,s2_conv

    @staticmethod
    def rnn_sentances( l1, l2, s1, s2):
        with tf.variable_scope("inference", ) as scope:
            embedding_size, s1_embeded, s2_embeded = BasicQuoraModel.embed_sentances(s1, s2)
            l1,l2,s1_conv,s2_conv = BasicQuoraModel.convolve_embedded_sentances(l1,l2,s1_embeded,s2_embeded,embedding_size)
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=FLAGS.hidden1)
            s1_lstmed = BasicQuoraModel.prepare_sentance(s1_conv, l1, cell=cell,)
            scope.reuse_variables()
            s2_lstmed = BasicQuoraModel.prepare_sentance(s2_conv, l2,  cell=cell,)
        return l1,l2,s1_lstmed, s2_lstmed

    @staticmethod
    def embed_sentances(s1, s2):
        with tf.device("/cpu:0"):
            embedding_size = int(np.sqrt(FLAGS.vocab_size) + 1)
            embedding_matrix = tf.get_variable("embedding_matrix", shape=[FLAGS.vocab_size, embedding_size],
                                               dtype=tf.float32)
            s1_embeded = tf.nn.embedding_lookup(embedding_matrix, s1)
            s2_embeded = tf.nn.embedding_lookup(embedding_matrix, s2)
        return embedding_size, s1_embeded, s2_embeded

    @staticmethod
    def loss(logits,targets):
        loss = tf.losses.sparse_softmax_cross_entropy(targets,logits)
        tf.summary.scalar("loss",loss)
        return loss

    @staticmethod
    def optimizer(loss,gs):
        opt = tf.train.AdamOptimizer(0.002,)
        return opt.minimize(loss,global_step=gs)













