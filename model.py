import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_accuracy

from arg_getter import FLAGS




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
        self.probs = tf.unstack(tf.nn.softmax(self.logits_op),0)[1]

    @staticmethod
    def metrics(probs,labels):
        with tf.name_scope("metrics"):
            true_probs = probs
            thresholds = [0.51, 0.6, 0.8, 0.95]
            preds = tf.arg_max(probs,dimension=1)
            metrics = []
            recall_thresh,update_op_rec_thresh = tf.metrics.recall_at_thresholds(labels,predictions=true_probs,thresholds=thresholds)
            for tensor,thresh in zip(tf.unstack(recall_thresh),thresholds):
                metrics.append(tf.summary.scalar(name="recall @ {}%".format(thresh*100), tensor=tensor))

            precision_thresh,update_op_prec_thresh = tf.metrics.precision_at_thresholds(labels,predictions=true_probs,thresholds=thresholds)
            for tensor,thresh in zip(tf.unstack(precision_thresh),thresholds):
                metrics.append(tf.summary.scalar(name="precision @ {}%".format(thresh*100), tensor=tensor))
        # with tf.name_scope("accuracy"):
        #     accuracy,update_op_acc_thresh = tf.metrics.accuracy(labels,predictions=preds)
        #     metrics.append(tf.summary.scalar(name="accuracy @ {}%".format(thresh*100), tensor=accuracy))
        update_op_acc_thresh=tf.no_op()
        metrics_update_ops = tf.group(update_op_rec_thresh,update_op_prec_thresh,update_op_acc_thresh)
        return metrics,metrics_update_ops


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
            grad_summaries= []
            for grad, var in grads:
                if not "LayerNorm" in var.name and not "layer_weight" in var.name:
                    grad_summaries.append(tf.summary.histogram(var.name + '/gradient', grad))
            return grad_summaries

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
        BasicQuoraModel.make_input_summaries(l1,l2)
        concat_lens = tf.stack([l1, l2], 1)
        dif_len = tf.reduce_max(concat_lens, 1)
        difs = tf.concat([s1_lstmed, s2_lstmed], 1)
        difs = tf.contrib.layers.fully_connected(difs, num_outputs=FLAGS.hidden2)
        cell2 = tf.contrib.rnn.GRUCell(num_units=FLAGS.hidden2)
        difs_output, difs_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell2, cell_bw=cell2, inputs=difs,
                                                                  dtype=tf.float32, sequence_length=dif_len)
        difs_state = tf.concat([difs_state[0], difs_state[1]], 1)
        return difs_state
    @staticmethod
    def convolve_embedded_sentances(l1, l2, s1, s2,embedding_size,stride=2):



        with tf.variable_scope("conv_1") as scope:
            filters_1 = tf.get_variable(name="filters_1", shape=[stride * 2, embedding_size, 2 * embedding_size])
            s1_conv = tf.nn.conv1d(s1,filters_1,stride,"VALID")
            s2_conv = tf.nn.conv1d(s2, filters_1, stride, "VALID")
            s1_conv = tf.contrib.layers.fully_connected(s1_conv,num_outputs=2*embedding_size,scope=scope,activation_fn=tf.nn.relu)
            scope.reuse_variables()
            s2_conv = tf.contrib.layers.fully_connected(s2_conv, num_outputs=2*embedding_size, scope=scope,activation_fn=tf.nn.relu)
        with tf.variable_scope("conv_2") as scope:
            filters_2 = tf.get_variable(name="filters_2", shape=[stride * 2, 2 * embedding_size, 4 * embedding_size])
            s1_conv = tf.nn.conv1d(s1_conv, filters_2, stride, "VALID")
            s2_conv = tf.nn.conv1d(s2_conv, filters_2, stride, "VALID")
            s1_conv = tf.contrib.layers.fully_connected(s1_conv,num_outputs=4*embedding_size,scope=scope,activation_fn=tf.nn.relu)
            scope.reuse_variables()
            s2_conv = tf.contrib.layers.fully_connected(s2_conv, num_outputs=4*embedding_size, scope=scope,activation_fn=tf.nn.relu)


        l1 = tf.clip_by_value(l1//(stride**2),1,s1_conv.shape[1])
        l2 = tf.clip_by_value(l2//(stride**2),1,s2_conv.shape[1])

        return l1,l2, s1_conv,s2_conv

    @staticmethod
    def rnn_sentances( l1, l2, s1, s2):
        with tf.variable_scope("inference", ) as scope:
            embedding_size, s1_embeded, s2_embeded = BasicQuoraModel.embed_sentances(s1, s2)
            l1,l2,s1_conv,s2_conv = BasicQuoraModel.convolve_embedded_sentances(l1,l2,s1_embeded,s2_embeded,embedding_size)
            cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.hidden1)
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

        return loss

    @staticmethod
    def optimizer(loss,gs):
        num_batches = 400000/FLAGS.batch_size
        decay_steps = num_batches
        lr = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,global_step=gs,decay_rate=0.9,decay_steps=decay_steps)
        opt = tf.train.AdamOptimizer(lr,)
        return opt.minimize(loss,global_step=gs)













