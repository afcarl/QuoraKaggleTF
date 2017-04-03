import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
import numpy as np
from bytenet.bytenet_quora import BytenetQuora
import os
import pandas as pd
import time
from datetime import datetime
import pickle
class DataProvider():
    padding = [0 for i in range(1500)]
    def __init__(self,mode="normal"):
        print("Loading data")
        fname = "data.hdf" if mode=="normal" else "test_data.hdf"
        path = os.path.join(FLAGS.data_dir,fname)
        self.train = self.clean_hack(pd.read_hdf(path,key="train"))
        self.val = self.clean_hack(pd.read_hdf(path,key="val"))
        self.test = self.clean_hack(pd.read_hdf(path,key="test"))
        self.mode = mode

        print("done loading data")
    def clean_hack(self,data):
        '''
        Remove questions with None in them as a quick fix
        :param data:
        :return:
        '''
        has_bad = data.apply(lambda x:None in x.question_x or None in x.question_y,1)
        return data[has_bad == False]
    def do_batch(self,data,batch_size):
        size = batch_size
        start = 0
        end = size
        i =0
        while end < len(data):
            batch = data[start:end]
            max_len = batch.max_len.max()
            ids = batch.index.values
            q1 = np.stack(batch.question_x.apply(lambda x: DataProvider.pad_list(x, max_len)).values)
            q2 = np.stack(batch.question_y.apply(lambda x: DataProvider.pad_list(x, max_len)).values)
            l1 = batch.length_x.values
            l2 = batch.length_y.values
            labels = batch.label.values
            batch_step = (ids,q1,q2,l1,l2,labels)
            i+=1
            start+=size
            end+=size
            yield batch_step
    def train_batch(self,batch_size):
        dups = self.train[self.train.label==1]
        no_dups =self.train[self.train.label==0]
        amnt = len(dups)
        data = dups.append(no_dups.sample(n=amnt))


        return self.do_batch(data.sample(frac=1),batch_size)
    def val_batch(self,batch_size):
        return self.do_batch(self.val,batch_size)
    def test_batch(self, batch_size):
        return self.do_batch(self.test, batch_size)

    @staticmethod
    def pad_list(l,max_len):
        return np.array((l+DataProvider.padding)[:max_len])




def main(__):
    #    label, inputs, lengths = prepareInputsBatch(FLAGS.batch_size)
    train_dir = os.path.join(FLAGS.save_dir,"train","results")
    val_dir = os.path.join(FLAGS.save_dir, "val","results")
    test_dir = os.path.join(FLAGS.save_dir, "test","results")
    gs = tf.contrib.framework.get_or_create_global_step()
    model = BytenetQuora(gs)
    init = tf.global_variables_initializer()
    total_parameters = 0
    print_paramater_count(total_parameters)
    train_writer = tf.summary.FileWriter(train_dir)
    val_writer = tf.summary.FileWriter(val_dir)
    test_writer = tf.summary.FileWriter(test_dir)

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(model.loss_op)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), self._step, loss_value,
                                    examples_per_sec, sec_per_batch))

    with MonitoredTrainingSession(
            checkpoint_dir=FLAGS.save_dir,
            save_summaries_steps=0,
            hooks=[]

    ) as sess:
        sess.run(init,)
        DP =DataProvider(mode=FLAGS.mode)
        for epoch in range(FLAGS.num_epochs):
            for batch_num,batch in enumerate(DP.train_batch(FLAGS.batch_size)):
                do_train_step(batch, batch_num, model, sess, train_writer)
            if FLAGS.mode != "test" or epoch %20 ==0:
                do_val_dlow(DP, epoch, model, sess, val_writer)
            print("Starting test")
            #do_test_flow(DP, epoch, model, sess, test_writer)


def do_test_flow(DP, epoch, model, sess, test_writer):
    test_results = []
    for batch_num, batch in enumerate(DP.test_batch(FLAGS.batch_size)):
        test_results.append(do_test_step(batch, model, sess, test_writer, epoch, batch_num))
    results = pd.DataFrame(np.concatenate(test_results))
    path = os.path.join(FLAGS.save_dir, "test", "results", "results.hdf")
    results.to_hdf(path, key="{}".format(epoch))


def do_val_dlow(DP, epoch, model, sess, val_writer):
    val_results = []
    for batch_num, batch in enumerate(DP.val_batch(FLAGS.batch_size)):
        val_results.append(do_val_step(batch, model, sess, val_writer, batch_num))
    results = pd.DataFrame(np.concatenate(val_results))
    path = os.path.join(FLAGS.save_dir, "val", "results", "results.hdf")
    results.to_hdf(path, key="{}".format(epoch))


def do_train_step(batch, batch_num, model, sess, train_writer):
        ids,feed = make_feed(batch, model)
        loss_val,gs, summary = do_train_fetches(feed, model, sess)
        print("train {gs} {loss}".format(gs=gs,loss=loss_val))
        if batch_num % 10 == 0:
            train_writer.add_summary(summary, gs)


def do_val_step(batch, model, sess, val_writer,batch_num):
    ids, feed = make_feed(batch, model)
    feed[model.dropout_pl] =1
    loss_val,gs, summary,probs = do_val_fetches(feed, model, sess)
    print("Val {gs} {loss}".format(gs=gs+batch_num, loss=loss_val))
    val_writer.add_summary(summary, gs+batch_num)
    results = np.stack([ids, probs], axis=1)
    return results
def do_test_step(batch, model, sess, test_writer,epoch,batch_num):
    ids, feed = make_feed(batch, model)
    feed[model.dropout_pl] = 1
    loss_val,gs, summary,probs = do_val_fetches(feed, model, sess)
    print("Test {gs} {loss}".format(gs=gs+batch_num, loss=loss_val))
    test_writer.add_summary(summary, gs+batch_num)
    results = np.stack([ids,probs],axis=1)
    path =os.path.join(FLAGS.save_dir,"test","results")
    return results





def do_train_fetches(feed, model, sess):
    _opt, _metrics, loss_val, summary, gs = sess.run(
        [
            model.train_op,
            model.metrics_update_op,
            model.loss_op,
            model.train_summaries,
            model.gs
        ],
        feed_dict=feed)
    return loss_val,gs, summary

def do_val_fetches(feed, model, sess):
    _metrics, loss_val, summary, gs ,probs= sess.run(
        [
            model.metrics_update_op,
            model.loss_op,
            model.val_summaries,
            model.gs,
            model.probs
        ],
        feed_dict=feed)
    return loss_val,gs, summary,probs

def make_feed(batch, model):
    ids,s1, s2, l1, l2, label = batch
    feed = {
        model.s1: s1,
        model.s2: s2,
        model.l1: l1,
        model.l2: l2,
        model.labels: label,
        model.dropout_pl:FLAGS.dropout_keep_prob
    }
    return ids,feed


def print_paramater_count(total_parameters):
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parametes = 1
        for dim in shape:
            print(dim)
            variable_parametes *= dim.value
        print(variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)


if __name__ == '__main__':
    tf.app.run()
