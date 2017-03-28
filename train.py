import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.contrib.learn.python.learn.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
import numpy as np
from bytenet.bytenet_quora import BytenetQuora
from input_loader import prepareInputsBatch
def main(__):
    #    label, inputs, lengths = prepareInputsBatch(FLAGS.batch_size)
    gs = tf.contrib.framework.get_or_create_global_step()
    model = BytenetQuora(gs)
    init = tf.global_variables_initializer()
    total_parameters = 0
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

    with MonitoredTrainingSession(
        checkpoint_dir=FLAGS.save_dir,
        save_summaries_steps=10,
        #hooks=[NanTensorHook(model.loss_op)]

    ) as sess:


        #init.run()
        # Start populating the filename queue.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        i = 0
        #s1,s2,l1,l2,lab = sess.run(inputs+lengths+[label])
        s1 = np.random.randint(0,100,[FLAGS.batch_size,25])
        s2 = np.random.randint(0, 100, [FLAGS.batch_size, 25])
        l1 = np.random.randint(0, 25, [FLAGS.batch_size,])
        l2 = np.random.randint(0, 25, [FLAGS.batch_size,])
        label = np.random.randint(0,2,[FLAGS.batch_size])
        feed = {
            model.s1: s1,
            model.s2: s2,
            model.l1: l1,
            model.l2: l2,
            model.labels: label
        }
        sess.run(init,feed_dict=feed)
        while True:
            feed = {
                model.s1 :s1,
                model.s2: s2,
                model.l1: l1,
                model.l2: l2,
                model.labels: label
            }
            _, loss_val,_ = sess.run([model.train_op, model.loss_op,model.metrics_op],feed_dict=feed)
            print(i, loss_val)
            i += 1
        coord.request_stop()
        coord.join(threads)
if __name__=='__main__':
    tf.app.run()