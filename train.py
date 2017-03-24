import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.contrib.learn.python.learn.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from input_loader import prepareInputsBatch
from model import BasicQuoraModel
def main(__):
    label, inputs, lengths = prepareInputsBatch(FLAGS.batch_size)
    gs = tf.contrib.framework.get_or_create_global_step()
    model = BasicQuoraModel(gs,inputs,lengths,label)
    init = tf.global_variables_initializer()
    with MonitoredTrainingSession(
        checkpoint_dir=FLAGS.save_dir,
        save_summaries_steps=1,
        hooks=[NanTensorHook(model.loss_op)]

    ) as sess:

        sess.run(init)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        i = 0
        while True:
            _, loss_val = sess.run([model.train_op, model.loss_op])
            print(i, loss_val)
            i += 1
        coord.request_stop()
        coord.join(threads)
if __name__=='__main__':
    tf.app.run()