import tensorflow as tf
from arg_getter import FLAGS
from tensorflow.contrib.learn.python.learn.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.monitored_session import MonitoredTrainingSession

from bytenet.bytenet_quora import BytenetQuora
from input_loader import prepareInputsBatch
def main(__):
    label, inputs, lengths = prepareInputsBatch(FLAGS.batch_size)
    gs = tf.contrib.framework.get_or_create_global_step()
    model = BytenetQuora(inputs[0],inputs[1],label,gs)
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
            _, loss_val,_ = sess.run([model.train_op, model.loss_op,model.metrics_op])
            print(i, loss_val)
            i += 1
        coord.request_stop()
        coord.join(threads)
if __name__=='__main__':
    tf.app.run()