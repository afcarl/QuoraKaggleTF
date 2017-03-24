import numpy as np
import tensorflow as tf

from csvInputPipeline import prepareInputsBatch
from simpleModel import VecDifModel, inference, loss, optimizer


class Test():
    class Args:
        vocab_size=1335
        hidden_size=100

    def run(self):
        a = self.Args()
        model = VecDifModel(a)
        s1 = np.random.randint(0,a.vocab_size,size=(20,5))
        s2 = np.random.randint(0, a.vocab_size, size=(20, 5))
        target = np.random.randint(0, 2, size=(20))
        lengths = np.ones([20])*5
        import missinglink
        missinglink_callback = missinglink.TensorFlowCallback(owner_id="36428cd1-b6bb-cd17-9c27-fb2eacb91298",
                                                              project_token="smtvMkkjjvfhaCxL")
        missinglink_callback.set_monitored_fetches({
            'loss': model.loss,
        })
        missinglink_callback.set_hyperparams(vocab_size=3,hidden_size=10)
        missinglink_callback.set_properties(display_name='PooNet', description='A network for poo')


        with tf.Session() as sess:
            feed = {
                model.s1_pl:s1,
                model.s2_pl:s2,
                model.targets_pl:target,
                model.lengths: lengths
            }
            init = tf.global_variables_initializer()
            sess.run(init)
            for _ in range(10000):
                loss,_ = sess.run_batch([model.loss,model.train_op],feed_dict=feed,epoch = 1)
                print(loss)
    def test_inference(self):
        a = self.Args()
        label, [q1, q2], [l1, l2] = prepareInputsBatch(20)
        logits = inference(a,q1,q2,l1,l2)
        loss_op = loss(logits, label)
        opt = optimizer(loss_op)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            i =0
            while True:
                _,loss_val = sess.run([opt,loss_op])
                print(i,loss_val)
                i+=1
            coord.request_stop()
            coord.join(threads)


A = Test()