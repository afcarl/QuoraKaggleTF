import tensorflow as tf
from arg_getter import FLAGS

def prepareInputs(mode="train"):
    data_path = './data/*.csv' if mode=="train" else './test_data/*.csv'
    max_len = 150
    reader = tf.TextLineReader()
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./data/*.csv"),num_epochs=FLAGS.num_epochs)
    key, value = reader.read(filename_queue)
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    defaults = [[0] for _ in range(max_len * 2 + 4)]
    csv_data = tf.decode_csv(value, record_defaults=defaults)
    label, max_len_tens, length_1, length_2 = csv_data[:4]
    remaining = csv_data[4:]
    q1 = tf.stack(remaining[:max_len])
    q2 = tf.stack(remaining[max_len:])
    return label, q1,q2, length_1,length_2

def prepareInputsBatch(batch_size=100,mode="train"):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    label, q1, q2, l1, l2= tf.train.shuffle_batch(
        prepareInputs(mode), batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return label, [q1, q2], [l1, l2]

# prepareInputs()
# with tf.Session() as sess:
#   # Start populating the filename queue.
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#
#   for i in range(10):
#     # Retrieve a single instance:
#     lbl,l1,l2 = sess.run([label, q1,length_2])
#     print(lbl,l1,l2)
#
#   coord.request_stop()
#   coord.join(threads)
#
