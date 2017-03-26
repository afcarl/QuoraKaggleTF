from pyspark import SparkContext
import os
import csv
import pickle
import argparse
parser = argparse.ArgumentParser()
PAD_TOKEN = chr(0)
PAD_ID = 0
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", help="Where to save")
parser.add_argument("--mode", help="Test data or train data",default="train")
parser.add_argument("--data_path", help="Where the data is",)
parser.add_argument("--vocab_path", help="Where the vocab is if it is test mode",default=None)







def get_vocab(rdd,args):
    if args.mode =="train":
        letters = rdd.flatMap(lambda x: set(x[3] + x[4])).distinct().collect()
        vocab = {letter: num + 1 for num, letter in enumerate(letters)}
        vocab[PAD_TOKEN] = 0
    if args.mode=="test":
        with open(args.vocab_path,'rb') as f:
            params =pickle.load(f)

    return params["vocab"]



def getQuestions(line,vocab_bd,max_len_bd,args):
    if args.mode =="train":
        id, idq1,idq2,q1,q2,label = line
    else:
        id, q1, q2, label = line
    label = int(label)
    q1_length = len(q1)
    q2_length = len(q2)
    q1 = list(map(vocab_bd.value.get, q1.ljust(max_len_bd.value,PAD_TOKEN)))
    q2 = list(map(vocab_bd.value.get, q2.ljust(max_len_bd.value,PAD_TOKEN)))
    max_len = max(q1_length,q2_length)
    result = {
        "q1":q1,
        "q2":q2,
        "label":[label],
        "max_len":max_len,
        "q1_length":[q1_length],
        "q2_length": [q2_length]
    }
    return result


def padItemsAndStringify(item,max_len_bd):
    listed_item = item["label"] +[max_len_bd.value] +item["q1_length"] +item["q2_length"] +item["q1"] + item["q2"]
    csv_line = ','.join(map(str,listed_item))
    return csv_line


def save_vocabs(args):
    global f
    params = {
        "vocab": vocab,
        "max_len": max_len
    }
    with open(os.path.join(args.save_dir,'vocab.pkl', 'wb')) as f:
        pickle.dump(params, f)


args = parser.parse_args()
sc = SparkContext("local[3]", "App Name", )
f = open(args.data_path)
lines = [line for line in csv.reader(f)]
rdd = sc.parallelize(lines[1:])
vocab = get_vocab(rdd, args=args)
vocab_bd = sc.broadcast(vocab)
max_len = 150
ml_bd = sc.broadcast(max_len)
questions = rdd.map(lambda x: getQuestions(x, vocab_bd, ml_bd)).filter(lambda x: x["max_len"] < max_len).cache()
padded_questions = questions.map(lambda x: padItemsAndStringify(x, ml_bd))
padded_questions.repartition(1).saveAsTextFile(args.save_dir)
save_vocabs(args)


