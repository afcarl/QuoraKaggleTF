from pyspark import SparkContext
import csv
import pickle
sc = SparkContext("local[3]", "App Name",)

PAD_TOKEN = chr(0)
PAD_ID = 0
data_path = '/tmp/train.csv'
f = open(data_path)
lines = [line for line in csv.reader(f)]
rdd =sc.parallelize(lines[1:])
letters = rdd.flatMap(lambda x:set(x[3]+x[4])).distinct().collect()
vocab = {letter:num+1 for num,letter in enumerate(letters) }
vocab[PAD_TOKEN] =0
vocab_bd = sc.broadcast(vocab)
max_len = 150
def getQuestions(line,vocab_bd,max_len_bd):
    id, idq1,idq2,q1,q2,label = line
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
ml_bd = sc.broadcast(max_len)
questions = rdd.map(lambda x: getQuestions(x,vocab_bd,ml_bd)).filter(lambda x:x["max_len"] <max_len).cache()

params = {
    "vocab" :vocab,
    "max_len": max_len
}


padded_questions = questions.map(lambda x:padItemsAndStringify(x,ml_bd))
padded_questions.repartition(1).saveAsTextFile('/tmp/test3')
with open('/tmp/test3/vocab.pkl','wb') as f:
    pickle.dump(params,f)


