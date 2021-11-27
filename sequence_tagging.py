import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import keras_contrib
import matplotlib.pyplot as plt

file_dir = "D:/文件雁栖湖/学习相关/nlp/作业三/作业三：序列标注/data"

with open(file_dir+'/train_corpus.txt','r',encoding='utf-8') as f:
    data = f.read()
train_data = data.split('\n')
maxlen = 0
for item in train_data:
    if len(item) > maxlen:
        maxlen = len(item)

# 使用tokenizer分词器构建5000词的语料库
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_data)
train_data = pad_sequences(sequences, maxlen=int(maxlen/2))
print('训练数据：',train_data.shape)


def process_label(filepath):
    with open(filepath,'r') as f:
        data = f.read()
    train_label = data.split('\n')
    for i in range(len(train_label)):
        train_label[i] = train_label[i].split(' ')
        train_label[i].pop(-1)
    # 标签转成数值型
    for i in range(len(train_label)):
        for j in range(len(train_label[i])):
            if train_label[i][j] == 'O':
                train_label[i][j] = 0
            elif train_label[i][j] == 'B-LOC':
                train_label[i][j] = 1
            elif train_label[i][j] == 'I-LOC':
                train_label[i][j] = 2
            elif train_label[i][j] == 'B-PER':
                train_label[i][j] = 3
            elif train_label[i][j] == 'I-PER':
                train_label[i][j] = 4
            elif train_label[i][j] == 'B-ORG':
                train_label[i][j] = 5
            elif train_label[i][j] == 'I-ORG':
                train_label[i][j] = 6
            else:
                print('error')
                print(train_label[i][j])
    # 对标签进行padding
    for i in range(len(train_label)):
        if len(train_label[i]) < 100:
            pad = np.zeros((100 - len(train_label[i]))).tolist()
            train_label[i] = pad + train_label[i]
    train_label = np.array(train_label)
    print('标签：',train_label.shape)
    return train_label
train_label = process_label(file_dir+'/train_label.txt')

# 减少训练数量
# train_data = train_data[:5000]
# train_label = train_label[:5000]
# # 划分验证集
# train_data = train_data[:int(len(train_data)*0.9)]
# train_label = train_label[:int(len(train_data)*0.9)]
# val_data = train_data[int(len(train_data)*0.9):]
# val_label = train_label[int(len(train_data)*0.9):]

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=5000, output_dim=50, input_length=100))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(150,return_sequences=True)))
# model.add(keras.layers.Bidirectional(keras.layers.LSTM(150,return_sequences=True)))
model.add(keras_contrib.layers.CRF(units=7,learn_mode='marginal',sparse_target=True))

model.compile(optimizer=tf.optimizers.Adam(),loss=keras.losses.sparse_categorical_crossentropy,metrics=['acc'])
model.summary()
history = model.fit(
    train_data,
    train_label,
    epochs=20,
    batch_size=128,
    verbose=1,
    validation_split=0.1
)
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.savefig('output.png')
plt.figure()
# plt.show()


with open(file_dir+'/test_corpus.txt','r',encoding='utf-8') as f:
    data = f.read()
test_data = data.split('\n')
sequences = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(sequences, maxlen=int(maxlen/2))
test_label = process_label(file_dir+'/test_label.txt')

pre = model.predict(test_data)
for i in range(1):
    print('真实的：',test_label[i])
    print('预测的：',pre[i].argmax(axis=1))

# 精确率precision 查准率
def precision(label,pre,test_label):
    flag = 0
    flag_tp = 0
    for i in range(len(pre)):
        tmp = pre[i].argmax(axis=1)
        for j in range(100):
            if tmp[j] == label:
                flag += 1
                if test_label[i,j] == label:
                    flag_tp += 1
    return flag_tp/flag

# 召回率recall 查全率
def recall(label,pre,test_label):
    flag = 0
    flag_tp = 0
    for i in range(len(test_label)):
        tmp = pre[i].argmax(axis=1)
        for j in range(100):
            if test_label[i,j] == label:
                flag += 1
                if tmp[j] == label:
                    flag_tp += 1
    return flag_tp/flag

# f1值
def f1(precision,recall):
    return 2 * precision * recall / (precision + recall)

def result(pre,test_label):
    metric = []
    for i in range(7):
        P = precision(i,pre,test_label)
        R = recall(i,pre,test_label)
        F = f1(P,R)
        print('标签为',i,'时的指标：',P,R,F)
        metric.append([P,R,F])
    metric = np.array(metric)
    print('平均指标为：',np.mean(metric,axis=0))
    metric = np.delete(metric,np.s_[0],axis=0)
    print('去除O后的平均指标：',np.mean(metric,axis=0))

result(pre,test_label)