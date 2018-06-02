import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from keras.layers import Embedding, Reshape, Merge, Activation, Input, merge
from keras.models import Sequential, Model
import keras.backend as K


def LINE(train_edges, test_edges, test_edges_false):
    epoch_num = 10
    factors = 100
    batch_size = 2000
    samples_per_epoch = 100
    negative_ratio = 2
    adj_list = train_edges

    n_nodes = np.max(adj_list.ravel()) + 1
    data_generate = batch_train(adj_list, n_nodes, batch_size, negative_ratio)

    print("Building model...")
    model_used, embed_generator = create_model(n_nodes, factors)
    model_used.compile(optimizer='rmsprop', loss={'left_right_dot': LINE_loss})
    model_used.fit_generator(data_generate, samples_per_epoch=samples_per_epoch, nb_epoch=epoch_num, verbose=1)
    
    print("Evaluating...")
    arr_test = np.concatenate((test_edges,test_edges_false),axis=0)
    y_test = np.zeros(arr_test.shape[0])
    bias = test_edges.shape[0]
    for i in range(bias):
        y_test[i] = 1

    n_total = arr_test.shape[0]
    y_pred = []
    for i in range(n_total):
        batch = [np.asarray([arr_test[i,0]]),np.asarray([arr_test[i,1]])]
        x = embed_generator.predict_on_batch(batch)
        dot = np.dot(x[0][0],x[1][0])
        y_pred.append(dot)

    y_pred = np.array(y_pred)
    print('AUC:', roc_auc_score(y_test, y_pred))


def create_model(n_nodes, factors):
    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))

    left_model = Sequential()
    left_model.add(Embedding(input_dim=n_nodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    left_model.add(Reshape((factors,)))

    right_model = Sequential()
    right_model.add(Embedding(input_dim=n_nodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    right_model.add(Reshape((factors,)))

    left_embed = left_model(left_input)
    right_embed = left_model(right_input)

    left_right_dot = merge([left_embed, right_embed], mode="dot", dot_axes=1, name="left_right_dot")

    model = Model(input=[left_input, right_input], output=[left_right_dot])
    embed_generator = Model(input=[left_input, right_input], output=[left_embed, right_embed])

    return model, embed_generator


def LINE_loss(y_true, y_pred):
    coeff = y_true*2 - 1
    return -K.mean(K.log(K.sigmoid(coeff*y_pred)))


def batch_train(adj_list, n_nodes, batch_size, negative_ratio):
    batch_size_ones = np.ones((batch_size), dtype=np.int8)
    nb_train_sample = adj_list.shape[0]
    index_array = np.arange(nb_train_sample)

    nb_batch = nb_train_sample // batch_size
    batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, nb_batch)]

    while 1:
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            pos_edge_list = index_array[batch_start:batch_end]
            pos_left_nodes = adj_list[pos_edge_list, 0]
            pos_right_nodes = adj_list[pos_edge_list, 1]

            pos_relation_y = batch_size_ones[0:len(pos_edge_list)]
            neg_left_nodes = np.zeros(len(pos_edge_list)*negative_ratio, dtype=np.int32)
            neg_right_nodes = np.zeros(len(pos_edge_list)*negative_ratio, dtype=np.int32)
                                        
            # Negative sampling's edge and node
            neg_relation_y = np.zeros(len(pos_edge_list)*negative_ratio, dtype=np.int8)

            left_nodes = np.concatenate((pos_left_nodes, neg_left_nodes), axis=0)
            right_nodes = np.concatenate((pos_right_nodes, neg_right_nodes), axis=0)
            relation_y = np.concatenate((pos_relation_y, neg_relation_y), axis=0)
            yield ([left_nodes, right_nodes], [relation_y])


if __name__=='__main__':
    train_edges=np.load('data/tencent/train_edges.npy')
    test_edges=np.load('data/tencent/test_edges.npy')
    test_edges_false=np.load('data/tencent/test_edges_false.npy')
    LINE(train_edges,test_edges,test_edges_false)
