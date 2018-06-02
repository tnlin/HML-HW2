from data_utils_cora import *
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2


class GCN_Layer(Layer):
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GCN_Layer, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape 

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias
        return self.activation(output)


X, A, y = load_data(dataset='cora')
y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y)
mask = np.zeros(y.shape[0])
mask[idx_train] = 1
train_mask = np.array(mask, dtype=np.bool)

X /= X.sum(1).reshape(-1, 1)
A_ = preprocess_adj(A, True)
support = 1
graph = [X, A_]
G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

X_in = Input(shape=(X.shape[1],))
H = Dropout(0.4)(X_in)
H = GCN_Layer(12, support, activation='relu', kernel_regularizer=l2(1e-4))([H]+G)
H = Dropout(0.4)(H)
Y = GCN_Layer(y.shape[1], support, activation='softmax')([H]+G)
model = Model(inputs=[X_in]+G, outputs=Y)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
wait = 0
preds = None
best_val_loss = 10000
for epoch in range(1, 100):

    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    preds = model.predict(graph, batch_size=A.shape[0])

    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    if epoch%10==1:
        print(
            "Epoch: {:04d}".format(epoch),
            "train_loss= {:.4f}".format(train_val_loss[0]),
            "train_acc= {:.4f}".format(train_val_acc[0]),
            "val_loss= {:.4f}".format(train_val_loss[1]),
            "val_acc= {:.4f}".format(train_val_acc[1])
        )

    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= 10:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1
        
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(test_acc[0]))