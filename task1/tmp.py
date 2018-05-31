# As usual, a bit of setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.layers import *
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from cs231n.layer_utils import conv_bn_relu_forward, conv_bn_relu_backward
from cs231n.solver import Solver


class ConvNet(object):

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32,64,128], filter_size=3,
                 hidden_dim=128, num_classes=10, weight_scale=1e-3, reg=0.001,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters[0], C, filter_size, filter_size)
        self.params['b1'] = np.zeros((1, num_filters[0]))
        self.params['gamma1'] = np.ones((1, num_filters[0]))
        self.params['beta1'] = np.zeros((1, num_filters[0]))
        self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size, filter_size)
        self.params['b2'] = np.zeros((1, num_filters[1]))
        self.params['gamma2'] = np.ones((1, num_filters[1]))
        self.params['beta2'] = np.zeros((1, num_filters[1]))
        self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size, filter_size)
        self.params['b3'] = np.zeros((1, num_filters[2]))
        self.params['gamma3'] = np.ones((1, num_filters[2]))
        self.params['beta3'] = np.zeros((1, num_filters[2]))
        self.params['W4'] = weight_scale * np.random.randn(int(num_filters[2] * H * W / 4 / 4 / 4), hidden_dim)
        self.params['b4'] = np.zeros((1, hidden_dim))
        self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b5'] = np.zeros((1, num_classes))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        bn_param = [{'mode': 'train'}, None, {'mode': 'train'}, None, {'mode': 'train'}, None]

        W1, b1 = self.params['W1'], self.params['b1']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        W2, b2 = self.params['W2'], self.params['b2']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        print('input',X.shape, W1.shape, b1.shape)
        a1, cache1 = conv_bn_relu_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param[0])
        a2, cache2 = max_pool_forward_fast(a1, pool_param)

        print('input',a2.shape, W2.shape, b2.shape)
        a3, cache3 = conv_bn_relu_forward(a2, W2, b2, gamma2, beta2, conv_param, bn_param[2])
        a4, cache4 = max_pool_forward_fast(a3, pool_param)
        a5, cache5 = conv_bn_relu_forward(a4, W3, b3, gamma3, beta3, conv_param, bn_param[4])
        a6, cache6 = max_pool_forward_fast(a5, pool_param)
        a7, cache7 = affine_relu_forward(a6, W4, b4)
        scores, cache8 = affine_forward(a7, W5, b5)

        if y is None:
            return scores

        data_loss, dscores = softmax_loss(scores, y)
        da7, dW5, db5 = affine_backward(dscores, cache8)
        da6, dW4, db4 = affine_relu_backward(da7, cache7)
        da5 = max_pool_backward_fast(da6, cache6)
        da4, dW3, db3, dgamma3, dbeta3 = conv_bn_relu_backward(da5, cache5)
        da3 = max_pool_backward_fast(da4, cache4)
        da2, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_backward(da3, cache3)
        da1 = max_pool_backward_fast(da2, cache2)
        dX, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(da1, cache1)
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5])
        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'gamma1': dgamma1, 'beta1': dbeta1, 'W2': dW2, 'b2': db2, 'gamma2': dgamma2, 'beta2': dbeta2, 'W3': dW3, 'b3': db3, 'gamma3': dgamma3, 'beta3': dbeta3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5}

        return loss, grads

if __name__ == '__main__':
    data = get_CIFAR10_data()
    for k, v in data.items():
        print('%s: ' % k, v.shape)

    model = ConvNet(weight_scale=1e-2, filter_size=3, reg=1e-3)
    solver = Solver(model, data, num_epochs=10, batch_size=100, update_rule='adam', optim_config={'learning_rate': 1e-3}, verbose=True, print_every=500)
    solver.train()

    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

