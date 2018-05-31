from __future__ import print_function
import numpy as np
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.layers import *
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from cs231n.layer_utils import conv_bn_relu_forward, conv_bn_relu_backward
from cs231n.solver import Solver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class SixLayerConvNet(object):
    """
    A six-layer convolutional network with the following architecture:

    conv - BN - ReLU
    2x2 max pooling
    conv - BN - ReLU
    2x2 max pooling
    affine - ReLU
    affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 64], filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=1e-3,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
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

        C, H, W = input_dim
        F1, F2 = num_filters[0], num_filters[1]
        HH, WW = filter_size, filter_size
        self.params['W1'] = weight_scale * np.random.randn(F1, C, HH, WW)
        self.params['W2'] = weight_scale * np.random.randn(F2, F1, HH, WW)
        self.params['W3'] = weight_scale * np.random.randn(int(F2 * H * W / 16), hidden_dim)
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros((1, F1))
        self.params['b2'] = np.zeros((1, F2))
        self.params['b3'] = np.zeros((1, hidden_dim))
        self.params['b4'] = np.zeros((1, num_classes))
        self.params['gamma1'] = np.ones((1, F1))
        self.params['gamma2'] = np.ones((1, F2))
        self.params['beta1'] = np.zeros((1, F1))
        self.params['beta2'] = np.zeros((1, F2))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the six-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        bn_param = [{'mode': 'train'}, {'mode': 'train'}]

        scores = None
        a1, cache1 = conv_bn_relu_forward(X, W1, b1, gamma1, beta1, conv_param, bn_param[0])
        a2, cache2 = max_pool_forward_fast(a1, pool_param)
        a3, cache3 = conv_bn_relu_forward(a2, W2, b2, gamma2, beta2, conv_param, bn_param[1])
        a4, cache4 = max_pool_forward_fast(a3, pool_param)
        a5, cache5 = affine_relu_forward(a4, W3, b3)
        scores, cache6 = affine_forward(a5, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscore = softmax_loss(scores, y)

        da5, grads['W4'], grads['b4'] = affine_backward(dscore, cache6)
        da4, grads['W3'], grads['b3'] = affine_relu_backward(da5, cache5)
        da3 = max_pool_backward_fast(da4, cache4)
        da2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_bn_relu_backward(da3, cache3)
        da1 = max_pool_backward_fast(da2, cache2)
        _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(da1, cache1)

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        grads['W4'] += self.reg * W4
        loss += 0.5 * self.reg * sum([np.sum(W**2) for W in [W1, W2, W3, W4]])

        return loss, grads

if __name__ == '__main__':
    data = get_CIFAR10_data()
    for k, v in data.items():
        print('%s: ' % k, v.shape)

    model = SixLayerConvNet(weight_scale=1e-2, filter_size=3, reg=1e-3)
    solver = Solver(
        model, data,
        num_epochs=10,
        batch_size=1000,
        update_rule='adam',
        optim_config={'learning_rate': 1e-3},
        verbose=True,
        print_every=1
    )
    solver.train()

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
    plt.savefig('result.png')

    y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
    print('Valid accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test accuracy: ', (y_test_pred == data['y_test']).mean())


