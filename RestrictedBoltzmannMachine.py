from __future__ import print_function
import numpy as np
from random import *
from matplotlib import pyplot as plt

class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def getRandomSamples(self, data, k=1000):
        samples = np.zeros((k, self.num_visible+1), dtype=np.dtype('b'))
        for i in range(k):
            samples[i] = data[randint(0, data.shape[0]-1)]
        return samples

    def initRandomSamples(self, k=1000):
        samples = np.ones((k, self.num_visible + 1))
        for i in range(k):
            samples[i,1:] = np.random.choice([0,1],self.num_visible)
        return samples

    def train(self, data, max_epochs=1000, learning_rate=0.1, m=1000, k=5):
        initSamples = self.initRandomSamples()
        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            samples = self.getRandomSamples(data)

            # positive phase
            pos_hidden_activations = np.dot(samples, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:,0] = 1 # bias unit
            pos_hidden_states = pos_hidden_probs > np.random.rand(m, self.num_hidden + 1)
            pos_associations = np.dot(samples.T, pos_hidden_states)

            # negative phase
            # Start the alternating Gibbs sampling.
            for i in range(1, k):
                hidden_activations = np.dot(initSamples, self.weights)
                hidden_probs = self._logistic(hidden_activations)
                hidden_probs[:,0] = 1 # bias unit.
                hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
                hidden_states[:0] = 1

                # Recalculate the probabilities that the visible units are on.
                visible_activations = np.dot(hidden_states, self.weights.T)
                visible_probs = self._logistic(visible_activations)
                visible_probs[:,0] = 1 # bias unit.
                initSamples = visible_probs > np.random.rand(self.num_visible + 1)

            neg_visible_activations = np.dot(hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((samples - neg_visible_probs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))



    def daydream(self, num_samples):
        samples = np.ones((num_samples, self.num_visible + 1))
        samples[0,1:] = np.random.choice([0,1],self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i-1,:]

            hidden_activations = np.dot(visible, self.weights)
            hidden_probs = self._logistic(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
    r = RBM(num_visible = 784, num_hidden = 784)
    dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
    mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0]
    imgs = np.zeros((10000, 784), dtype=np.dtype('b'))
    imgs[mnist > 127] = 1
    r.train(imgs, max_epochs = 500)
    generated = r.daydream(10)

    for img in generated:
        first_image = img
        first_image = np.array(first_image, dtype='float')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()