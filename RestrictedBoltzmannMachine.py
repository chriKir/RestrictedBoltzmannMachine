from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)). One could vary the
        # standard deviation by multiplying the interval with appropriate value.
        # Here we initialize the weights with mean 0 and standard deviation 0.1.
        # Reference: Understanding the difficulty of training deep feedforward
        # neural networks by Xavier Glorot and Yoshua Bengio
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs = 1000, learning_rate = 0.1):
        """
        Train the machine.
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)

        visible_probs = np.zeros(data.shape,dtype='double')

        for epoch in range(max_epochs):
            #while error > 200000:
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:,0] = 1 # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            # Start the alternating Gibbs sampling.
            visible_states = np.random.rand(len(data), len(data[0]))

            for i in range(1, 5):
                # Calculate the activations of the hidden units.
                hidden_activations = np.dot(visible_states, self.weights)
                # Calculate the probabilities of turning the hidden units on.
                hidden_probs = self._logistic(hidden_activations)
                # Turn the hidden units on with their specified probabilities.
                hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
                # Always fix the bias unit to 1.
                hidden_probs[:,0] = 1

                # Recalculate the probabilities that the visible units are on.
                visible_activations = np.dot(hidden_states, self.weights.T)
                visible_probs = self._logistic(visible_activations)
                visible_states = visible_probs > np.random.rand(self.num_visible + 1)
                visible_probs[:,0] = 1 # Fix the bias unit.

            neg_hidden_activations = np.dot(visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - visible_probs) ** 2)

            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

    def daydream(self, num_samples):
        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0,1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i-1,:]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
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
    r = RBM(num_visible = 784, num_hidden = 10)
    dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
    mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0]
    imgs = np.zeros((10000, 784), dtype=np.dtype('b'))
    imgs[mnist > 127] = 1
    r.train(imgs, max_epochs = 1000)
    generated = r.daydream(10)

    for img in generated:
        first_image = img
        first_image = np.array(first_image, dtype='float')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()