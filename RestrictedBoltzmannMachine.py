import numpy as np
import sys
from random import *
from matplotlib import pyplot as plt

class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden_node = num_hidden
        self.num_visible_nodes = num_visible

        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def getRandomSamples(self, data, k=1000):
        samples = np.zeros((k, self.num_visible_nodes+1), dtype=np.dtype('b'))
        for i in range(k):
            samples[i] = data[randint(0, data.shape[0]-1)]
        return samples

    def initRandomSamples(self, k=1000):
        if(k==-1):
            k = 10000 #default data size
        samples = np.ones((k, self.num_visible_nodes + 1))
        for i in range(k):
            samples[i,1:] = np.random.choice([0,1],self.num_visible_nodes)
        return samples

    def train(self, data, max_epochs=1000, learning_rate=0.1, m=1000, k=5):
        initSamples = self.initRandomSamples(m)
        if(m == -1):
            num_examples = data.shape[0]
        else:
            num_examples = m

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        # For plotting
        error_list = list()

        for epoch in range(max_epochs):
            if(m != -1):
                samples = self.getRandomSamples(data,m)
            # positive phase
            if(m == -1):
                pos_activations = np.dot(data, self.weights)
            else:
                pos_activations = np.dot(samples, self.weights)

            pos_probs = self._logistic(pos_activations)
            pos_probs[:,0] = 1 # bias unit

            if(m == -1):
                pos_associations = np.dot(data.T, pos_probs)
            else:
                pos_associations = np.dot(samples.T, pos_probs)

            # negative phase
            # Start the alternating Gibbs sampling.
            for i in range(1, k):
                hidden_activations = np.dot(initSamples, self.weights)
                hidden_probs = self._logistic(hidden_activations)
                hidden_probs[:,0] = 1 # bias unit.
                hidden_states = hidden_probs > np.random.rand(self.num_hidden_node + 1)
                hidden_states[:0] = 1

                # Recalculate the probabilities that the visible units are on.
                visible_activations = np.dot(hidden_states, self.weights.T)
                visible_probs = self._logistic(visible_activations)
                visible_probs[:,0] = 1 # bias unit.
                initSamples = visible_probs > np.random.rand(self.num_visible_nodes + 1)

            neg_activations = np.dot(hidden_states, self.weights.T)
            neg_probs = self._logistic(neg_activations)
            neg_probs[:,0] = 1 # bias unit.
            neg_hidden_activations = np.dot(neg_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_probs.T, neg_hidden_probs)

            # Update weights
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            if(m == -1):
                error = np.sum((data - neg_probs) ** 2)
            else:
                error = np.sum((samples - neg_probs) ** 2)

            print("Epoch %s: error is %s" % (epoch, error))
            error_list.append(error)
        return error_list



    def markovChain(self, num_samples):
        samples = np.ones((num_samples, self.num_visible_nodes + 1))
        samples[0,1:] = np.random.choice([0,1],self.num_visible_nodes)

        for i in range(1, num_samples):
            visible = samples[i-1,:]

            hidden_activations = np.dot(visible, self.weights)
            hidden_probs = self._logistic(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden_node + 1)
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible_nodes + 1)
            samples[i,:] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

    def setSeed(self):
        seed = np.random.randint(1000000)
        np.random.seed(seed)

if __name__ == '__main__':

    if(len(sys.argv) < 3):
        num_visible=784
        num_hidden=1000
    else:
        num_visible = int(sys.argv[1])
        num_hidden = int(sys.argv[2])

    r = RBM(num_visible, num_hidden)
    r.setSeed()

    dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
    mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0]
    imgs = np.zeros((10000, 784), dtype=np.dtype('b'))
    imgs[mnist > 127] = 1
    max_epochs = 20
    error_list_default = r.train(imgs, max_epochs)
    error_list_lr0_5 = r.train(imgs, max_epochs, learning_rate=0.5)
    error_list_lr0_01 = r.train(imgs, max_epochs, learning_rate=0.01)

    # Plotting
    #error_list = [x for x in error_list]
    epochs = np.arange(max_epochs)
    plt.title('Error values with default settings')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error values')
    plt.xticks(np.arange(max_epochs, step=100))
    plt.plot(epochs, error_list_default, 'r',
             epochs, error_list_lr0_5, 'g',
             epochs, error_list_lr0_01, 'b')
    plt.legend(['learning rate=0.1', 'learning rate=0.5', 'learning rate=0.01'])
    plt.savefig('result_original.png')

    lst = []
    generated1 = r.markovChain(20)
    generated2 = r.markovChain(20)
    generated3 = r.markovChain(20)
    lst.append(generated1)
    lst.append(generated2)
    lst.append(generated3)

    for generated in lst:
	    for img in generated:
	        first_image = img
	        first_image = np.array(first_image, dtype='float')
	        pixels = first_image.reshape((28, 28))
	        plt.imshow(pixels, cmap='gray')
	        plt.show()