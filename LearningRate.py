import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LearningRate:

    def __init__(self, x, y, tf_sess, loss, accuracy, dataset, epochs=50, learning_rate=1e-5, plot_charts=False):
        self.tf_sess = tf_sess
        self.loss = loss
        self.accuracy = accuracy
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.plot_charts = plot_charts
        self.x = x
        self.y = y
        self.final_learning_rate = self.get_optimal_learning_rate(self.epochs, self.learning_rate, self.plot_charts)

    def get_optimal_learning_rate(self, epochs=50, learning_rate=1e-5, plot_charts=False):

        self.tf_sess.run(tf.global_variables_initializer())
        rates, t_loss, t_acc = [], [], []

        self.tf_sess.run(self.dataset.train_init)
        for i in range(epochs):
            # Store learning rate in a tf variable and update it
            # g_step = tf.Variable(0, trainable=False)
            # lr = tf.train.exponential_decay(learning_rate, g_step, 100000, 0.96, staircase=True)

            learning_rate *= 1.1
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

            bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
            feed_dict = {
                self.x: bx,
                self.y: by
            }

            self.tf_sess.run(optimizer, feed_dict=feed_dict)
            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            if np.isnan(loss):
                loss = np.nan_to_num(loss)
            rates.append(learning_rate)
            t_loss.append(loss)
            t_acc.append(acc)

            print(f'epoch {i + 1}: learning rate = {learning_rate:.10f}, loss = {loss:.10f}')
        if plot_charts:
            iters = np.arange(len(rates))
            plt.title("Learning Rate (log) vs. Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")
            plt.plot(iters, rates, 'b')
            plt.show()

            plt.plot(rates, t_loss, 'b')
            plt.title("Loss vs. Learning Rate (log)")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.show()

        # Calculate the learning rate based on the biggest derivative betweeen the loss and learning rate
        dydx = list(np.divide(np.diff(t_loss), np.diff(rates)))
        start = rates[dydx.index(max(dydx))]
        print("Chosen start learning rate:", start)
        print()
        # self.tf_sess.close()
        return start
