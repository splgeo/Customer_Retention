# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:58:58 2018

@author: david
"""

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

raw_csv_data = np.loadtxt('Audiobooks-data.csv',delimiter=',')
unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)



class Audiobooks_Data_Reader():
    def __init__(self, dataset, batch_size = None):
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size
    
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        return inputs_batch, targets_one_hot
    
    def __iter__(self):
        return self
    

input_size = 10
output_size = 2
hidden_layer_size = 50

tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])


weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, output_size])
biases_3 = tf.get_variable("biases_3", [output_size])

outputs = tf.matmul(outputs_2, weights_3) + biases_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mean_loss)

sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()
sess.run(initializer)

batch_size = 50

max_epochs = 500
prev_validation_loss = 9999999.

train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0.
    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss], 
            feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss
    curr_epoch_loss /= train_data.batch_count
    validation_loss = 0.
    validation_accuracy = 0.

    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})
    
    print('Epoch '+str(epoch_counter+1)+
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss
    
print('End of training.')

test_data = Audiobooks_Data_Reader('test')

for inputs_batch, targets_batch in test_data:
    test_accuracy = sess.run([accuracy],
                     feed_dict={inputs: inputs_batch, targets: targets_batch})

test_accuracy_percent = test_accuracy[0] * 100.

print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')