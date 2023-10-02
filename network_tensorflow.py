import tensorflow as tf
tf.config.run_functions_eagerly(True)


class NeuralNetworkTf(tf.keras.Sequential):

  def __init__(self, sizes, random_state=1):
    
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)

    self.add(tf.keras.layers.Flatten(input_shape=(28,28))) #We flatten the input data here so that shapes are compatible later (mistake 1)
    
    for i in range(0, len(sizes)):
      if i == len(sizes) - 1:
        #self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid')) 
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax')) # Mistake 2: We changed the activation function for the ouput layer from sigmoid to softmax, which is suited for multi-class classification. 
      
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid')) # Minor adjustment: Using sigmoid instead of softmax here
        
  
  def compile_and_fit(self, x_train, y_train, epochs=50, learning_rate=0.01, batch_size=1, validation_data = None):
    
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

    loss_function = tf.keras.losses.CategoricalCrossentropy() # Mistake 3: Using categorial cross entropy instead of binary since we are dealing with multiple classes (digits)
    
    eval_metrics = ['accuracy']

    super().compile(optimizer=optimizer, loss=loss_function, metrics=eval_metrics)
    return super().fit(x_train, y_train, epochs=epochs, 
                       batch_size=batch_size, 
                       validation_data=validation_data)


class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''TODO: Implement a time-based learning rate that takes as input a 
  positive integer (initial_learning_rate) and at each step reduces the
  learning rate by 1 until minimal learning rate of 1 is reached.
    '''

  def __init__(self, initial_learning_rate, decay=1, min_learning_rate=1):
      
      super(TimeBasedLearningRate, self).__init__()
      self.initial_learning_rate = initial_learning_rate
      self.decay = tf.cast(decay, dtype=tf.float32) # adding decay and min learning rate as tuneable parameters to be able to correct the model in the notebook later on
      self.min_learning_rate = min_learning_rate

  
  def __call__(self, step):
      learning_rate = tf.maximum(self.min_learning_rate, self.initial_learning_rate - step * self.decay)
      return learning_rate
    