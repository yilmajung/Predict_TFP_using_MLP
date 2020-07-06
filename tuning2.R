# Flags
FLAGS <- flags(flag_integer('dense_units1', 64),
               flag_integer('dense_units2', 32),
               flag_integer('dense_units3', 32),
               flag_numeric('l2_lambda1', 0.01),
               flag_numeric('l2_lambda2', 0.01),
               flag_numeric('l2_lambda3', 0.01))

# Model
l2_growth <- keras_model_sequential()
l2_growth %>% 
  layer_dense(units = FLAGS$dense_units1, 
              activation = "relu", input_shape = 782,
              kernel_regularizer = regularizer_l2(l = FLAGS$l2_lambda1)) %>% 
  layer_dense(units = FLAGS$dense_units2, activation = "relu",
              kernel_regularizer = regularizer_l2(l = FLAGS$l2_lambda2)) %>% 
  layer_dense(units = FLAGS$dense_units3, activation = "relu",
              kernel_regularizer = regularizer_l2(l = FLAGS$l2_lambda3)) %>% 
  layer_dense(units = 4, activation = "softmax") 
summary(l2_growth)

# Compile the model
l2_growth %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy")) 

# Fit the data
history <- l2_growth %>% 
  fit(growth_x_train, growth_y_train, epoch = 20, batch_size = 128, validation_split = 0.2, verbose = 2)
