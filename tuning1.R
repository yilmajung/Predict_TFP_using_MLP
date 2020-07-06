# Flags
FLAGS <- flags(flag_integer('dense_units1', 64),
               flag_integer('dense_units2', 32),
               flag_integer('dense_units3', 32),
               flag_string('activation1', "relu"),
               flag_string('activation2', "relu"),
               flag_string('activation3', "relu"))

# Model
bm_growth <- keras_model_sequential()
bm_growth %>% 
  layer_dense(units = FLAGS$dense_units1,
              activation = FLAGS$activation1, input_shape = 782) %>% 
  layer_dense(units = FLAGS$dense_units2,
              activation = FLAGS$activation2) %>% 
  layer_dense(units = FLAGS$dense_units3,
              activation = FLAGS$activation3) %>% 
  layer_dense(units = 4, activation = "softmax") 
summary(bm_growth)

# Compile the model
bm_growth %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- bm_growth %>% 
  fit(growth_x_train, growth_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)


