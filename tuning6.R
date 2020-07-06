# Flags
FLAGS <- flags(flag_integer('dense_units1', 64),
               flag_integer('dense_units2', 32),
               flag_integer('dense_units3', 32),
               flag_numeric('dropout1', 0.4),
               flag_numeric('dropout2', 0.4),
               flag_numeric('dropout3', 0.4))


# Model
bm_tfp <- keras_model_sequential()
bm_tfp %>% 
  layer_dense(units = FLAGS$dense_units1,
              activation = "relu", input_shape = 782) %>% 
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$dense_units2,
              activation = "relu") %>% 
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$dense_units3,
              activation = "relu") %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = 4, activation = "softmax") 


# Compile the model
bm_tfp %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- bm_tfp %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 20, batch_size = 128, validation_split = 0.2, verbose = 2)


