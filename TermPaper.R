library(keras)
mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

head(train_images)
dim(train_images)
str(train_images)

train_images <- array_reshape(train_images, c(60000, 28*28)) / 255
test_images <- array_reshape(test_images, c(10000, 28*28)) / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

head(train_images)
dim(train_images)
dim(test_images)
class(train_images)
class(test_images)
head(test_labels)


# Tensors
x <- c(1, 2, 3, 4)
attributes(x)
dim(x)
x <- as.array(x)
x
attributes(x)

x <- matrix(seq(1, 9), 3, 3)
x
dim(x)
attributes(x)

# 3D Tensor
x <- array(seq(1, 18), dim = c(3,3,2))
x

mini_batch <- train_images[5:24,,]
dim(mini_batch)

plot(as.raster(train_images[5,,], max = 255))
sessionInfo()

# Sweeping (broadcasting in Python)
x <- array(seq(-5, 18), dim = c(4, 3, 2))
x
y <- c(100, 50, 10, 5)
y
sweep(x, 1, y, '+')
?sweep

A <- array(1:24, dim = 4:2)
sweep(A, 2, 5)

x <- matrix(seq(1, 4), 2, 2)
x

array_reshape(x, dim = c(4, 2))

#################
boston_housing <- keras::dataset_boston_housing()
head(boston_housing)
str(boston_housing)
head(boston_housing$train)

c(train_data, train_target) %<-% boston_housing$train
c(test_data, test_target) %<-% boston_housing$test

column_names <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat")
colnames(train_data) <- column_names
train_df <- tibble::as_tibble(train_data)

col_means_train <- attr(train_data, "scaled:center")
col_stddevs_train <- attr(train_data, "scaled:scale")

test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

round(head(train_data[, 1:5]), 2)

build_model <- function(dropout = 0) {
  model <- keras::keras_model_sequential()
  model %>% 
    layer_dense(units = 64, 
                activation = "relu",
                input_shape = dim(train_data)[2]) %>% 
    layer_dropout(dropout) %>% 
    layer_dense(units = 64, 
                activation = "relu") %>% 
    layer_dropout(dropout) %>% 
    layer_dense(units = 1)
  model %>% compile(loss = "mse",
                    optimizer = "adam",
                    metrics = list("mean_absolute_error"))
  model
}

model <- build_model(); model %>% summary()

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
) 

epochs <- 500

fitted_model_hp <- model %>% fit(train_data, train_target, 
                                 epochs = epochs,
                                 validation_split = 0.2,
                                 verbose = 0,
                                 callbacks = list(print_dot_callback))

library(ggplot2)
plot(fitted_model_hp, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
epochs <- 200
model <- build_model()
fitted_model_hp <- model %>% fit(train_data, train_target, 
                                 epochs = epochs,
                                 validation_split = 0.2,
                                 verbose = 0,
                                 callbacks = list(early_stop, print_dot_callback))

plot(fitted_model_hp, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

c(loss, mae) %<-% (model %>% evaluate(test_data, test_target, verbose = 0))
paste0("MAE on test set: $", sprintf("%.2f", mae*1000))

dropout_model <- build_model(dropout = 0.6)
dropout_fitted_model_hp <- dropout_model %>% fit(train_data, train_target,
                                                 epochs = 50,
                                                 verbose = 0,
                                                 batch_size = 25)
c(loss, mae) %<-% (dropout_model %>% evaluate(test_data, test_target, verbose = 0))
paste0("MAE of the dropout model on test set: $", sprintf("%.2f", mae*1000))

# Cross validation
fake_train_data <- matrix(seq(1, 50, 5), 10, 1)
indices <- sample(1:nrow(fake_train_data))
indices

folds <- cut(indices, breaks = 2, labels = FALSE)

k <- 5
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 50
all_scores <- c()
dim(train_data)


for(i in 1:k) {
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_target <- train_target[val_indices]
  partial_train_data <- train_data[-val_indices,]
  partial_train_target <- train_target[-val_indices]
  model <- build_model()
  model %>% fit(partial_train_data, partial_train_target,
                epochs = num_epochs, batch_size = 25, verbose = 0)
  results <- model %>% evaluate(val_data, val_target, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}
all_scores


# per-epoch
num_epochs <- 200
all_mae_histories <- NULL

for(i in 1:k) {
  i
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_target <- train_target[val_indices]
  partial_train_data <- train_data[-val_indices,]
  partial_train_target <- train_target[-val_indices]
  model <- build_model()
  history <- model %>% fit(partial_train_data, partial_train_target,
                           validation_data = list(val_data, val_target),
                           epochs = num_epochs, batch_size = 25, verbose = 0)
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}
all_mae_histories

average_mae_history <- data.frame(epoch = seq(1:ncol(all_mae_histories)), validation_mae = apply(all_mae_histories, 2, mean))

average_mae_history %>% 
  ggplot(aes(epoch, validation_mae)) + geom_smooth()

model <- build_model()
model %>% fit(train_data, train_target, epochs = 100, batch_size = 25, verbose = 0)
result <- model %>% evaluate(test_data, test_target)

##############################
install.packages("RSNNS")
library(RSNNS)

rm(list = ls())
x <- scale(rnorm(50, 0, 1))
y <- scale(rnorm(50, 0, 1))
net1 <- mlp(x = x, y = y, size = c(10), maxit = 10000, learnFuncParams = 0.0001, linOut = TRUE)

plot(x, y)
plot(y)
points(net1$fitted.values, col = "red", pch = 3)

asset <- data.frame(matrix(10, 2520, 1))
mu <- 0.05; sigma <- 0.6
for (i in 2:2520) {
  asset[i,1] <- asset[i-1,]*exp(rnorm(1, mu/252, sigma^2/252))
}
plot(asset[,1], type = 'l')

max_asset <- max(asset)
min_asset <- min(asset)

strikes <- seq(min_asset*0.5, max_asset*1.5, length.out = 100)
plot(strikes)

options <- data.frame(matrix(NA, dim(asset)[1], 3))
options[,1] <- sample(strikes, size = dim(asset)[1], replace = TRUE)
options[,2] <- asset[sample(dim(asset)[1]),]

install.packages("timeSeries")
library(timeSeries)

rtrn <- na.omit(timeSeries::returns(asset[,1]))
?returns
sigma_est <- sqrt(sd(rtrn[,1]) * 252)
mu_est <- mean(rtrn[,1])*252



#####
# CNN

rm(list = ls())
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images/255
str(train_images)
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images/255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = "relu", padding = "same",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu", padding = "same") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu", padding = "same")

summary(model)
model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 100, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
  )

model %>% fit(train_images, train_labels, epochs = 15, batch_size = 65)

test_results <- model %>% evaluate(test_images, test_labels); test_results
