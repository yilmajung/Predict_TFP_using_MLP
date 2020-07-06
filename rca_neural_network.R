# Termpaper
# Load library
library(haven)
library(ggplot2)
library(tidyverse)
library(plyr)
library(tidytext)
library(ggrepel)
library(keras)

getwd()
head(table)
unique(table$year)
unique(table$location_id)

pid <- read_dta("Termpaper/dataset/hs_product.dta")
head(pid)
loc <- read_dta("Termpaper/dataset/location.dta")
head(loc)
head(table)

df <- merge(table, pid, by="product_id", all.x = TRUE)
df <- merge(df, loc[,c(1,2,3,4)], by="location_id", all.x = TRUE)

# SITC
rm(list = ls())
head(table)

# Load SITC Product code and Location code
pid <- read_dta("Termpaper/dataset/sitc_product.dta")
loc <- read_dta("Termpaper/dataset/location.dta")
head(pid)
head(loc)

# Merge pid and loc into main data frame
df <- merge(table, pid[,-5], by="product_id", all.x = TRUE)
df <- merge(df, loc[,c(1,2,3,4)], by="location_id", all.x = TRUE)

# Select variables needed for this study
df <- df %>% 
  select(location_id, product_id, year, export_rca, sitc_product_name_short_en, location_code.x, location_name_short_en)

# Change column name
colnames(df) <- c("lid", "pid", "year", "rca", "product", "iso", "country")
unique(df$year)

# Load gdp growth
gdp <- read_csv("Termpaper/dataset/gdp_growth.csv")
head(gdp)

########################
# Transform to long data
head(gdp)
gdp <- gather(gdp, year, growth, `1961`:`2018`)
gdp %>% 
  group_by(year) %>% 
  mutate(growth_5 = ntile(growth, 5)) -> gdp
gdp <- gdp[,-4]
##########################

# Transform to long data
head(gdp)
gdp <- gather(gdp, year, growth, `1961`:`2018`)
?quantile
quantile(gdp$growth, na.rm = TRUE)
gdp %>% 
  mutate(growth_5 = ntile(growth, 4)) -> gdp
gdp <- gdp[,-4]



# Merge growth into df
head(df)
df2 <- merge(df, gdp, by = c("iso", "year"), all.x = TRUE)

# Remain only complete cases
df2 <- df2 %>% 
  drop_na()

# DF3
df3 <- df2 %>% 
  select(iso, year, pid, rca, growth_5)

head(df3)
df3 <- df3 %>% 
  spread(pid, rca)

head(df3)
df4 <- df3 %>% 
  mutate_at(vars(3:785), funs(ifelse(is.na(.), 0, .)))
df4 <- df4 %>% 
  mutate(growth_5 = ifelse(growth_5 == 1, 0,
                           ifelse(growth_5 == 2, 1,
                                  ifelse(growth_5 == 3, 2,
                                         ifelse(growth_5 == 4, 3, 4)))))
str(df4)
head(df4)
df5 <- df4[,-c(1,2)]

# Transformation into a matrix
df.mat <- as.matrix(df4)
dimnames(df.mat) = NULL

# Seperate into train and test set
set.seed(123)
idx <- sample(2, nrow(df.mat), replace = TRUE, prob = c(0.9, 0.1))
head(df3)
x_train <- df.mat[idx == 1, 4:785]
x_test <- df.mat[idx == 2, 4:785]

y_test_actual <- df.mat[idx == 2, 3]

y_train <- to_categorical(df.mat[idx == 1, 3])
y_test <- to_categorical(df.mat[idx == 2, 3])

cbind(y_test_actual[1:10], y_test[1:10,])


# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 20, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "rmsprop",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(x_train, y_train, epoch = 10, batch_size = 30, validation_split = 0.1, verbose = 2)



###########################
# Transformation into a matrix
df.mat <- as.matrix(df5)
dimnames(df.mat) = NULL

# Seperate into train and test set
set.seed(123)
idx <- sample(2, nrow(df.mat), replace = TRUE, prob = c(0.9, 0.1))
head(df5)
x_train <- df.mat[idx == 1, 2:783]
x_test <- df.mat[idx == 2, 2:783]

y_test_actual <- df.mat[idx == 2, 1]

y_train <- to_categorical(df.mat[idx == 1, 1])
y_test <- to_categorical(df.mat[idx == 2, 1])

cbind(y_test_actual[1:10], y_test[1:10,])


# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 70, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 40, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(x_train, y_train, epoch = 10, batch_size = 30, validation_split = 0.1, verbose = 2)

############
# Binary
head(df5)
df6 <- df5 %>% 
  mutate_at(vars(2:783), funs(ifelse(.>=1, 1, 0)))
head(df6)

# Transformation into a matrix
df.mat <- as.matrix(df6)
dimnames(df.mat) = NULL

# Seperate into train and test set
set.seed(123)
idx <- sample(2, nrow(df.mat), replace = TRUE, prob = c(0.9, 0.1))
head(df6)
x_train <- df.mat[idx == 1, 2:783]
x_test <- df.mat[idx == 2, 2:783]

y_test_actual <- df.mat[idx == 2, 1]

y_train <- to_categorical(df.mat[idx == 1, 1])
y_test <- to_categorical(df.mat[idx == 2, 1])

cbind(y_test_actual[1:10], y_test[1:10,])


# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 70, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 40, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "rmsprop",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(x_train, y_train, epoch = 10, batch_size = 30, verbose = 2)

metrics <- model %>% 
  evaluate(x_test, y_test)
metrics

model %>% predict_classes(x_test[1:10,])
y_test_actual[1:10]
length(unique(table$product_id))

unique(df3$year)
length(unique(df2$pid))

head(df2)
df2 %>% 
  group_by(year) %>% 
  summarise(nc = count(iso))

################################################
# TFP

pwt <- read_csv("Termpaper/dataset/penntable.csv")
head(pwt)
pwt <- pwt %>% 
  select(iso, country, year, ctfp)


# Transform to long data
head(pwt)
quantile(pwt$ctfp, na.rm = TRUE)
pwt %>% 
  mutate(tfp = ntile(ctfp, 5)) -> pwt

pwt <- pwt[,-4]

# Merge TFP into df
head(df)
df_tfp <- merge(df, pwt, by = c("iso", "year"), all.x = TRUE)
unique(df$iso)
unique(pwt$iso)

head(df_tfp)


# Remain only complete cases
df_tfp <- df_tfp %>% 
  drop_na()

head(df_tfp)
df_tfp <- df_tfp %>% 
  select(iso, year, pid, rca, tfp)

# Transform to wide
df_tfp <- df_tfp %>% 
  spread(pid, rca)

head(df_tfp)
df_tfp2 <- df_tfp %>% 
  mutate_at(vars(3:785), funs(ifelse(is.na(.), 0, .)))

df_tfp2 <- df_tfp2 %>% 
  mutate(tfp = ifelse(tfp == 1, 0,
                           ifelse(tfp == 2, 1,
                                  ifelse(tfp == 3, 2,
                                         ifelse(tfp == 4, 3, 4)))))
str(df_tfp2)

df_tfp3 <- df_tfp2[,-c(1,2)]

# Transformation into a matrix
mat.tfp <- as.matrix(df_tfp3)
dimnames(mat.tfp) = NULL

# Seperate into train and test set
set.seed(123)
idx <- sample(2, nrow(mat.tfp), replace = TRUE, prob = c(0.8, 0.2))
head(mat.tfp)
tfp_x_train <- mat.tfp[idx == 1, 2:783]
tfp_x_test <- mat.tfp[idx == 2, 2:783]

tfp_y_test_actual <- mat.tfp[idx == 2, 1]

tfp_y_train <- to_categorical(mat.tfp[idx == 1, 1])
tfp_y_test <- to_categorical(mat.tfp[idx == 2, 1])

cbind(tfp_y_test_actual[1:10], tfp_y_test[1:10,])


# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 50, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 30, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)




################################################
# TFP

pwt <- read_csv("Termpaper/dataset/penntable.csv")
head(pwt)
pwt <- pwt %>% 
  select(iso, country, year, ctfp)


# Transform to long data
head(pwt)
pwt %>% 
  mutate(tfp = ntile(ctfp, 5)) -> pwt

pwt <- pwt[,-4]

# Merge TFP into df
head(df)
df_tfp <- merge(df, pwt, by = c("iso", "year"), all.x = TRUE)
unique(df$iso)
unique(pwt$iso)

head(df_tfp)

# Remain only complete cases
df_tfp <- df_tfp %>% 
  drop_na()

head(df_tfp)
df_tfp <- df_tfp %>% 
  select(iso, year, pid, rca, tfp)

# Transform to wide
df_tfp <- df_tfp %>% 
  spread(pid, rca)

head(df_tfp)

df_tfp2 <- df_tfp %>% 
  mutate_at(vars(3:785), funs(ifelse(is.na(.), 0, .)))

df_tfp2 <- df_tfp2 %>% 
  mutate(tfp = ifelse(tfp == 1, 0,
                      ifelse(tfp == 2, 1,
                             ifelse(tfp == 3, 2,
                                    ifelse(tfp == 4, 3, 4)))))

# Binary
head(df_tfp2)
df_tfp2 <- df_tfp2 %>% 
  mutate_at(vars(4:785), funs(ifelse(.>=1, 1, 0)))
str(df_tfp2)

df_tfp3 <- df_tfp2[,-c(1,2)]

# Transformation into a matrix
mat.tfp <- as.matrix(df_tfp3)
dimnames(mat.tfp) = NULL

# Seperate into train and test set
set.seed(123)
idx <- sample(2, nrow(mat.tfp), replace = TRUE, prob = c(0.8, 0.2))
head(mat.tfp)
tfp_x_train <- mat.tfp[idx == 1, 2:783]
tfp_x_test <- mat.tfp[idx == 2, 2:783]

tfp_y_test_actual <- mat.tfp[idx == 2, 1]

tfp_y_train <- to_categorical(mat.tfp[idx == 1, 1])
tfp_y_test <- to_categorical(mat.tfp[idx == 2, 1])

cbind(tfp_y_test_actual[1:10], tfp_y_test[1:10,])


# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 50, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 30, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)


# smaller model
# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 10, activation = "relu", input_shape = 782) %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)


# L2 Regularization
# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 50, activation = "relu", input_shape = 782,
              kernel_regularizer = regularizer_l2(0.1)) %>% 
  layer_dense(units = 30, activation = "relu",
              kernel_regularizer = regularizer_l2(0.1)) %>% 
  layer_dense(units = 20, activation = "relu",
              kernel_regularizer = regularizer_l2(0.1)) %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model 
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)



# Dropout
# Create a model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 50, activation = "relu", input_shape = 782) %>% 
  layer_dropout(0.6) %>% 
  layer_dense(units = 30, activation = "relu") %>% 
  layer_dropout(0.6) %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dropout(0.6) %>% 
  layer_dense(units = 5, activation = "softmax") 
summary(model)

# Compile the model 
model %>% 
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit the data
history <- model %>% 
  fit(tfp_x_train, tfp_y_train, epoch = 10, batch_size = 64, validation_split = 0.1, verbose = 2)

?fit
?keras::fit
