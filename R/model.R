# model.R
#
# William Zywiec

Model <- function(dataset, layers, loss, opt.alg, learning.rate) {

  # library(keras)
  # library(magrittr)

  layers <- strsplit(layers, '-') %>% unlist() %>% as.integer()

  model <- keras_model_sequential() %>% layer_dense(units = layers[1], activation = 'relu', input_shape = dim(dataset$training.df)[2])

  if (length(layers) >= 2) {
    model <- model %>% layer_dense(units = layers[2], activation = 'relu')
  } else if (length(layers) >= 3) {
    model <- model %>% layer_dense(units = layers[3], activation = 'relu')
  } else if (length(layers) >= 4) {
    model <- model %>% layer_dense(units = layers[4], activation = 'relu')
  } else if (length(layers) >= 5) {
    model <- model %>% layer_dense(units = layers[5], activation = 'relu')
  } else if (length(layers) >= 6) {
    model <- model %>% layer_dense(units = layers[6], activation = 'relu')
  } else if (length(layers) >= 7) {
    model <- model %>% layer_dense(units = layers[7], activation = 'relu')
  } else if (length(layers) == 8) {
    model <- model %>% layer_dense(units = layers[8], activation = 'relu')
  } else if (length(layers) == 9) {
    model <- model %>% layer_dense(units = layers[9], activation = 'relu')
  } else if (length(layers) == 10) {
    model <- model %>% layer_dense(units = layers[10], activation = 'relu')
  }

  model <- model %>% layer_dense(units = 1, activation = 'linear')

  if (opt.alg == 'adadelta') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adadelta(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adagrad') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adagrad(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adam(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adamax') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adamax(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'nadam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_nadam(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'rmsprop') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_rmsprop(lr = learning.rate),
      metrics = c('mean_absolute_error'))
  }

}
