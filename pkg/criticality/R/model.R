# model.R
#
#' Model Function
#'
#' This function builds the deep neural network metamodel architecture.
#' @param dataset Training and test data
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param ext.dir External directory (full path)
#' @return A deep neural network metamodel of Monte Carlo radiation transport code simulation data
#' @export
#' @import keras
#' @import magrittr

Model <- function(
  dataset,
  layers = '8192-256-256-256-256-16',
  loss = 'sse',
  opt.alg = 'adamax',
  learning.rate = 0.00075,
  ext.dir) {

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
  }

  model <- model %>% layer_dense(units = 1, activation = 'linear')

  if (opt.alg == 'adadelta') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adadelta(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adagrad') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adagrad(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adamax') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adamax(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'nadam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_nadam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'rmsprop') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_rmsprop(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  }

}
