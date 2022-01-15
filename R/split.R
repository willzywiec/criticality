# split.R
#
# William Zywiec
#
#' Split Function
#'
#' This function is a wrapper for the NN function, which subdivides the deep neural network metamodel based on fissile material form (e.g., "alpha", "delta", "heu").
#' @param dataset Training and test data
#' @param batch.size Batch size
#' @param ensemble.size Number of deep neural networks to train for the ensemble
#' @param epochs Number of training epochs
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param val.split Validation split
#' @param replot Boolean (TRUE/FALSE) that determines if plots should be regenerated
#' @param verbose Visualize TensorFlow output
#' @param ext.dir External directory
#' @export
#' @examples
#' Split(
#'   dataset,
#'   batch.size = 128,
#'   ensemble.size = 3,
#'   epochs = 50,
#'   layers = '8192-256-256-256-256-16',
#'   loss = 'sse',
#'   opt.alg = 'adamax',
#'   learning.rate = 0.00075,
#'   val.split = 0.2,
#'   replot = TRUE,
#'   verbose = TRUE,
#'   ext.dir = paste0(.libPaths(), "/criticality/example")
#' )

Split <- function(
  dataset,
  batch.size = 8192,
  ensemble.size = 5,
  epochs = 1500,
  layers = '8192-256-256-256-256-16',
  loss = 'sse',
  opt.alg = 'adamax',
  learning.rate = 0.00075,
  val.split = 0.2,
  replot = TRUE,
  verbose = TRUE,
  ext.dir) {

  # library(magrittr)

  form <- names(table(dataset$output$form))

  output <- training.data <- training.df <- test.data <- test.df <- list()

  for (i in 1:length(form)) {

    output[[i]] <- subset(dataset$output, form == form[i])

    j <- which(colnames(dataset$training.data) == paste0('form', form[i]))

    training.data[[i]] <- subset(dataset$training.data, dataset$training.data[[j]] == 1)
    training.df[[i]] <- as.data.frame(dataset$training.df)
    training.df[[i]] <- subset(training.df[[i]], training.df[[i]][[j]] == 1) %>% as.matrix()

    test.data[[i]] <- subset(dataset$test.data, dataset$test.data[[j]] == 1)
    test.df[[i]] <- as.data.frame(dataset$test.df)
    test.df[[i]] <- subset(test.df[[i]], test.df[[i]][[j]] == 1) %>% as.matrix()

  }

  training.mean <- dataset$training.mean
  training.sd <- dataset$training.sd

  dataset <- list()

  for (i in 1:length(form)) {
    dataset[[i]] <- list(output[[i]], training.data[[i]], training.mean, training.sd, training.df[[i]], test.data[[i]], test.df[[i]])
    names(dataset[[i]]) <- c('output', 'training.data', 'training.mean', 'training.sd', 'training.df', 'test.data', 'test.df')
  }

  metamodel <- list()

  for (i in 1:length(form)) {
    training.dir <- paste0(ext.dir, '/training/', form[i])
    dir.create(training.dir, recursive = TRUE, showWarnings = FALSE)
    metamodel[[i]] <- NN(dataset[[i]], batch.size, ensemble.size, epochs, layers, loss ,opt.alg, learning.rate, val.split, replot, verbose, training.dir)
  }

  names(metamodel) <- form

  return(metamodel)

}
