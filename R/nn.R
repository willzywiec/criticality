# nn.R
#
# William Zywiec
#
#' NN Function
#'
#' This function ties the Model, Fit, Plot, and Test functions together to build, train, and test a deep neural network metamodel.
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
#' @param training.dir Training directory
#' @export
#' @examples
#' NN(
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
#'   training.dir = paste0(.libPaths(), "/criticality/data")
#' )
#' @import keras
#' @import magrittr

NN <- function(
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
  verbose = FALSE,
  training.dir) {

  model.dir <- paste0(training.dir, '/model')
  dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)

  setwd(model.dir)

  # build custom loss function
  if (loss == 'sse') loss <- SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

  model.files <- list.files(pattern = '\\.h5$')

  # train metamodel
  metamodel <- history <- rep(list(0), length(ensemble.size))

  if (length(model.files) < ensemble.size) {
    for (i in (length(model.files) + 1):ensemble.size) {
      metamodel[[i]] <- Model(dataset, layers, loss, opt.alg, learning.rate)
      history[[i]] <- Fit(dataset, metamodel[[i]], batch.size, epochs, val.split, verbose)
      Plot(i, history[[i]])
      save_model_hdf5(metamodel[[i]], paste0(i, '.h5'))
    }
  } else if (replot == TRUE) {
    for (i in 1:ensemble.size) Plot(i)
  }

  model.files <- list.files(pattern = '\\.h5$')

  for (i in 1:ensemble.size) metamodel[[i]] <- load_model_hdf5(model.files[i], custom_objects = c(loss = loss))

  # retrain metamodel
  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(remodel.dir, showWarnings = FALSE)
  
  setwd(remodel.dir)
  
  remodel.files <- list.files(pattern = '\\.h5$')

  history <- list()
  
  if (length(remodel.files) < ensemble.size * epochs / 10) {
    for (i in 1:ensemble.size) {
      remodel.files <- list.files(pattern = paste0(i, '-.+\\.h5$'))
      if (length(remodel.files) < epochs / 10) {
        history[[i]] <- Fit(dataset, metamodel[[i]], batch.size, epochs / 10, val.split, verbose, remodel.dir, i)
        Plot(i, history[[i]])
      } else {
        Plot(i)
      }
    }
  } else if (replot == TRUE) {
    for (i in 1:ensemble.size) Plot(i)
  }
  
  # test metamodel
  training.mae <- val.mae <- numeric()
  
  for (i in 1:ensemble.size) {
    metrics <- read.csv(paste0(i, '.csv'))
    training.mae[i] <- metrics$mae[which.min(metrics$mae + metrics$val.mae)]
    val.mae[i] <- metrics$val.mae[which.min(metrics$mae + metrics$val.mae)]
    metamodel[[i]] <- load_model_hdf5(paste0(i, '-', metrics$epoch[which.min(metrics$mae + metrics$val.mae)], '.h5'), custom_objects = c(loss = loss))
  }
  
  wt <- Test(dataset, metamodel, training.mae, val.mae, training.dir)
  
  return(list(metamodel, wt))

}
