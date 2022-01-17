# nn.R
#
# William Zywiec
#
#' NN Function
#'
#' This function ties the Model, Fit, Plot, and Test functions together to build, train, and test a deep neural network metamodel.
#' @param batch.size Batch size
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param ensemble.size Number of deep neural networks in the ensemble
#' @param epochs Number of training epochs
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param val.split Validation split
#' @param replot Boolean (TRUE/FALSE) that determines if plots should be regenerated
#' @param verbose Visualize TensorFlow output
#' @param ext.dir External directory
#' @param training.dir Training directory
#' @export
#' @examples
#' NN(
#'   batch.size = 128,
#'   code = "mcnp",
#'   ensemble.size = 1,
#'   epochs = 10,
#'   layers = "8192-256-256-256-256-16",
#'   loss = "sse",
#'   opt.alg = "adamax",
#'   learning.rate = 0.00075,
#'   val.split = 0.2,
#'   replot = TRUE,
#'   verbose = FALSE,
#'   ext.dir = paste0(.libPaths()[1], "/criticality/data"),
#'   training.dir = paste0(.libPaths()[1], "/criticality/data")
#' )
#' @import keras
#' @import magrittr

NN <- function(
  batch.size = 8192,
  code = 'mcnp',
  ensemble.size = 5,
  epochs = 1500,
  layers = '8192-256-256-256-256-16',
  loss = 'sse',
  opt.alg = 'adamax',
  learning.rate = 0.00075,
  val.split = 0.2,
  replot = TRUE,
  verbose = FALSE,
  ext.dir,
  training.dir) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  model.dir <- paste0(training.dir, '/model')
  dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)

  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)

  setwd(model.dir)

  # build custom loss function
  if (loss == 'sse') loss <- SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

  model.files <- list.files(pattern = '\\.h5$')

  # train metamodel
  metamodel <- history <- rep(list(0), length(ensemble.size))

  Fit <- function(dataset, model, batch.size, epochs, val.split, verbose, remodel.dir, i) {
    if (missing(i)) {
      model %>% fit(
        dataset$training.df,
        dataset$training.data$keff,
        batch_size = batch.size,
        epochs = epochs,
        validation_split = val.split,
        verbose = verbose)
    } else {
      checkpoint <- callback_model_checkpoint(paste0(remodel.dir, '/', i, '-{epoch:1d}.h5'), monitor = 'mean_absolute_error')
      model %>% fit(
        dataset$training.df,
        dataset$training.data$keff,
        batch_size = batch.size,
        epochs = epochs / 10,
        validation_split = val.split,
        verbose = verbose,
        callbacks = c(checkpoint))
    }
  }

  if (length(model.files) < ensemble.size) {
    for (i in (length(model.files) + 1):ensemble.size) {
      metamodel[[i]] <- Model(code, dataset, layers, loss, opt.alg, learning.rate, ext.dir)
      history[[i]] <- Fit(dataset, metamodel[[i]], batch.size, epochs, val.split, verbose)
      Plot(i, history[[i]])
      save_model_hdf5(metamodel[[i]], paste0(i, '.h5'))
    }
  } else {
    model.files <- list.files(pattern = '\\.h5$')
    for (i in 1:ensemble.size) {
      metamodel[[i]] <- load_model_hdf5(model.files[i], custom_objects = c(loss = loss))
      if (replot == TRUE) Plot(i)
    }
  }

  # retrain metamodel
  setwd(remodel.dir)
  
  remodel.files <- list.files(pattern = '\\.h5$')

  history <- list()
  
  if (length(remodel.files) < ensemble.size * epochs / 10) {
    for (i in 1:ensemble.size) {
      remodel.files <- list.files(pattern = paste0(i, '-.+\\.h5$'))
      if (length(remodel.files) < epochs / 10) {
        history[[i]] <- Fit(dataset, metamodel[[i]], batch.size, epochs, val.split, verbose, remodel.dir, i)
        Plot(i, history[[i]])
      } else {
        Plot(i)
      }
    }
  } else if (replot == TRUE) {
    for (i in 1:ensemble.size) Plot(i)
  }

  # optimize existing metamodel and generate .csv predictions for all training and test data
  wt <- Test(code, dataset, ensemble.size, loss, ext.dir, training.dir)

  return(list(metamodel, wt))

}
