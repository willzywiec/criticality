# nn.R
#
#' NN Function
#'
#' This function imports the Tabulate, Scale, Model, Fit, Plot, and Test functions to train an ensemble of deep neural networks to predict keff values.
#' @param batch.size Batch size
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param ensemble.size Number of deep neural networks in the ensemble
#' @param epochs Number of training epochs
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param val.split Validation split
#' @param replot Boolean (TRUE/FALSE) that determines if .png files should be replotted
#' @param verbose Boolean (TRUE/FALSE) that determines if TensorFlow and Test function output should be displayed
#' @param ext.dir External directory (full path)
#' @param training.dir Training directory (full path)
#' @return A list of lists containing an ensemble of deep neural networks and weights
#' @export
#' @examples
#'
#' ext.dir <- paste0(tempdir(), "/criticality/extdata")
#' dir.create(ext.dir, recursive = TRUE, showWarnings = FALSE)
#'
#' extdata <- paste0(.libPaths()[1], "/criticality/extdata")
#' file.copy(paste0(extdata, "/facility.csv"), ext.dir, recursive = TRUE)
#' file.copy(paste0(extdata, "/mcnp-dataset.RData"), ext.dir, recursive = TRUE)
#'
#' config <- FALSE
#' try(config <- reticulate::py_config()$available)
#' try(if (config == TRUE) {
#'   NN(
#'     batch.size = 128,
#'     code = "mcnp",
#'     ensemble.size = 1,
#'     epochs = 10,
#'     layers = "8192-256-256-256-256-16",
#'     loss = "sse",
#'     opt.alg = "adamax",
#'     learning.rate = 0.00075,
#'     val.split = 0.2,
#'     replot = FALSE,
#'     verbose = FALSE,
#'     ext.dir = ext.dir
#'   )
#' })
#'
#' @import keras
#' @import magrittr
#' @import reticulate

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

  if (missing(training.dir)) training.dir <- ext.dir

  model.dir <- paste0(training.dir, '/model')
  dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)

  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)

  # build custom loss function
  if (loss == 'sse') loss <- SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

#
# train metamodel
#
  model.files <- list.files(path = model.dir, pattern = '\\.h5$')

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
      Plot(i = i, history = history[[i]], plot.dir = model.dir)
      save_model_hdf5(metamodel[[i]], paste0(model.dir, '/', i, '.h5'))
    }
  } else {
    model.files <- list.files(path = model.dir, pattern = '\\.h5$')
    for (i in 1:ensemble.size) {
      metamodel[[i]] <- load_model_hdf5(paste0(model.dir, '/', model.files[i]), custom_objects = c(loss = loss))
      if (replot == TRUE) Plot(i = i, plot.dir = model.dir)
    }
  }

#
# retrain metamodel
#
  remodel.files <- list.files(path = remodel.dir, pattern = '\\.h5$')

  history <- list()
  
  if (length(remodel.files) < ensemble.size * epochs / 10) {
    for (i in 1:ensemble.size) {
      remodel.files <- list.files(path = remodel.dir, pattern = paste0(i, '-.+\\.h5$'))
      if (length(remodel.files) < epochs / 10) {
        history[[i]] <- Fit(dataset, metamodel[[i]], batch.size, epochs, val.split, verbose, remodel.dir, i)
        Plot(i = i, history = history[[i]], plot.dir = remodel.dir)
      } else {
        Plot(i = i, plot.dir = remodel.dir)
      }
    }
  } else if (replot == TRUE) {
    for (i in 1:ensemble.size) Plot(i = i, plot.dir = remodel.dir)
  }

  # set metamodel weights and generate .csv predictions for all training and test data
  wt <- Test(code, dataset, ensemble.size, loss, verbose, ext.dir, training.dir)

  return(list(metamodel, wt))

}
