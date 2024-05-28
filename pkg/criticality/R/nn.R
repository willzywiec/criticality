# nn.R
#
#' NN Function
#'
#' This function trains an ensemble of deep neural networks to predict keff values (imports Tabulate, Scale, Model, Fit, Plot, and Test functions).
#' @param batch.size Batch size
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dataset Training and test data
#' @param ensemble.size Number of deep neural networks in the ensemble
#' @param epochs Number of training epochs
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param val.split Validation split
#' @param overwrite Boolean (TRUE/FALSE) that determines if files should be overwritten
#' @param replot Boolean (TRUE/FALSE) that determines if .png files should be replotted
#' @param reweight Boolean (TRUE/FALSE) that determines if metamodel weights should be recalculated
#' @param verbose Boolean (TRUE/FALSE) that determines if TensorFlow and Fit function output should be displayed
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
#'     ensemble.size = 1,
#'     epochs = 10,
#'     layers = "256-256-16",
#'     loss = "sse",
#'     replot = FALSE,
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
  dataset,
  ensemble.size = 5,
  epochs = 1500,
  layers = '8192-256-256-256-256-16',
  loss = 'sse',
  opt.alg = 'adamax',
  learning.rate = 0.00075,
  val.split = 0.2,
  overwrite = FALSE,
  replot = TRUE,
  reweight = FALSE,
  verbose = FALSE,
  ext.dir,
  training.dir = NULL) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  if (is.null(training.dir)) training.dir <- paste0(ext.dir, '/training')

  model.dir <- paste0(training.dir, '/model')
  dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)

  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)

  new.settings <- data.frame(V1 = c(
    'model settings',
    paste0('batch size: ', batch.size),
    paste0('code: ', code),
    paste0('ensemble size: ', ensemble.size),
    paste0('epochs: ', epochs),
    paste0('layers: ', layers),
    paste0('loss: ', loss),
    paste0('optimization algorithm: ', opt.alg),
    paste0('learning rate: ', learning.rate),
    paste0('validation split: ', val.split),
    paste0('external directory: ', ext.dir),
    paste0('training directory: ', training.dir)))

#
# check metamodel settings
#
  if (file.exists(paste0(training.dir, '/model-settings.txt'))) {
    old.settings <- utils::read.table(paste0(training.dir, '/model-settings.txt'), sep = '\n') %>% as.data.frame()
    if (!identical(new.settings[-4, ], old.settings[-4, ])) {
      if (overwrite == TRUE) {
        unlink(model.dir, recursive = TRUE)
        unlink(remodel.dir, recursive = TRUE)
        utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), quote = FALSE, row.names = FALSE, col.names = FALSE)
        dir.create(model.dir, recursive = TRUE, showWarnings = FALSE)
        dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)
      } else {
        stop('files could not be overwritten', call. = FALSE)
      }
    } else {
      utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
  } else {
    utils::write.table(new.settings, file = paste0(training.dir, '/model-settings.txt'), quote = FALSE, row.names = FALSE, col.names = FALSE)
  }

  # build sum of squared errors (SSE) loss function
  if (loss == 'sse') loss <- SSE <- function(y_true, y_pred) k_sum(k_pow(y_true - y_pred, 2))

#
# load metamodel
#
  if (
    file.exists(paste0(training.dir, '/metamodel.RData')) &&
    identical(new.settings[-4, ], old.settings[-4, ]) &&
    ensemble.size == dim(utils::read.csv(paste0(training.dir, '/test-mae.csv')))[1] && # check that ensemble.size is equal to the number of rows in test-mae.csv
    ensemble.size <= length(list.files(path = model.dir)[grep('.*h5$', list.files(path = model.dir))]) && # redundant check to ensure files haven't been deleted
    reweight == FALSE) {

    wt <- min.wt <- numeric() # 'min.wt' must be defined prior to loading 'metamodel.RData'

    load(paste0(training.dir, '/metamodel.RData')) # load 'min.wt'

    metamodel <- rep(list(0), length(ensemble.size))

    for (i in 1:ensemble.size) {
      metamodel[[i]] <- load_model_hdf5(paste0(remodel.dir, '/', i, '-', min.wt[[1]][[i]], '.h5'), custom_objects = c(loss = loss))
      wt[i] <- min.wt[[2]][[i]]
    }
#
# train metamodel
#
  } else {

    model.files <- list.files(path = model.dir, pattern = '\\.h5$')

    metamodel <- history <- rep(list(0), length(ensemble.size))

    Fit <- function(dataset, model, batch.size, epochs, val.split, verbose, remodel.dir, i = NULL) {
      if (is.null(i)) {
        model %>% keras::fit(
          dataset$training.df,
          dataset$training.data$keff,
          batch_size = batch.size,
          epochs = epochs,
          validation_split = val.split,
          verbose = verbose)
      } else {
        checkpoint <- callback_model_checkpoint(paste0(remodel.dir, '/', i, '-{epoch:1d}.h5'), monitor = 'mean_absolute_error')
        model %>% keras::fit(
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
        metamodel[[i]] <- Model(dataset, layers, loss, opt.alg, learning.rate, ext.dir)
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
# retrain metamodel (epochs / 10)
#
    remodel.files <- list.files(path = remodel.dir, pattern = '\\.h5$')

    history <- rep(list(0), length(ensemble.size))
  
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

    # set metamodel weights, generate predictions for all training and test data, and save predictions as a .csv file
    wt <- Test(dataset, ensemble.size, loss, ext.dir, training.dir)

  }

  return(list(metamodel, wt))

}
