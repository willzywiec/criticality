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
#'   verbose = TRUE,
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
      metamodel[[i]] <- Model(code, layers, loss, opt.alg, learning.rate, ext.dir)
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

  Test <- function(code, dataset, ensemble.size, loss, ext.dir, training.dir) {

    training.mae <- val.mae <- numeric()

    for (i in 1:ensemble.size) {
      metrics <- read.csv(paste0(i, '.csv'))
      training.mae[i] <- metrics$mae[which.min(metrics$mae + metrics$val.mae)]
      val.mae[i] <- metrics$val.mae[which.min(metrics$mae + metrics$val.mae)]
      metamodel[[i]] <- load_model_hdf5(paste0(i, '-', metrics$epoch[which.min(metrics$mae + metrics$val.mae)], '.h5'), custom_objects = c(loss = loss))
    }

    setwd(training.dir)

    meta.len <- length(metamodel)

    test.data <- dataset$test.data
    test.pred <- matrix(nrow = nrow(dataset$test.df), ncol = meta.len)

    test.mae <- avg <- nm <- bfgs <- sa <- numeric()

    nm.wt <- bfgs.wt <- sa.wt <- list()

    Objective <- function(x) mean(abs(test.data$keff - rowSums(test.pred * x, na.rm = TRUE)))

    # minimize objective function
    for (i in 1:meta.len) {

      test.pred[ , i] <- metamodel[[i]] %>% predict(dataset$test.df)

      test.mae[i] <- mean(abs(test.data$keff - test.pred[ , i]))

      nm.wt[[i]] <- optim(rep(0, i), Objective, method = 'Nelder-Mead', lower = 0)
      bfgs.wt[[i]] <- optim(rep(0, i), Objective, method = 'BFGS', lower = 0)
      sa.wt[[i]] <- optim(rep(0, i), Objective, method = 'SANN', lower = 0)

      avg[i] <- mean(abs(test.data$keff - rowMeans(test.pred, na.rm = TRUE)))
      nm[i] <- mean(abs(test.data$keff - rowSums(test.pred * nm.wt[[i]][[1]], na.rm = TRUE)))
      bfgs[i] <- mean(abs(test.data$keff - rowSums(test.pred * bfgs.wt[[i]][[1]], na.rm = TRUE)))
      sa[i] <- mean(abs(test.data$keff - rowSums(test.pred * sa.wt[[i]][[1]], na.rm = TRUE)))

      if (i == 1) {
        cat('\n', sep = '')
        progress.bar <- txtProgressBar(min = 0, max = meta.len, style = 3)
        setTxtProgressBar(progress.bar, i)
        if (i == meta.len) {
          cat('\n\n', sep = '')
        }
      } else if (i == meta.len) {
        setTxtProgressBar(progress.bar, i)
        cat('\n\n', sep = '')
      } else {
        setTxtProgressBar(progress.bar, i)
      }

    }

    write.csv(data.frame(avg = avg, nm = nm, bfgs = bfgs, sa = sa), file = 'test-mae.csv', row.names = FALSE)

    cat('Mean Training MAE = ', mean(training.mae) %>% sprintf('%.6f', .), '\n', sep = '')
    cat('Mean Cross-Validation MAE = ', mean(val.mae) %>% sprintf('%.6f', .), '\n', sep = '')
    cat('Mean Test MAE = ', mean(test.mae) %>% sprintf('%.6f', .), '\n\n', sep = '')
    cat('Ensemble Test MAE = ', avg[meta.len] %>% sprintf('%.6f', .), '\n', sep = '')

    if (nm[meta.len] == bfgs[meta.len] && nm[meta.len] == sa[meta.len]) {
      cat.str <- ' (Nelder-Mead, BFGS, SA)\n'
    } else if (nm[meta.len] == bfgs[meta.len] && nm[meta.len] < sa[meta.len]) {
      cat.str <- ' (Nelder-Mead, BFGS)\n'
    } else if (nm[meta.len] == sa[meta.len] && nm[meta.len] < bfgs[meta.len]) {
      cat.str <- ' (Nelder-Mead, SA)\n'
    } else if (bfgs[meta.len] == sa[meta.len] && bfgs[meta.len] < nm[meta.len]) {
      cat.str <- ' (BFGS, SA)\n'
    } else if (nm[meta.len] < bfgs[meta.len] && nm[meta.len] < sa[meta.len]) {
      cat.str <- ' (Nelder-Mead)\n'
    } else if (bfgs[meta.len] < nm[meta.len] && bfgs[meta.len] < sa[meta.len]) {
      cat.str <- ' (BFGS)\n'
    } else if (sa[meta.len] < nm[meta.len] && sa[meta.len] < bfgs[meta.len]) {
      cat.str <- ' (SA)\n'
    }

    cat('Ensemble Test MAE = ', nm[meta.len] %>% sprintf('%.6f', .), cat.str, sep = '')

    test.min <- min(c(avg[which.min(avg)], nm[which.min(nm)], bfgs[which.min(bfgs)], sa[which.min(sa)]))

    if (test.min == nm[which.min(nm)]) {
      wt <- nm.wt[[which.min(nm)]]$par
    } else if (test.min == bfgs[which.min(bfgs)]) {
      wt <- bfgs.wt[[which.min(bfgs)]]$par
    } else if (test.min == sa[which.min(sa)]) {
      wt <- sa.wt[[which.min(sa)]]$par
    } else {
      wt <- 0
    }

    wt.len <- length(wt)

    if (wt.len < meta.len && wt[1] != 0) {

      if (wt.len == 1) {
        cat('-\nTest MAE reaches a local minimum with ', wt.len, ' neural network\n\n', sep = '')
      } else {
        cat('-\nTest MAE reaches a local minimum with ', wt.len, ' neural networks\n\n', sep = '')
      }

      cat('Ensemble Test MAE = ', avg[wt.len] %>% sprintf('%.6f', .), '\n', sep = '')

      if (nm[wt.len] == bfgs[wt.len] && nm[wt.len] == sa[wt.len]) {
        cat.str <- ' (Nelder-Mead, BFGS, SA)\n'
      } else if (nm[wt.len] == bfgs[wt.len] && nm[wt.len] < sa[wt.len]) {
        cat.str <- ' (Nelder-Mead, BFGS)\n'
      } else if (nm[wt.len] == sa[wt.len] && nm[wt.len] < bfgs[wt.len]) {
        cat.str <- ' (Nelder-Mead, SA)\n'
      } else if (bfgs[wt.len] == sa[wt.len] && bfgs[wt.len] < nm[wt.len]) {
        cat.str <- ' (BFGS, SA)\n'
      } else if (nm[wt.len] < bfgs[wt.len] && nm[wt.len] < sa[wt.len]) {
        cat.str <- ' (Nelder-Mead)\n'
      } else if (bfgs[wt.len] < nm[wt.len] && bfgs[wt.len] < sa[wt.len]) {
        cat.str <- ' (BFGS)\n'
      } else if (sa[wt.len] < nm[wt.len] && sa[wt.len] < bfgs[wt.len]) {
        cat.str <- ' (SA)\n'
      }

      cat('Ensemble Test MAE = ', nm[meta.len] %>% sprintf('%.6f', .), cat.str, sep = '')

    }

    # adjust predicted keff values
    if (!file.exists('training-data.csv') || !file.exists('test-data.csv')) {

      training.data <- dataset$training.data

      training.pred <- matrix(nrow = nrow(dataset$training.df), ncol = wt.len)

      if (wt.len == 1) {

        training.pred[ , 1] <- metamodel[[1]] %>% predict(dataset$training.df)

        training.data$avg <- training.pred[ , 1]
        training.data$nm <- training.pred[ , 1] * nm.wt[[wt.len]][[1]]
        training.data$bfgs <- training.pred[ , 1] * bfgs.wt[[wt.len]][[1]]
        training.data$sa <- training.pred[ , 1] * sa.wt[[wt.len]][[1]]

        test.data$avg <- test.pred[ , 1]
        test.data$nm <- test.pred[ , 1] * nm.wt[[wt.len]][[1]]
        test.data$bfgs <- test.pred[ , 1] * bfgs.wt[[wt.len]][[1]]
        test.data$sa <- test.pred[ , 1] * sa.wt[[wt.len]][[1]]

      } else {

        for (i in 1:wt.len) {
          training.pred[ , i] <- metamodel[[i]] %>% predict(dataset$training.df)
        }

        training.data$avg <- rowMeans(training.pred[ , 1:wt.len])
        training.data$nm <- rowSums(training.pred[ , 1:wt.len] * nm.wt[[wt.len]][[1]])
        training.data$bfgs <- rowSums(training.pred[ , 1:wt.len] * bfgs.wt[[wt.len]][[1]])
        training.data$sa <- rowSums(training.pred[ , 1:wt.len] * sa.wt[[wt.len]][[1]])

        test.data$avg <- rowMeans(test.pred[ , 1:wt.len])
        test.data$nm <- rowSums(test.pred[ , 1:wt.len] * nm.wt[[wt.len]][[1]])
        test.data$bfgs <- rowSums(test.pred[ , 1:wt.len] * bfgs.wt[[wt.len]][[1]])
        test.data$sa <- rowSums(test.pred[ , 1:wt.len] * sa.wt[[wt.len]][[1]])

      }

      write.csv(training.data, file = 'training-data.csv', row.names = FALSE)
      write.csv(test.data, file = 'test-data.csv', row.names = FALSE)

    }

    return(wt)

  }

  # optimize existing metamodel and generate .csv predictions for all training and test data
  wt <- Test(code, dataset, ensemble.size, loss, ext.dir, training.dir)

  return(list(metamodel, wt))

}
