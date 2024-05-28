# test.R
#
#' Test Function
#'
#' This function calculates deep neural network metamodel weights and generates keff predictions for all training and test data.
#' @param dataset Training and test data
#' @param ensemble.size Number of deep neural networks in the ensemble
#' @param loss Loss function
#' @param ext.dir External directory (full path)
#' @param training.dir Training directory (full path)
#' @return A list of deep neural network weights
#' @export
#' @import magrittr

Test <- function(
  dataset,
  ensemble.size = 5,
  loss = 'sse',
  ext.dir,
  training.dir = NULL) {

  if (is.null(training.dir)) training.dir <- paste0(ext.dir, '/training')

  remodel.dir <- paste0(training.dir, '/remodel')
  dir.create(remodel.dir, recursive = TRUE, showWarnings = FALSE)

  min.mae <- training.mae <- val.mae <- numeric()

  metamodel <- rep(list(0), length(ensemble.size))

  for (i in 1:ensemble.size) {
    metrics <- utils::read.csv(paste0(remodel.dir, '/', i, '.csv'))
    min.mae[i] <- which.min(metrics$mae + metrics$val.mae)
    training.mae[i] <- metrics$mae[min.mae[i]]
    val.mae[i] <- metrics$val.mae[min.mae[i]]
    metamodel[[i]] <- load_model_hdf5(paste0(remodel.dir, '/', i, '-', metrics$epoch[min.mae[i]], '.h5'), custom_objects = c(loss = loss))
  }

#
# minimize objective function
#
  test.data <- dataset$test.data

  test.pred <- matrix(nrow = nrow(dataset$test.df), ncol = ensemble.size)

  test.mae <- avg <- nm <- bfgs <- sa <- numeric()

  nm.wt <- bfgs.wt <- sa.wt <- rep(list(0), length(ensemble.size))

  Objective <- function(x) mean(abs(test.data$keff - rowSums(test.pred * x, na.rm = TRUE))) %>% suppressWarnings()

  progress.bar <- utils::txtProgressBar(max = ensemble.size, style = 3)

  for (i in 1:ensemble.size) {

    test.pred[ , i] <- metamodel[[i]] %>% stats::predict(dataset$test.df, verbose = FALSE)

    test.mae[i] <- mean(abs(test.data$keff - test.pred[ , i]))

    nm.wt[[i]] <- stats::optim(rep(1 / i, i), Objective, method = 'Nelder-Mead') %>% suppressWarnings()
    bfgs.wt[[i]] <- stats::optim(rep(1 / i, i), Objective, method = 'BFGS')
    sa.wt[[i]] <- stats::optim(rep(1 / i, i), Objective, method = 'SANN')

    avg[i] <- mean(abs(test.data$keff - rowMeans(test.pred, na.rm = TRUE)))
    nm[i] <- mean(abs(test.data$keff - rowSums(test.pred * nm.wt[[i]][[1]], na.rm = TRUE))) %>% suppressWarnings()
    bfgs[i] <- mean(abs(test.data$keff - rowSums(test.pred * bfgs.wt[[i]][[1]], na.rm = TRUE))) %>% suppressWarnings()
    sa[i] <- mean(abs(test.data$keff - rowSums(test.pred * sa.wt[[i]][[1]], na.rm = TRUE))) %>% suppressWarnings()

    utils::setTxtProgressBar(progress.bar, i)

  }

  close(progress.bar)
  
  utils::write.csv(data.frame(avg = avg, nm = nm, bfgs = bfgs, sa = sa), file = paste0(training.dir, '/test-mae.csv'), row.names = FALSE)

  message('Mean Training MAE = ', sprintf('%.6f', mean(training.mae)))
  message('Mean Cross-Validation MAE = ', sprintf('%.6f', mean(val.mae)))
  message('Mean Test MAE = ', sprintf('%.6f', mean(test.mae)))
  message('Ensemble Test MAE = ', sprintf('%.6f', avg[ensemble.size]))

  if (nm[ensemble.size] == bfgs[ensemble.size] && nm[ensemble.size] == sa[ensemble.size]) {
    msg.str <- ' (Nelder-Mead, BFGS, SA)'
  } else if (nm[ensemble.size] == bfgs[ensemble.size] && nm[ensemble.size] < sa[ensemble.size]) {
    msg.str <- ' (Nelder-Mead, BFGS)'
  } else if (nm[ensemble.size] == sa[ensemble.size] && nm[ensemble.size] < bfgs[ensemble.size]) {
    msg.str <- ' (Nelder-Mead, SA)'
  } else if (bfgs[ensemble.size] == sa[ensemble.size] && bfgs[ensemble.size] < nm[ensemble.size]) {
    msg.str <- ' (BFGS, SA)'
  } else if (nm[ensemble.size] < bfgs[ensemble.size] && nm[ensemble.size] < sa[ensemble.size]) {
    msg.str <- ' (Nelder-Mead)'
  } else if (bfgs[ensemble.size] < nm[ensemble.size] && bfgs[ensemble.size] < sa[ensemble.size]) {
    msg.str <- ' (BFGS)'
  } else if (sa[ensemble.size] < nm[ensemble.size] && sa[ensemble.size] < bfgs[ensemble.size]) {
    msg.str <- ' (SA)'
  }

  obj.min <- min(c(nm[ensemble.size], bfgs[ensemble.size], sa[ensemble.size]))

  message('Ensemble Test MAE = ', sprintf('%.6f', obj.min), msg.str)

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

  if (wt.len < ensemble.size && wt[1] != 0) {

    if (wt.len == 1) {
      message('-\nTest MAE reaches a local minimum with ', wt.len, ' neural network')
    } else {
      message('-\nTest MAE reaches a local minimum with ', wt.len, ' neural networks')
    }

    message('Ensemble Test MAE = ', sprintf('%.6f', avg[wt.len]))

    if (nm[wt.len] == bfgs[wt.len] && nm[wt.len] == sa[wt.len]) {
      msg.str <- ' (Nelder-Mead, BFGS, SA)'
    } else if (nm[wt.len] == bfgs[wt.len] && nm[wt.len] < sa[wt.len]) {
      msg.str <- ' (Nelder-Mead, BFGS)'
    } else if (nm[wt.len] == sa[wt.len] && nm[wt.len] < bfgs[wt.len]) {
      msg.str <- ' (Nelder-Mead, SA)'
    } else if (bfgs[wt.len] == sa[wt.len] && bfgs[wt.len] < nm[wt.len]) {
      msg.str <- ' (BFGS, SA)'
    } else if (nm[wt.len] < bfgs[wt.len] && nm[wt.len] < sa[wt.len]) {
      msg.str <- ' (Nelder-Mead)'
    } else if (bfgs[wt.len] < nm[wt.len] && bfgs[wt.len] < sa[wt.len]) {
      msg.str <- ' (BFGS)'
    } else if (sa[wt.len] < nm[wt.len] && sa[wt.len] < bfgs[wt.len]) {
      msg.str <- ' (SA)'
    }

    obj.min <- min(c(nm[ensemble.size], bfgs[ensemble.size], sa[ensemble.size]))

    message('Ensemble Test MAE = ', sprintf('%.6f', obj.min), msg.str)
    
  }

  if (abs(obj.min - avg[ensemble.size]) > 0.1 || abs(mean(training.mae) - mean(test.mae)) > 0.1) {
    message('-\nWarning: weights do not converge')
  }

#
# set metamodel weights
#
  training.data <- dataset$training.data

  training.pred <- matrix(nrow = nrow(dataset$training.df), ncol = wt.len)

  if (wt.len == 1) {

    training.pred[ , 1] <- metamodel[[1]] %>% stats::predict(dataset$training.df, verbose = FALSE)

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
      training.pred[ , i] <- metamodel[[i]] %>% stats::predict(dataset$training.df, verbose = FALSE)
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

  min.wt <- list(min.mae, wt)

  save(min.wt, file = paste0(training.dir, '/metamodel.RData'), compress = 'xz')

  utils::write.csv(training.data, file = paste0(training.dir, '/training-data.csv'), row.names = FALSE)
  utils::write.csv(test.data, file = paste0(training.dir, '/test-data.csv'), row.names = FALSE)

  return(wt)

}
