# test.R
#
# William Zywiec
#
#' Test Function
#'
#' This function optimizes an existing deep neural network metamodel and generates .csv predictions for all training and test data.
#' @param dataset Training and test data
#' @param metamodel List of deep neural network metamodels
#' @param training.mae Training data mean absolute errors
#' @param val.mae Cross-validation data mean absolute errors
#' @param training.dir Training directory
#' @export
#' @examples
#' Test(
#'   dataset,
#'   metamodel,
#'   training.mae,
#'   val.mae,
#'   training.dir = paste0(.libPaths(), "/criticality/data")
#' )
#' @import keras
#' @import magrittr

Test <- function(
  dataset,
  metamodel,
  training.mae,
  val.mae,
  training.dir) {

  setwd(training.dir)

  meta.len <- length(metamodel)

  test.data <- dataset$test.data

  test.pred <- matrix(nrow = nrow(dataset$test.df), ncol = meta.len)

  test.mae <- avg <- nm <- bfgs <- sa <- numeric()

  nm.wt <- bfgs.wt <- sa.wt <- list()

  Objective <- function(x) mean(abs(test.data$keff - rowSums(test.pred * x, na.rm = TRUE)))

  # minimize objective function
  for (i in 1:meta.len) {

    test.pred[ , i] <- metamodel[[i]] %>% predict_on_batch(dataset$test.df)

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

      training.pred[ , 1] <- metamodel[[1]] %>% predict_on_batch(dataset$training.df)

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
        training.pred[ , i] <- metamodel[[i]] %>% predict_on_batch(dataset$training.df)
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
