# scale.R
#
#' Scale Function
#'
#' This function centers, scales, and one-hot encodes variables.
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dataset Training and test data
#' @param output Processed output from Monte Carlo radiation transport code simulations
#' @param ext.dir External directory (full path)
#' @return A list of centered, scaled, and one-hot-encoded training and test data
#' @export
#' @import caret
#' @import dplyr

Scale <- function(
  code = 'mcnp',
  dataset = NULL,
  output,
  ext.dir) {

  # set bindings for nonstandard evaluation
  mass <- rad <- sd <- NULL

  output$shape <- output$ht <- output$hd <- NULL

  labels <- names(output)

  Nullify <- function(output, labels) {
    for (i in length(labels):1) {
      groups <- output %>% group_by(get(labels[i])) %>% summarize(no_rows = length(get(labels[i])))
      if (nrow(groups) == 1) {
        output[[i]] <- NULL
      }
    }
    return(output)
  }

  # nullify one-factor variables and one-hot encode categorical variables
  if (is.null(dataset)) {
    null.output <- Nullify(output, labels)
    dummy <- dummyVars(~ ., data = null.output, sep = '')
    training.data <- data.frame(stats::predict(dummy, newdata = null.output))
    training.data <- filter(training.data, sd < 0.001)
  } else if (ncol(output) == ncol(dataset$output)) {
    comb.output <- rbind(output, dataset$output)
    null.output <- Nullify(comb.output, labels)
    dummy <- dummyVars(~ ., data = null.output, sep = '')
    training.data <- data.frame(stats::predict(dummy, newdata = null.output))
    training.data <- training.data[1:nrow(output), ]
    training.data <- filter(training.data, sd < 0.001)
  } else {
    comb.output <- rbind(output, dataset$output[ , 1:(ncol(dataset$output) - 2)])
    null.output <- Nullify(comb.output, labels)
    dummy <- dummyVars(~ ., data = null.output, sep = '')
    training.data <- data.frame(stats::predict(dummy, newdata = null.output))
    training.data <- training.data[1:nrow(output), ]
  }

  # partition data
  if (is.null(dataset)) {
    test.data <- filter(training.data, mass > 100 & rad > 7.62 & rad < 45.72)
    test.data <- slice_sample(test.data, n = round(nrow(training.data) * 0.2))
    training.data <- anti_join(training.data, test.data, by = 'mass')
    # test.data <- slice_sample(training.data, n = round(nrow(training.data) * 0.2))
    # training.data <- anti_join(training.data, test.data, by = 'mass')
  }

#
# scale data
#
  labels <- c('mass', 'rad', 'thk', 'vol', 'conc') # missing 'ht' and 'hd'

  if (is.null(dataset)) {
    training.mean <- apply(training.data[labels], 2, mean)
    training.sd <- apply(training.data[labels], 2, sd)
  } else {
    training.mean <- dataset$training.mean
    training.sd <- dataset$training.sd
  }

  if (is.null(dataset)) {
    training.df <- training.data[ , 1:(ncol(training.data) - 2)]
    test.df <- test.data[ , 1:(ncol(test.data) - 2)]
  } else if (ncol(output) == ncol(dataset$output)) {
    training.df <- training.data[ , 1:(ncol(training.data) - 2)]
  } else {
    training.df <- training.data
  }

  for (i in 1:length(labels)) {
    j <- which(colnames(training.df) == labels[i])
    training.df[[j]] <- scale(training.df[[j]], center = training.mean[i], scale = training.sd[i])
    if (is.null(dataset)) {
      test.df[[j]] <- scale(test.df[[j]], center = training.mean[i], scale = training.sd[i])
    }
  }

  # convert data frames to matrices (Keras requirement)
  training.df <- as.matrix(training.df)

  if (is.null(dataset)) {
    test.df <- as.matrix(test.df)
    dataset <- list(output, training.data, training.mean, training.sd, training.df, test.data, test.df)
    names(dataset) <- c('output', 'training.data', 'training.mean', 'training.sd', 'training.df', 'test.data', 'test.df')
    save(dataset, file = paste0(ext.dir, '/', code, '-dataset.RData'), compress = 'xz')
  } else if (ncol(output) == ncol(dataset$output)) {
    dataset <- list(output, training.data, training.mean, training.sd, training.df)
    names(dataset) <- c('output', 'training.data', 'training.mean', 'training.sd', 'training.df')
    save(dataset, file = paste0(ext.dir, '/', code, '-dataset.RData'), compress = 'xz')
  } else {
    dataset <- training.df
  }

  return(dataset)

}
