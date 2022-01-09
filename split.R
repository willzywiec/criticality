# split.R
#
# William Zywiec

Split <- function(dataset, batch.size, ensemble.size, epochs, layers, loss, opt.alg, learning.rate, val.split, replot, source.dir, test.dir) {

  # library(magrittr)

  # load function
  source(paste0(source.dir, '/nn.R'))

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

  for (i in 1:length(dataset)) {
    metamodel[[i]] <- NN(dataset[[i]], batch.size, ensemble.size, epochs, layers, loss ,opt.alg, learning.rate, val.split, replot, source.dir, paste0(test.dir, '/', form[i]))
  }

  names(metamodel) <- form

  return(metamodel)

}
