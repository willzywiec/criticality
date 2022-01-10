# fit.R
#
# William Zywiec
#
#' Fit Function
#'
#' This function trains an existing deep neural network metamodel.
#' @param dataset Training and test data
#' @param model Keras model
#' @param batch.size Batch size
#' @param epochs Number of training epochs
#' @param val.split Validation split
#' @param remodel.dir Directory that contains model files that are saved after every epoch
#' @param x Model number
#' @export
#' @examples
#' Fit(dataset, model, batch.size, epochs, val.split, remodel.dir, x)

Fit <- function(
  dataset,
  model,
  batch.size = 8192,
  epochs = 1500,
  val.split = 0.2,
  remodel.dir,
  x) {

  # library(keras)
  # library(magrittr)

  if (missing(x)) {
    model %>% fit(
      dataset$training.df,
      dataset$training.data$keff,
      batch_size = batch.size,
      epochs = epochs,
      validation_split = val.split,
      verbose = TRUE)
  } else {
    checkpoint <- callback_model_checkpoint(paste0(remodel.dir, '/', i, '-{epoch:1d}.h5'), monitor = 'mean_absolute_error')
    model %>% fit(
      dataset$training.df,
      dataset$training.data$keff,
      batch_size = batch.size,
      epochs = epochs,
      validation_split = val.split,
      verbose = TRUE,
      callbacks = c(checkpoint))
  }

}
