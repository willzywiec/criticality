# fit.R
#
# William Zywiec

Fit <- function(dataset, model, batch.size, epochs, val.split, remodel.dir, x) {

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
