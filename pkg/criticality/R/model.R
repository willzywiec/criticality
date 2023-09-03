# model.R
#
#' Model Function
#'
#' This function builds the deep neural network metamodel architecture.
#' @param dataset Training and test data
#' @param layers String that defines the deep neural network architecture (e.g., "64-64")
#' @param loss Loss function
#' @param opt.alg Optimization algorithm
#' @param learning.rate Learning rate
#' @param ext.dir External directory (full path)
#' @return A deep neural network metamodel of Monte Carlo radiation transport code simulation data
#' @export
#' @import keras
#' @import magrittr

Model <- function(
  dataset,
  layers = '8192-256-256-256-256-16',
  loss = 'sse',
  opt.alg = 'adamax',
  learning.rate = 0.00075,
  ext.dir) {

  layers <- strsplit(layers, '-') %>% unlist() %>% as.integer()

  model <- keras_model_sequential() %>% layer_dense(units = layers[1], activation = 'relu', input_shape = dim(dataset$training.df)[2])

  if (length(layers) >= 2) {
    model <- model %>% layer_dense(units = layers[2], activation = 'relu')
  } else if (length(layers) >= 3) {
    model <- model %>% layer_dense(units = layers[3], activation = 'relu')
  } else if (length(layers) >= 4) {
    model <- model %>% layer_dense(units = layers[4], activation = 'relu')
  } else if (length(layers) >= 5) {
    model <- model %>% layer_dense(units = layers[5], activation = 'relu')
  } else if (length(layers) >= 6) {
    model <- model %>% layer_dense(units = layers[6], activation = 'relu')
  } else if (length(layers) >= 7) {
    model <- model %>% layer_dense(units = layers[7], activation = 'relu')
  } else if (length(layers) == 8) {
    model <- model %>% layer_dense(units = layers[8], activation = 'relu')
  } else if (length(layers) == 9) {
    model <- model %>% layer_dense(units = layers[9], activation = 'relu')
  }

  model <- model %>% layer_dense(units = 1, activation = 'linear')

#
# refactor functions to set up legacy optimizers for Apple M1/M2 chips
# https://github.com/rstudio/keras/blob/main/R/utils.R
#
  assert_all_dots_named <- function(envir = parent.frame(), cl) {

    x <- eval(quote(list(...)), envir)

    if(!length(x)) return()

    x <- names(x)

    if(is.character(x) && !anyNA(x) && all(x != "")) return()

    stop('All arguments provided to `...` must be named.\n',
      'Call with unnamed arguments in dots:\n  ',
      paste(deparse(cl, 500L), collapse = '\n'))

  }

  capture_args <- function(
    cl, modifiers = NULL, ignore = NULL,
    envir = parent.frame(), fn = sys.function(-1)) {

    fn_arg_nms <- names(formals(fn))
    known_args <- intersect(names(cl), fn_arg_nms)
    known_args <- setdiff(known_args, ignore)
    names(known_args) <- known_args
    cl2 <- c(quote(list), lapply(known_args, as.symbol))

    if("..." %in% fn_arg_nms && !"..." %in% ignore) {
      assert_all_dots_named(envir, cl)
      cl2 <- c(cl2, quote(...))
    }

    args <- eval(as.call(cl2), envir)

    # check `ignore` again, since arg might have been in `...`
    for (nm in intersect(names(args), ignore)) {
      args[[nm]] <- NULL
    }

    nms_to_modify <- intersect(names(args), names(modifiers))

    for (nm in nms_to_modify) {
      args[nm] <- list(modifiers[[nm]](args[[nm]]))
    }

    return(args)

  }

  legacy_optimizer_adadelta <- function(
    learning_rate = 0.001, rho = 0.95, epsilon = 1e-07,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    jit_compile = TRUE, name = "Adadelta", ...) {
    args <- capture_args(match.call(), NULL)
    do.call(keras$optimizers$legacy$Adadelta, args)
  }

  legacy_optimizer_adagrad <- function(
    learning_rate = 0.001, initial_accumulator_value = 0.1,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, jit_compile = TRUE,
    name = 'Adagrad', ...) {
    args <- capture_args(match.call(), NULL)
    do.call(keras$optimizers$legacy$Adagrad, args)
  }

  legacy_optimizer_adam <- function(
    learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, amsgrad = FALSE, weight_decay = NULL, clipnorm = NULL,
    clipvalue = NULL, global_clipnorm = NULL, use_ema = FALSE,
    ema_momentum = 0.99, ema_overwrite_frequency = NULL, jit_compile = TRUE,
    name = 'Adam', ...) {
    args <- capture_args(match.call(), NULL)
    do.call(keras$optimizers$legacy$Adam, args)
  }

  legacy_optimizer_adamax <- function(
    learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, jit_compile = TRUE,
    name = 'Adamax', ...) {
    args <- capture_args(match.call(), NULL)
    do.call(keras$optimizers$legacy$Adamax, args)
  }

  legacy_optimizer_nadam <- function(
    learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, jit_compile = TRUE,
    name = 'Nadam', ...) {
    args <- capture_args(match.call(), NULL)
    do.call(keras$optimizers$legacy$Nadam, args)
  }

  legacy_optimizer_rmsprop <- function(
    learning_rate = 0.001, rho = 0.9, momentum = 0, epsilon = 1e-07,
    centered = FALSE, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = 100L, jit_compile = TRUE,
    name = "RMSprop", ...) {
    args <- capture_args(match.call(), list(ema_overwrite_frequency = as.integer))
    do.call(keras$optimizers$legacy$RMSprop, args)
  }

  if (opt.alg == 'adadelta') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adadelta(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adagrad') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adagrad(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'adamax') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_adamax(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'nadam') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_nadam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'rmsprop') {
    model %>% compile(
      loss = loss,
      optimizer = optimizer_rmsprop(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_adadelta') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_adadelta(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_adagrad') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_adagrad(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_adam') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_adam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_adamax') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_adamax(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_nadam') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_nadam(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  } else if (opt.alg == 'legacy_rmsprop') {
    model %>% compile(
      loss = loss,
      optimizer = legacy_optimizer_rmsprop(learning_rate = learning.rate),
      metrics = c('mean_absolute_error'))
  }

}
