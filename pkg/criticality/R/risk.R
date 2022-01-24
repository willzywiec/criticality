# risk.R
#
#' Risk Function
#'
#' This function imports the Sample function and estimates process criticality accident risk.
#' @param bn Bayesian network
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param facility Facility name or building number (.csv file name)
#' @param keff.cutoff keff cutoff value (e.g., 0.95)
#' @param metamodel List of deep neural network metamodels and weights
#' @param risk.pool Number of times risk is calculated
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory (full path)
#' @export
#' @examples
#' config <- FALSE
#' try(config <- reticulate::py_config()$available)
#' try(if (config == TRUE) {
#'   Risk(
#'     bn = BN(
#'       facility = "facility",
#'       dist = "gamma",
#'       ext.dir = paste0(.libPaths()[1], "/criticality/extdata")),
#'     code = "mcnp",
#'     dist = "gamma",
#'     facility = "facility",
#'     keff.cutoff = 0.5,
#'     metamodel = NN(
#'       batch.size = 128,
#'       code = "mcnp",
#'       ensemble.size = 1,
#'       epochs = 10,
#'       layers = "8192-256-256-256-256-16",
#'       loss = "sse",
#'       opt.alg = "adamax",
#'       learning.rate = 0.00075,
#'       val.split = 0.2,
#'       replot = TRUE,
#'       verbose = TRUE,
#'       ext.dir = paste0(.libPaths()[1], "/criticality/extdata")),
#'     risk.pool = 10,
#'     sample.size = 1e+04,
#'     ext.dir = paste0(.libPaths()[1], "/criticality/extdata")
#'   )
#' })
#' @import dplyr
#' @import keras
#' @import magrittr
#' @import reticulate

Risk <- function(
  bn,
  code = 'mcnp',
  dist = 'gamma',
  facility,
  keff.cutoff = 0.9,
  metamodel,
  risk.pool = 100,
  sample.size = 1e+09,
  ext.dir = getwd()) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  if (keff.cutoff > 0) {
    risk.dir <- paste0(ext.dir, '/risk/', facility, '-', dist, '-', formatC(sample.size, format = 'e', digits = 0), '-', keff.cutoff)
    dir.create(risk.dir, recursive = TRUE, showWarnings = FALSE)
  } else {
    risk.dir <- paste0(ext.dir, '/risk/', facility, '-', dist, '-', formatC(sample.size, format = 'e', digits = 0))
    dir.create(risk.dir, recursive = TRUE, showWarnings = FALSE)
  }

  # restrict sample size (~12 GB RAM)
  if (sample.size > 5e+08) {
    risk.pool <- risk.pool * sample.size / 5e+08
    sample.size <- 5e+08
  }

#
# estimate process criticality accident risk
#
  if (file.exists(paste0(risk.dir, 'risk.csv')) && length(utils::read.csv(paste0(risk.dir, 'risk.csv'), fileEncoding = 'UTF-8-BOM')[ , 1]) >= risk.pool) {

    bn.data <- readRDS(paste0(risk.dir, '/bn-data.RData'))
    risk <- utils::read.csv(paste0(risk.dir, '/risk.csv'), fileEncoding = 'UTF-8-BOM')[ , 1]

    if (mean(risk) != 0) {
      cat('Risk = ', formatC(mean(risk), format = 'e', digits = 3), '\n', sep = '')
      cat('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3), '\n', sep = '')
    } else {
      cat('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0), '\n', sep = '')
    }

    if (mean(risk) != 0) cat('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3), '\n', sep = '')

  } else {

    bn.data <- list()
    risk <- pooled.risk <- numeric()

    for (i in 1:risk.pool) {
      bn.data[[i]] <- Sample(bn, code, dataset, keff.cutoff, metamodel, sample.size, ext.dir, risk.dir)
      risk[i] <- length(bn.data[[i]]$keff[bn.data[[i]]$keff >= 0.95]) / sample.size # USL = 0.95
      if (i == 1) {
        cat('\n', sep = '')
        progress.bar <- utils::txtProgressBar(min = 0, max = risk.pool, style = 3)
        utils::setTxtProgressBar(progress.bar, i)
        if (i == risk.pool) {
          cat('\n', sep = '')
        }
      } else if (i == risk.pool) {
        utils::setTxtProgressBar(progress.bar, i)
        cat('\n', sep = '')
      } else {
        utils::setTxtProgressBar(progress.bar, i)
      }
    }

    close(progress.bar)

    if (risk.pool > 100) {
      breaks <- seq((length(risk) / 100), length(risk), (length(risk) / 100))
      for (i in 1:100) pooled.risk[i] <- sum(risk[(breaks[i] - (length(risk) / 100 - 1)):breaks[i]])
      risk <- pooled.risk
    }
  
    saveRDS(bn.data, file = paste0(ext.dir, '/bn-data.RData'))
    utils::write.csv(as.data.frame(risk, col.names = 'risk'), file = paste0(risk.dir, '/risk.csv'), row.names = FALSE)

    if (mean(risk) != 0) {
      cat('Risk = ', formatC(mean(risk), format = 'e', digits = 3), '\n', sep = '')
      cat('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3), '\n', sep = '')
    } else {
      cat('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0), '\n', sep = '')
    }

  }

  bn.data <- bind_rows(bn.data)

  return(list(risk, bn.data))

}