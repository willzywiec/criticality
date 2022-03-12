# risk.R
#
#' Risk Function
#'
#' This function imports the Sample function and estimates process criticality accident risk.
#' @param bn Bayesian network
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param cores Number of CPU cores to use for generating Bayesian network samples
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param facility.data .csv file name
#' @param keff.cutoff keff cutoff value (e.g., keff >= 0.9)
#' @param metamodel List of deep neural network metamodels and weights
#' @param risk.pool Number of times risk is calculated
#' @param sample.size Number of samples used to calculate risk
#' @param usl Upper subcritical limit (e.g., keff >= 0.95)
#' @param ext.dir External directory (full path)
#' @return A list of lists containing process criticality accident risk estimates and Bayesian network samples
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
#'   Risk(
#'     bn = BN(
#'       facility.data = "facility.csv",
#'       dist = "gamma",
#'       ext.dir = ext.dir),
#'     code = "mcnp",
#'     cores = 1,
#'     dist = "gamma",
#'     facility.data = "facility.csv",
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
#'       ext.dir = ext.dir),
#'     risk.pool = 10,
#'     sample.size = 1e+04,
#'     usl = 0.95,
#'     ext.dir = ext.dir
#'   )
#' })
#'
#' @import dplyr
#' @import keras
#' @import magrittr
#' @import reticulate

Risk <- function(
  bn,
  code = 'mcnp',
  cores = parallel::detectCores() / 2,
  dist = 'gamma',
  facility.data,
  keff.cutoff = 0.9,
  metamodel,
  risk.pool = 100,
  sample.size = 1e+09,
  usl = 0.95,
  ext.dir) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  facility <- gsub('.csv', '', facility.data)

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
      message('Risk = ', formatC(mean(risk), format = 'e', digits = 3))
      message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))
    } else {
      message('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0))
    }

    if (mean(risk) != 0) message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))

  } else {

    bn.data <- list()
    risk <- pooled.risk <- numeric()

    for (i in 1:risk.pool) {
      bn.data[[i]] <- Sample(bn, code, cores, keff.cutoff, metamodel, sample.size, ext.dir, risk.dir)
      risk[i] <- length(bn.data[[i]]$keff[bn.data[[i]]$keff >= usl]) / sample.size
      if (i == 1) {
        progress.bar <- utils::txtProgressBar(min = 0, max = risk.pool, style = 3)
        utils::setTxtProgressBar(progress.bar, i)
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
      message('Risk = ', formatC(mean(risk), format = 'e', digits = 3))
      message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))
    } else {
      message('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0))
    }

  }

  bn.data <- bind_rows(bn.data)

  return(list(risk, bn.data))

}
