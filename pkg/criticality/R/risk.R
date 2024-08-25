# risk.R
#
#' Risk Function
#'
#' This function estimates process criticality accident risk (imports Sample function).
#' @param bn Bayesian network
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param cores Number of CPU cores to use for generating Bayesian network samples
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param metamodel List of deep neural network metamodels and weights
#' @param keff.cutoff keff cutoff value (e.g., 0.9)
#' @param mass.cutoff mass cutoff (grams)
#' @param rad.cutoff radius cutoff (cm)
#' @param risk.pool Number of times risk is calculated
#' @param sample.size Number of samples used to calculate risk
#' @param usl Upper subcritical limit (e.g., 0.95)
#' @param ext.dir External directory (full path)
#' @param training.dir Training directory (full path)
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
#'       ext.dir = ext.dir),
#'     code = "mcnp",
#'     cores = 1,
#'     metamodel = NN(
#'       batch.size = 128,
#'       ensemble.size = 1,
#'       epochs = 10,
#'       layers = "256-256-16",
#'       replot = FALSE,
#'       ext.dir = ext.dir),
#'     keff.cutoff = 0.5,
#'     mass.cutoff = 100,
#'     rad.cutoff = 7,
#'     risk.pool = 10,
#'     sample.size = 1e+04,
#'     usl = 0.95,
#'     ext.dir = ext.dir,
#'     training.dir = NULL
#'   )
#' })
#'
#' @import dplyr
#' @import magrittr
#' @import reticulate

Risk <- function(
  bn,
  code = 'mcnp',
  cores = parallel::detectCores() / 2,
  dist = 'gamma',
  metamodel,
  keff.cutoff = 0.9,
  mass.cutoff = 100,
  rad.cutoff = 7,
  risk.pool = 100,
  sample.size = 1e+06,
  usl = 0.95,
  ext.dir,
  training.dir = NULL) {

  # if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  if (is.null(training.dir)) training.dir <- paste0(ext.dir, '/training')

  time.stamp <- Sys.time() %>% as.character() %>% strsplit(split = ' ') %>% .[[1]]

  date <- time.stamp[1]
  time <- time.stamp[2] %>% strsplit(split = '[.]') %>% .[[1]] %>% .[1] %>% gsub(':', '.', .)

  facility <- substitute(bn) %>% deparse()

  risk.dir <- paste0(ext.dir, '/risk/', facility, '-', date, '-', time)

  dir.create(risk.dir, recursive = TRUE, showWarnings = FALSE)

  risk.settings <- data.frame(V1 = c(
    'risk settings',
    paste0('distribution: ', dist),
    paste0('facility: ', facility),
    paste0('keff cutoff: ', keff.cutoff),
    paste0('mass cutoff (g): ', mass.cutoff),
    paste0('rad cutoff (cm): ', rad.cutoff),
    paste0('risk pool: ', risk.pool),
    paste0('sample size: ', sample.size),
    paste0('upper subcritical limit: ', usl),
    paste0('external directory: ', ext.dir),
    paste0('training directory: ', training.dir),
    paste0('risk directory: ', risk.dir)))

  utils::write.table(risk.settings, file = paste0(risk.dir, '/risk-settings.txt'), quote = FALSE, row.names = FALSE, col.names = FALSE)

  # restrict sample size (~12 GB RAM)
  if (sample.size > 5e+08) {
    risk.pool <- risk.pool * sample.size / 5e+08
    sample.size <- 5e+08
  }

#
# estimate process criticality accident risk
#
  if (file.exists(paste0(risk.dir, '/risk.csv')) && length(utils::read.csv(paste0(risk.dir, '/risk.csv'), fileEncoding = 'UTF-8-BOM')[ , 1]) >= risk.pool) {

    bn.dist <- readRDS(paste0(risk.dir, '/bn-risk.RData'))
    risk <- utils::read.csv(paste0(risk.dir, '/risk.csv'), fileEncoding = 'UTF-8-BOM')[ , 1]

    if (mean(risk) != 0) {
      message('Risk = ', formatC(mean(risk), format = 'e', digits = 3))
      message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))
    } else {
      message('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0))
    }

    if (mean(risk) != 0) {
      if (risk.pool == 1) {
        message('SD = NA')
      } else {
        message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))
      }
    }

  } else {

    bn.dist <- list()
    
    risk <- pooled.risk <- numeric()

    progress.bar <- utils::txtProgressBar(max = risk.pool, style = 3)

    for (i in 1:risk.pool) {
      bn.dist[[i]] <- Predict(
        bn = bn,
        cores = cores,
        metamodel = metamodel,
        keff.cutoff = keff.cutoff,
        mass.cutoff = mass.cutoff,
        rad.cutoff = rad.cutoff,
        sample.size = sample.size,
        ext.dir = ext.dir)
      risk[i] <- length(bn.dist[[i]]$keff[bn.dist[[i]]$keff >= usl]) / sample.size
      utils::setTxtProgressBar(progress.bar, i)
    }

    close(progress.bar)
  
    utils::write.csv(as.data.frame(risk, col.names = 'risk'), file = paste0(risk.dir, '/risk.csv'), row.names = FALSE)
    saveRDS(bn.dist, file = paste0(risk.dir, '/bn-risk.RData'))

    if (mean(risk) != 0) {
      message('Risk = ', formatC(mean(risk), format = 'e', digits = 3))
      if (risk.pool == 1) {
        message('SD = NA')
      } else {
        message('SD = ', formatC(stats::sd(risk), format = 'e', digits = 3))
      }
    } else {
      message('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0))
    }

  }

  bn.dist <- dplyr::bind_rows(bn.dist)

  return(list(risk, bn.dist))

}
