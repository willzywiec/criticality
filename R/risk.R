# risk.R
#
# William Zywiec
#
#' Risk Function
#'
#' This function imports the Sample function and estimates process criticality accident risk.
#' @param bn Bayesian network object
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dataset Training and test data
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param facility Facility name or building number (.csv file name)
#' @param keff.cutoff keff cutoff value (e.g., 0.95)
#' @param metamodel List of deep neural network metamodels and weights
#' @param risk.pool Number of times risk is calculated
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory
#' @export
#' @examples
#' Risk(bn, code, dataset, dist, facility, keff.cutoff, metamodel, risk.pool, sample.size, ext.dir)

Risk <- function(
  bn,
  code = 'mcnp',
  dataset,
  dist = 'gamma',
  facility,
  keff.cutoff = 0.9,
  metamodel,
  risk.pool = 100,
  sample.size = 1e+09,
  ext.dir) {

  library(dplyr)

  if (keff.cutoff > 0) {
    risk.dir <- paste0(ext.dir, '/risk/', facility, '-', dist, '-', formatC(sample.size, format = 'e', digits = 0), '-', keff.cutoff)
    dir.create(risk.dir, recursive = TRUE, showWarnings = FALSE)
  } else {
    risk.dir <- paste0(ext.dir, '/risk/', facility, '-', dist, '-', formatC(sample.size, format = 'e', digits = 0))
    dir.create(risk.dir, recursive = TRUE, showWarnings = FALSE)
  }

  setwd(risk.dir)

  # restrict sample size (~12 GB RAM)
  if (sample.size > 5e+08) {
    risk.pool <- risk.pool * sample.size / 5e+08
    sample.size <- 5e+08
  }

  if (file.exists('risk.csv')) {

    bn.data <- readRDS('bn-data.RData')
    risk <- read.csv('risk.csv', fileEncoding = 'UTF-8-BOM')
    cat('Risk = ', formatC(mean(risk$risk), format = 'e', digits = 3), '\n', sep = '')
    cat('SD = ', formatC(sd(risk$risk), format = 'e', digits = 3), '\n', sep = '')

  } else {

    bn.data <- list()
    risk <- pooled.risk <- numeric()

    for (i in 1:risk.pool) {
      bn.data[[i]] <- Sample(bn, code, dataset, keff.cutoff, metamodel, sample.size, ext.dir, risk.dir)
      risk[i] <- length(bn.data[[i]]$keff[bn.data[[i]]$keff >= 0.95]) / sample.size # USL = 0.95
      if (i == 1) {
        cat('\n', sep = '')
        progress.bar <- txtProgressBar(min = 0, max = risk.pool, style = 3)
        setTxtProgressBar(progress.bar, i)
        if (i == risk.pool) {
          cat('\n', sep = '')
        }
      } else if (i == risk.pool) {
        setTxtProgressBar(progress.bar, i)
        cat('\n', sep = '')
      } else {
        setTxtProgressBar(progress.bar, i)
      }
    }

    if (risk.pool > 100) {
      breaks <- seq((length(risk) / 100), length(risk), (length(risk) / 100))
      for (i in 1:100) pooled.risk[i] <- sum(risk[(breaks[i] - (length(risk) / 100 - 1)):breaks[i]])
      risk <- pooled.risk
    }
  
    saveRDS(bn.data, file = 'bn-data.RData')
    write.csv(as.data.frame(risk, col.names = 'risk'), file = 'risk.csv', row.names = FALSE)
    cat('\nRisk = ', formatC(mean(risk), format = 'e', digits = 3), '\n', sep = '')
    cat('SD = ', formatC(sd(risk), format = 'e', digits = 3), '\n', sep = '')

  }

  bn.data <- bind_rows(bn.data)

  return(list(risk, bn.data))

}
