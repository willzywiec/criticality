# risk.R
#
# William Zywiec
#
#' Risk Function
#'
#' This function imports the Sample function and estimates process criticality accident risk.
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param facility Facility name or building number (.csv file name)
#' @param keff.cutoff keff cutoff value (e.g., 0.95)
#' @param metamodel List of deep neural network metamodels and weights
#' @param risk.pool Number of times risk is calculated
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory
#' @export
#' @examples
#' Risk(
#'   bn = BN(
#'     facility = "facility",
#'     dist = "gamma",
#'     ext.dir = paste0(.libPaths()[1], "/criticality/data")),
#'   code = "mcnp",
#'   dist = "gamma",
#'   facility = "facility",
#'   keff.cutoff = 0.5,
#'   metamodel = NN(
#'     batch.size = 128,
#'     code = "mcnp",
#'     ensemble.size = 1,
#'     epochs = 10,
#'     layers = "8192-256-256-256-256-16",
#'     loss = "sse",
#'     opt.alg = "adamax",
#'     learning.rate = 0.00075,
#'     val.split = 0.2,
#'     replot = TRUE,
#'     verbose = TRUE,
#'     ext.dir = paste0(.libPaths()[1], "/criticality/data"),
#'     training.dir = paste0(.libPaths()[1], "/criticality/data")),
#'   risk.pool = 10,
#'   sample.size = 1e+05,
#'   ext.dir = paste0(.libPaths()[1], "/criticality/data")
#' )
#' @import dplyr

Risk <- function(
  bn,
  code = 'mcnp',
  dist = 'gamma',
  facility,
  keff.cutoff = 0.9,
  metamodel,
  risk.pool = 100,
  sample.size = 1e+09,
  ext.dir) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

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

  Sample <- function(bn, code, keff.cutoff, metamodel, sample.size, ext.dir, risk.dir) {
  
    cluster <- parallel::makeCluster((parallel::detectCores() / 2), type = 'SOCK')
  
    # sample conditional probability tables
    if (keff.cutoff > 0) {
      bn.data <- cpdist(
        bn,
        nodes = names(bn),
        evidence = (as.integer(mass) > 100) & (as.integer(rad) > 7),
        batch = sample.size,
        cluster = cluster,
        n = sample.size) %>% na.omit()
    } else {
      bn.data <- cpdist(
        bn,
        nodes = names(bn),
        evidence = TRUE,
        batch = sample.size,
        cluster = cluster,
        n = sample.size) %>% na.omit()
    }
  
    parallel::stopCluster(cluster)
  
    bn.data[[3]] <- unlist(bn.data[[3]]) %>% as.character() %>% as.numeric() # mass
    bn.data[[4]] <- unlist(bn.data[[4]])                                     # form
    bn.data[[5]] <- unlist(bn.data[[5]])                                     # mod
    bn.data[[6]] <- unlist(bn.data[[6]]) %>% as.character() %>% as.numeric() # rad
    bn.data[[7]] <- unlist(bn.data[[7]])                                     # ref
    bn.data[[8]] <- unlist(bn.data[[8]]) %>% as.character() %>% as.numeric() # thk
  
    # set Pu density (g/cc)
    pu.density <- ifelse((bn.data$form == 'alpha'), 19.86, 11.5)
  
    # calculate vol (cc)
    vol <- 4/3 * pi * bn.data$rad^3
  
    # fix mod, vol (cc), and rad (cm)
    bn.data$mod[vol <= bn.data$mass / pu.density] <- 'none'
    vol[vol <= bn.data$mass / pu.density] <- bn.data$mass[vol <= bn.data$mass / pu.density] / pu.density[vol <= bn.data$mass / pu.density]
    bn.data$rad <- (3/4 * vol / pi)^(1/3)
  
    # fix ref and thk (cm)
    bn.data$ref[bn.data$thk == 0] <- 'none'
    bn.data$thk[bn.data$ref == 'none'] <- 0
  
    # calculate conc (g/cc)
    conc <- ifelse((vol == 0), 0, (bn.data$mass / vol))
  
    # set form, vol (cc), and conc (g/cc)
    bn.data$form <- ifelse((pu.density == 19.86), 'alpha', 'puo2')
    bn.data$vol <- vol
    bn.data$conc <- conc
  
    bn.df <- Scale(code, subset(bn.data, select = -c(op, ctrl)))
  
    # predict keff values
    if (keff.cutoff > 0) {
      bn.data$keff <- metamodel[[1]][[1]] %>% predict(bn.df)
      bn.df <- cbind(bn.df, bn.data$keff) %>% subset(bn.data$keff > keff.cutoff)
      bn.df <- bn.df[ , -ncol(bn.df)]
      bn.data <- subset(bn.data, keff > keff.cutoff)
      if (nrow(bn.data) == 0) {
        setwd(ext.dir)
        unlink(risk.dir, recursive = TRUE, force = TRUE)
        stop(paste0('There were no keff values > ', keff.cutoff))
      }
    }
  
    if (typeof(metamodel[[2]]) == 'list') {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[2]][[1]]))
      for (i in 1:length(metamodel[[2]][[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% predict(bn.df)
      bn.data$keff <- rowSums(keff * metamodel[[2]][[1]])
    } else {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
      for (i in 1:length(metamodel[[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% predict(bn.df)
      bn.data$keff <- rowMeans(keff)
    }
  
    return(bn.data)
  
  }

  # estimate process criticality accident risk
  if (file.exists('risk.csv') && length(read.csv('risk.csv', fileEncoding = 'UTF-8-BOM')[ , 1]) >= risk.pool) {

    bn.data <- readRDS('bn-data.RData')
    risk <- read.csv('risk.csv', fileEncoding = 'UTF-8-BOM')[ , 1]

    if (mean(risk) != 0) {
      cat('Risk = ', formatC(mean(risk), format = 'e', digits = 3), '\n', sep = '')
      cat('SD = ', formatC(sd(risk), format = 'e', digits = 3), '\n', sep = '')
    } else {
      cat('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0), '\n', sep = '')
    }

    if (mean(risk) != 0) cat('SD = ', formatC(sd(risk), format = 'e', digits = 3), '\n', sep = '')

  } else {

    bn.data <- list()
    risk <- pooled.risk <- numeric()

    for (i in 1:risk.pool) {
      bn.data[[i]] <- Sample(bn, code, keff.cutoff, metamodel, sample.size, ext.dir, risk.dir)
      risk[i] <- length(bn.data[[i]]$keff[bn.data[[i]]$keff >= 0.95]) / sample.size # USL = 0.95
      if (i == 1) {
        cat('\n', sep = '')
        progress.bar <- txtProgressBar(min = 0, max = risk.pool, style = 3)
        setTxtProgressBar(progress.bar, i)
        if (i == risk.pool) {
          cat('\n\n', sep = '')
        }
      } else if (i == risk.pool) {
        setTxtProgressBar(progress.bar, i)
        cat('\n\n', sep = '')
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

    if (mean(risk) != 0) {
      cat('Risk = ', formatC(mean(risk), format = 'e', digits = 3), '\n', sep = '')
      cat('SD = ', formatC(sd(risk), format = 'e', digits = 3), '\n', sep = '')
    } else {
      cat('Risk < ', formatC(risk.pool * sample.size, format = 'e', digits = 0), '\n', sep = '')
    }

  }

  bn.data <- bind_rows(bn.data)

  return(list(risk, bn.data))

}
