# sample.R
#
#' Sample Function
#'
#' This function samples the Bayesian network and generates keff predictions using a deep neural network metamodel.
#' @param bn Bayesian network object
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param cores Number of CPU cores to use for generating Bayesian network samples
#' @param keff.cutoff keff cutoff value (e.g., 0.9)
#' @param metamodel List of deep neural network metamodels and weights
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory (full path)
#' @param risk.dir Risk directory
#' @return A list of Bayesian network samples with predicted keff values
#' @export
#' @import bnlearn
#' @import keras
#' @import magrittr
#' @import parallel

Sample <- function(
  bn,
  code = 'mcnp',
  cores = parallel::detectCores() / 2,
  keff.cutoff = 0.9,
  metamodel,
  sample.size = 1e+09,
  ext.dir,
  risk.dir = NULL) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  # set bindings for nonstandard evaluation
  op <- ctrl <- mass <- rad <- NULL

#
# sample conditional probability tables
#
  if (cores > 1) {
    if (cores > parallel::detectCores()) cores <- parallel::detectCores()
    cluster <- parallel::makeCluster(cores)
  }

  if (cores > 1 && keff.cutoff > 0) {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = (as.integer(mass) > 100) & (as.integer(rad) > 7),
      batch = sample.size,
      cluster = cluster,
      n = sample.size) %>% stats::na.omit()
  } else if (cores == 1 && keff.cutoff > 0) {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = (as.integer(mass) > 100) & (as.integer(rad) > 7),
      batch = sample.size,
      n = sample.size) %>% stats::na.omit()
  } else if (cores > 1) {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = TRUE,
      batch = sample.size,
      cluster = cluster,
      n = sample.size) %>% stats::na.omit()
  } else {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = TRUE,
      batch = sample.size,
      n = sample.size) %>% stats::na.omit()
  }

  if (cores > 1) parallel::stopCluster(cluster)

  bn.data[[3]] <- unlist(bn.data[[3]]) %>% as.character() %>% as.numeric() # mass
  bn.data[[4]] <- unlist(bn.data[[4]])                                     # form
  bn.data[[5]] <- unlist(bn.data[[5]])                                     # mod
  bn.data[[6]] <- unlist(bn.data[[6]]) %>% as.character() %>% as.numeric() # rad
  bn.data[[7]] <- unlist(bn.data[[7]])                                     # ref
  bn.data[[8]] <- unlist(bn.data[[8]]) %>% as.character() %>% as.numeric() # thk

  # set fissile material density (g/cc)
  fiss.density <- as.character(bn.data$form)
  fiss.density[fiss.density == 'alpha'] <- 19.86
  fiss.density[fiss.density == 'delta'] <- 15.92
  fiss.density[fiss.density == 'puo2'] <- 11.5
  fiss.density[fiss.density == 'heu'] <- 18.85
  fiss.density[fiss.density == 'uo2'] <- 10.97
  fiss.density <- as.numeric(fiss.density)

  # calculate vol (cc)
  vol <- 4/3 * pi * bn.data$rad^3

  # fix mod, vol (cc), and rad (cm)
  bn.data$mod[vol <= bn.data$mass / fiss.density] <- 'none'
  vol[vol <= bn.data$mass / fiss.density] <- bn.data$mass[vol <= bn.data$mass / fiss.density] / fiss.density[vol <= bn.data$mass / fiss.density]
  bn.data$rad <- (3/4 * vol / pi)^(1/3)

  # fix ref and thk (cm)
  bn.data$ref[bn.data$thk == 0] <- 'none'
  bn.data$thk[bn.data$ref == 'none'] <- 0

  # calculate conc (g/cc)
  bn.data$conc <- ifelse((vol == 0), 0, (bn.data$mass / vol))

  # set vol (cc)
  bn.data$vol <- vol

  bn.df <- Scale(
    code = code,
    dataset = dataset,
    output = subset(bn.data, select = -c(op, ctrl)),
    ext.dir = ext.dir)

#
# predict keff values
#
  if (keff.cutoff > 0) {

    bn.data$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)
    bn.data <- subset(bn.data, keff > keff.cutoff)

    while (nrow(subset(bn.data, keff > keff.cutoff)) == 0) {
      if (as.character(keff.cutoff) %>% strsplit('[.]') %>% unlist() %>% .[2] %>% nchar() > 1) {
        keff.cutoff <- trunc(keff.cutoff * 100) / 100 - 0.01
        bn.data$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)
      } else {
        keff.cutoff <- trunc(keff.cutoff * 100) / 100 - 0.1
        bn.data$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)
      }
    }

    bn.df <- cbind(bn.df, bn.data$keff) %>% subset(bn.data$keff > keff.cutoff) %>% .[ , -ncol(bn.df)]

  }

  if (typeof(metamodel[[2]]) == 'list') {
    keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[2]][[1]]))
    for (i in 1:length(metamodel[[2]][[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE)
    bn.data$keff <- rowSums(keff * metamodel[[2]][[1]])
  } else {
    keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
    for (i in 1:length(metamodel[[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE)
    bn.data$keff <- rowMeans(keff)
  }

  return(bn.data)

}
