# sample.R
#
# William Zywiec
#
#' Sample Function
#'
#' This function samples a Bayesian network object and uses an existing deep neural network metamodel to predict keff values.
#' @param bn Bayesian network object
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param dataset Training and test data
#' @param keff.cutoff keff cutoff value (e.g., 0.9)
#' @param metamodel List of deep neural network metamodels and weights
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory (PATH)
#' @param risk.dir Risk directory
#' @export
#' @import bnlearn
#' @import keras
#' @import magrittr
#' @import parallel

Sample <- function(
  bn,
  code = 'mcnp',
  dataset,
  keff.cutoff = 0.9,
  metamodel,
  sample.size = 1e+09,
  ext.dir,
  risk.dir) {

  # set bindings for nonstandard evaluation
  op <- ctrl <- mass <- rad <- NULL

#
# sample conditional probability tables
#
  cluster <- parallel::makeCluster((parallel::detectCores() / 2), type = 'SOCK')

  if (keff.cutoff > 0) {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = (as.integer(mass) > 100) & (as.integer(rad) > 7),
      batch = sample.size,
      cluster = cluster,
      n = sample.size) %>% stats::na.omit()
  } else {
    bn.data <- cpdist(
      bn,
      nodes = names(bn),
      evidence = TRUE,
      batch = sample.size,
      cluster = cluster,
      n = sample.size) %>% stats::na.omit()
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

  bn.df <- Scale(
    code = code,
    dataset = dataset,
    output = subset(bn.data, select = -c(op, ctrl)),
    ext.dir = ext.dir)

#
# predict keff values
#
  if (keff.cutoff > 0) {
    bn.data$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df)
    bn.df <- cbind(bn.df, bn.data$keff) %>% subset(bn.data$keff > keff.cutoff)
    bn.df <- bn.df[ , -ncol(bn.df)]
    bn.data <- subset(bn.data, keff > keff.cutoff)
    if (nrow(bn.data) == 0) {
      unlink(risk.dir, recursive = TRUE, force = TRUE)
      stop(paste0('There were no keff values > ', keff.cutoff))
    }
  }

  if (typeof(metamodel[[2]]) == 'list') {
    keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[2]][[1]]))
    for (i in 1:length(metamodel[[2]][[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df)
    bn.data$keff <- rowSums(keff * metamodel[[2]][[1]])
  } else {
    keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
    for (i in 1:length(metamodel[[1]])) keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df)
    bn.data$keff <- rowMeans(keff)
  }

  return(bn.data)

}
