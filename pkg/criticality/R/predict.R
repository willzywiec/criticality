# predict.R
#
#' Predict Function
#'
#' This function samples the Bayesian network and generates keff predictions using a deep neural network metamodel.
#' @param bn Bayesian network object
#' @param cores Number of CPU cores to use for generating Bayesian network samples
#' @param metamodel List of deep neural network metamodels and weights
#' @param keff.cutoff keff cutoff value (e.g., 0.9)
#' @param mass.cutoff mass cutoff (grams)
#' @param rad.cutoff radius cutoff (cm)
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory (full path)
#' @return A list of Bayesian network samples with predicted keff values
#' @export
#' @import bnlearn
#' @import dplyr
#' @import magrittr
#' @import parallel

Predict <- function(
  bn,
  cores = parallel::detectCores() / 2,
  metamodel,
  keff.cutoff = 0.9,
  mass.cutoff = 100,
  rad.cutoff = 7,
  sample.size = 1e+06,
  ext.dir) {

  # set bindings for nonstandard evaluation
  op <- ctrl <- mass <- rad <- NULL

#
# sample conditional probability tables
#
  if (cores > parallel::detectCores()) cores <- parallel::detectCores()

  cluster <- parallel::makeCluster(cores)

  bn.dist <- bnlearn::cpdist(
    bn,
    nodes = names(bn),
    evidence = TRUE,
    cluster = cluster,
    n = sample.size) %>% stats::na.omit()

  parallel::stopCluster(cluster)

  cat('\nBN samples generated')

  bn.dist <- bn.dist %>% dplyr::filter(as.numeric(mass) > mass.cutoff & as.numeric(rad) > rad.cutoff)

  # convert factors to atomic vectors
  bn.dist$mass <- unlist(bn.dist$mass) %>% as.character() %>% as.numeric()
  bn.dist$form <- unlist(bn.dist$form) %>% as.character()
  bn.dist$mod <- unlist(bn.dist$mod) %>% as.character()
  bn.dist$rad <- unlist(bn.dist$rad) %>% as.character() %>% as.numeric()
  bn.dist$ref <- unlist(bn.dist$ref) %>% as.character()
  bn.dist$thk <- unlist(bn.dist$thk) %>% as.character() %>% as.numeric()

  cat('\nBN filtering complete (', nrow(bn.dist), ')', sep = '')

  # set fissile material density (g/cc)
  fiss.density <- bn.dist$form
  fiss.density[fiss.density == 'alpha'] <- 19.86
  fiss.density[fiss.density == 'delta'] <- 15.92
  fiss.density[fiss.density == 'puo2'] <- 11.5
  fiss.density[fiss.density == 'heu'] <- 18.85
  fiss.density[fiss.density == 'uo2'] <- 10.97
  fiss.density <- as.numeric(fiss.density)

  # calculate vol (cc)
  vol <- 4/3 * pi * bn.dist$rad^3

  # fix mod, vol (cc), and rad (cm)
  bn.dist$mod[vol <= bn.dist$mass / fiss.density] <- 'none'
  vol[vol <= bn.dist$mass / fiss.density] <- bn.dist$mass[vol <= bn.dist$mass / fiss.density] / fiss.density[vol <= bn.dist$mass / fiss.density]
  bn.dist$rad <- (3/4 * vol / pi)^(1/3)

  # fix ref and thk (cm)
  bn.dist$ref[bn.dist$thk == 0] <- 'none'
  bn.dist$thk[bn.dist$ref == 'none'] <- 0

  # calculate conc (g/cc)
  bn.dist$conc <- ifelse((vol == 0), 0, (bn.dist$mass / vol))

  # set vol (cc)
  bn.dist$vol <- vol

  bn.df <- Scale(
    output = bn.dist %>% dplyr::select(!c(op, ctrl)),
    ext.dir = ext.dir)

  cat('\nBN processing complete')

#
# predict keff values
#
  if (keff.cutoff > 0 & nrow(bn.dist) > 1) {
    old.nrow <- nrow(bn.dist)
    bn.dist$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)
    bn.df <- cbind(bn.df, bn.dist$keff) %>% subset(bn.dist$keff >= keff.cutoff)
    bn.df <- bn.df[ , -ncol(bn.df)]
    bn.dist <- bn.dist %>% subset(keff >= keff.cutoff)
    new.nrow <- nrow(bn.dist)
    cat('\nInitial predictions complete (', old.nrow, ' --> ', new.nrow, ')', sep = '')
  }

  if (nrow(bn.dist) > 1) {
    keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
    for (i in 1:length(metamodel[[1]])) {
      keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE) %>% suppressWarnings()
      keff[ , i] <- keff[ , i] * metamodel[[2]][[i]]
      if (i < length(metamodel[[1]])) {
        cat('\nPredictions complete (', i, '/', length(metamodel[[1]]), ')', sep = '')
      } else {
        cat('\nPredictions complete (', i, '/', length(metamodel[[1]]), ')\n', sep = '')
      }
    }
    bn.dist$keff <- rowSums(keff)
  }

  return(bn.dist)

}
