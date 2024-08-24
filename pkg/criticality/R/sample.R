# sample.R
#
#' Sample Function
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

Sample <- function(
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

  bn.risk <- cpdist(
    bn,
    nodes = names(bn),
    evidence = TRUE,
    cluster = cluster,
    n = sample.size) %>% stats::na.omit()

  parallel::stopCluster(cluster)

  cat('\nBN samples generated')

  bn.risk <- bn.risk %>% dplyr::filter(as.numeric(mass) > mass.cutoff & as.numeric(rad) > rad.cutoff)

  # convert factors to atomic vectors
  bn.risk$mass <- unlist(bn.risk$mass) %>% as.character() %>% as.numeric()
  bn.risk$form <- unlist(bn.risk$form) %>% as.character()
  bn.risk$mod <- unlist(bn.risk$mod) %>% as.character()
  bn.risk$rad <- unlist(bn.risk$rad) %>% as.character() %>% as.numeric()
  bn.risk$ref <- unlist(bn.risk$ref) %>% as.character()
  bn.risk$thk <- unlist(bn.risk$thk) %>% as.character() %>% as.numeric()

  cat('\nBN filtering complete (', nrow(bn.risk), ')', sep = '')

  # set fissile material density (g/cc)
  fiss.density <- bn.risk$form
  fiss.density[fiss.density == 'alpha'] <- 19.86
  fiss.density[fiss.density == 'delta'] <- 15.92
  fiss.density[fiss.density == 'puo2'] <- 11.5
  fiss.density[fiss.density == 'heu'] <- 18.85
  fiss.density[fiss.density == 'uo2'] <- 10.97
  fiss.density <- as.numeric(fiss.density)

  # calculate vol (cc)
  vol <- 4/3 * pi * bn.risk$rad^3

  # fix mod, vol (cc), and rad (cm)
  bn.risk$mod[vol <= bn.risk$mass / fiss.density] <- 'none'
  vol[vol <= bn.risk$mass / fiss.density] <- bn.risk$mass[vol <= bn.risk$mass / fiss.density] / fiss.density[vol <= bn.risk$mass / fiss.density]
  bn.risk$rad <- (3/4 * vol / pi)^(1/3)

  # fix ref and thk (cm)
  bn.risk$ref[bn.risk$thk == 0] <- 'none'
  bn.risk$thk[bn.risk$ref == 'none'] <- 0

  # calculate conc (g/cc)
  bn.risk$conc <- ifelse((vol == 0), 0, (bn.risk$mass / vol))

  # set vol (cc)
  bn.risk$vol <- vol

  bn.df <- Scale(
    output = bn.risk %>% dplyr::select(!c(op, ctrl)),
    ext.dir = ext.dir)

  cat('\nBN processing complete')

#
# predict keff values
#
  if (keff.cutoff > 0 & nrow(bn.risk) > 1) {

    old.len <- nrow(bn.risk) # DELETE

    bn.risk$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)

    bn.df <- cbind(bn.df, bn.risk$keff) %>% subset(bn.risk$keff >= keff.cutoff)
    bn.df <- bn.df[ , -ncol(bn.df)]

    bn.risk <- bn.risk %>% subset(keff >= keff.cutoff)

    new.len <- nrow(bn.risk)

    cat('\nInitial predictions complete (', old.len, ' --> ', new.len, ')', sep = '')
    cat('')

  }

  if (nrow(bn.risk) > 1) {
    if (typeof(metamodel[[2]]) == 'double') {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[2]][[1]]))
      for (i in 1:length(metamodel[[2]][[1]])) {
        keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE) %>% suppressWarnings()
        cat('\nPredictions complete (', i, '/', length(metamodel[[2]][[1]]), ')', sep = '')
        cat('')
      }
      bn.risk$keff <- rowSums(keff * metamodel[[2]][[1]])
    } else {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
      for (i in 1:length(metamodel[[1]])) {
        keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE) %>% suppressWarnings()
        cat('\nPredictions complete (', i, '/', length(metamodel[[1]]), ')', sep = '')
        cat('')
      }
      bn.risk$keff <- rowMeans(keff)
    }
  }

  return(bn.risk)

}
