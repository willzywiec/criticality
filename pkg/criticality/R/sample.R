# sample.R
#
#' Sample Function
#'
#' This function samples the Bayesian network and generates keff predictions using a deep neural network metamodel.
#' @param bn Bayesian network object
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param cores Number of CPU cores to use for generating Bayesian network samples
#' @param metamodel List of deep neural network metamodels and weights
#' @param keff.cutoff keff cutoff value (e.g., 0.9)
#' @param mass.cutoff mass cutoff in grams (e.g., 100)
#' @param rad.cutoff radius cutoff in cm (e.g., 7)
#' @param sample.size Number of samples used to calculate risk
#' @param ext.dir External directory (full path)
#' @return A list of Bayesian network samples with predicted keff values
#' @export
#' @import bnlearn
#' @import magrittr
#' @import parallel

Sample <- function(
  bn,
  code = 'mcnp',
  cores = parallel::detectCores() / 2,
  metamodel,
  keff.cutoff = 0.9,
  mass.cutoff = 100,
  rad.cutoff = 7,
  sample.size = 1e+09,
  ext.dir) {

  if (!exists('dataset')) dataset <- Tabulate(code, ext.dir)

  # set bindings for nonstandard evaluation
  op <- ctrl <- mass <- rad <- NULL

#
# sample conditional probability tables
#
  # if (cores > 1) {
  #   if (cores > parallel::detectCores()) cores <- parallel::detectCores()
  #   cluster <- parallel::makeCluster(cores)
  # } else {
  #   cluster <- parallel::makeCluster(1)
  # }

  bn.risk <- cpdist(
    bn,
    nodes = names(bn),
    evidence = (as.integer(mass) > 100) & (as.integer(rad) > 7),
    batch = sample.size,
    # cluster = cluster,
    n = sample.size) %>% stats::na.omit()

  # parallel::stopCluster(cluster)

  cat('\nBN predictions generated...') # DELETE

  bn.risk[[3]] <- unlist(bn.risk[[3]]) %>% as.character() %>% as.numeric() # mass
  bn.risk[[4]] <- unlist(bn.risk[[4]])                                     # form
  bn.risk[[5]] <- unlist(bn.risk[[5]])                                     # mod
  bn.risk[[6]] <- unlist(bn.risk[[6]]) %>% as.character() %>% as.numeric() # rad
  bn.risk[[7]] <- unlist(bn.risk[[7]])                                     # ref
  bn.risk[[8]] <- unlist(bn.risk[[8]]) %>% as.character() %>% as.numeric() # thk

  cat('\nBN predictions unlisted...') # DELETE

  # set fissile material density (g/cc)
  fiss.density <- as.character(bn.risk$form)
  fiss.density[fiss.density == 'alpha'] <- 19.86
  fiss.density[fiss.density == 'delta'] <- 15.92
  fiss.density[fiss.density == 'puo2'] <- 11.5
  fiss.density[fiss.density == 'heu'] <- 18.85
  fiss.density[fiss.density == 'uo2'] <- 10.97
  fiss.density <- as.numeric(fiss.density)

  cat('\nDensity set...') # DELETE

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

  cat('\nCalculations performed...') # DELETE

  bn.df <- Scale(
    code = code,
    # dataset = dataset,
    output = subset(bn.risk, select = -c(op, ctrl)),
    ext.dir = ext.dir)

  cat('\nBN processing complete...') # DELETE

#
# predict keff values
#
  if (keff.cutoff > 0) {

    old.len <- nrow(bn.risk) # DELETE

    bn.risk$keff <- metamodel[[1]][[1]] %>% stats::predict(bn.df, verbose = FALSE)

    dec.len <- 0 # decimal length of keff cutoff value

    while (nrow(subset(bn.risk, keff > keff.cutoff)) == 0) {
      dec.len <- as.character(keff.cutoff) %>% strsplit('[.]') %>% unlist() %>% nchar() %>% .[2]
      if (is.na(dec.len)) {
        break
      } else if (dec.len > 1) {
        keff.cutoff <- trunc(keff.cutoff * 100) / 100 - 0.01
      } else {
        keff.cutoff <- trunc(keff.cutoff * 100) / 100 - 0.1
      }
    }

    bn.df <- cbind(bn.df, bn.risk$keff) %>% subset(bn.risk$keff > keff.cutoff)
    bn.df <- bn.df[ , -ncol(bn.df)]
    bn.risk <- subset(bn.risk, keff > keff.cutoff)

    new.len <- nrow(bn.risk) # DELETE

    cat(paste0('\nFirst prediction complete (w/keff cutoff ', old.len, ' --> ', new.len, ')...')) # DELETE

  }

  if (nrow(bn.risk) > 1) {
    if (typeof(metamodel[[2]]) == 'list') {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[2]][[1]]))
      for (i in 1:length(metamodel[[2]][[1]])) {
        cat(paste0('\nLoop prediction started (list ', i, '/', length(metamodel[[2]][[1]]), ')...')) # DELETE
        keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE) %>% suppressWarnings()
        cat(paste0('\nLoop prediction complete (\'list\' ', i, '/', length(metamodel[[2]][[1]]), ')...')) # DELETE
      }
      bn.risk$keff <- rowSums(keff * metamodel[[2]][[1]])
    } else {
      keff <- matrix(nrow = nrow(bn.df), ncol = length(metamodel[[1]]))
      for (i in 1:length(metamodel[[1]])) {
        keff[ , i] <- metamodel[[1]][[i]] %>% stats::predict(bn.df, verbose = FALSE) %>% suppressWarnings()
        cat(paste0('\nLoop prediction complete (\'not list\' ', i, '/', length(metamodel[[1]]), ')...')) # DELETE
      }
      bn.risk$keff <- rowMeans(keff)
    }
  }

  return(bn.risk)

}
