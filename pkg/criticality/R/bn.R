# bn.R
#
#' BN Function
#'
#' This function creates a Bayesian network from pre-formatted nuclear facility data.
#' @param dist Truncated probability distribution (e.g., "gamma", "normal")
#' @param facility.data .csv file name
#' @param ext.dir External directory (full path)
#' @return A Bayesian network that models fissile material operations (op), controls (ctrl), 
#'         and parameters that affect nuclear criticality safety
#' @export
#' @import bnlearn
#' @import dplyr
#' @import evd
#' @import fitdistrplus
#' @import magrittr

BN <- function(
  dist = 'gamma',
  facility.data,
  ext.dir) {

  facility <- gsub('.csv', '', facility.data)

  facility.data <- utils::read.csv(paste0(ext.dir, '/', facility.data))

  # set categorical parameters
  op <- table(facility.data$op) %>% names()
  ctrl <- table(facility.data$ctrl) %>% names()
  form <- table(facility.data$form) %>% names()
  mod <- table(facility.data$mod) %>% names()
  ref <- table(facility.data$ref) %>% names()

  # set discrete parameters
  mass <- seq(0, 4000, 1)
  rad <- seq(0, 18, 0.25) * 2.54
  thk <- seq(0, 11, 0.25) * 2.54

  Operation <- function(x) {
    sum(facility.data$op == x) / nrow(facility.data) %>% return()
  }

  Control <- function(x, y) {
    sum(facility.data$op == x & facility.data$ctrl == y) / sum(facility.data$op == x) %>% return()
  }

  Parameter <- function(op, ctrl, par, dist) {

    par.fit <- vector(mode = 'list', length(op)) %>% list()
    par.fit <- rep(par.fit, length(ctrl))

    par.str <- substitute(par) %>% deparse()

    for (i in 1:length(op)) {
      for (j in 1:length(ctrl)) {

        x <- op[i]
        y <- ctrl[j]

        filter.data <- filter(facility.data, (op == x & ctrl == y))
        par.fit[[j]][[i]] <- filter.data[[par.str]]

        if (typeof(par) == 'character' && length(par.fit[[j]][[i]]) > 1) {

          par.df <- data.frame(par.fit[[j]][[i]])
          par.list <- double()

          for (k in par) {
            par.list <- append(par.list, sum(par.df == k) / length(par.fit[[j]][[i]]))
          }

          par.fit[[j]][[i]] <- par.list

        } else if (typeof(par) == 'character' && length(par.fit[[j]][[i]]) == 0) {

          par.fit[[j]][[i]] <- c(1, rep.int(0, length(par) - 1))

        } else if (typeof(par) == 'double' && length(par.fit[[j]][[i]]) > 1) {

          # round up to prevent fitting error
          if (length(unique(par.fit[[j]][[i]])) == 1) {
            par.fit[[j]][[i]][length(par.fit[[j]][[i]])] <- par.fit[[j]][[i]][length(par.fit[[j]][[i]])] %>% ceiling()
          }

          if (dist == 'gamma') {
            par.fit[[j]][[i]] <- fitdist(par.fit[[j]][[i]], distr = 'gamma', method = 'mle') %>% suppressWarnings()
            par.fit[[j]][[i]] <- stats::dgamma(par, rate = par.fit[[j]][[i]]$estimate[[2]], shape = par.fit[[j]][[i]]$estimate[[1]])
            par.fit[[j]][[i]] <- par.fit[[j]][[i]] / sum(par.fit[[j]][[i]])
          } else if (dist == 'gev') {
            par.fit[[j]][[i]] <- fgev(par.fit[[j]][[i]], method = 'BFGS', std.err = FALSE) %>% suppressWarnings()
            par.fit[[j]][[i]] <- dgev(par, loc = par.fit[[j]][[i]]$estimate[[1]], scale = par.fit[[j]][[i]]$estimate[[2]], shape = par.fit[[j]][[i]]$estimate[[3]])
            par.fit[[j]][[i]] <- par.fit[[j]][[i]] / sum(par.fit[[j]][[i]])
          } else if (dist == 'normal') {
            par.fit[[j]][[i]] <- fitdist(par.fit[[j]][[i]], distr = 'norm', method = 'mle') %>% suppressWarnings()
            par.fit[[j]][[i]] <- stats::dnorm(par, mean = par.fit[[j]][[i]]$estimate[[1]], sd = par.fit[[j]][[i]]$estimate[[2]])
            par.fit[[j]][[i]] <- par.fit[[j]][[i]] / sum(par.fit[[j]][[i]])
          } else if (dist == 'log-normal') {
            par.fit[[j]][[i]] <- fitdist(par.fit[[j]][[i]], distr = 'lnorm', method = 'mle') %>% suppressWarnings()
            par.fit[[j]][[i]] <- stats::dlnorm(par, meanlog = par.fit[[j]][[i]]$estimate[[1]], sdlog = par.fit[[j]][[i]]$estimate[[2]])
            par.fit[[j]][[i]] <- par.fit[[j]][[i]] / sum(par.fit[[j]][[i]])
          } else if (dist == 'weibull') {
            par.fit[[j]][[i]] <- fitdist(par.fit[[j]][[i]], distr = 'weibull', method = 'mle') %>% suppressWarnings()
            par.fit[[j]][[i]] <- stats::dweibull(par, shape = par.fit[[j]][[i]]$estimate[[1]], scale = par.fit[[j]][[i]]$estimate[[2]])
            par.fit[[j]][[i]] <- par.fit[[j]][[i]] / sum(par.fit[[j]][[i]])
          }

        } else if (typeof(par) == 'double' && length(par.fit[[j]][[i]]) == 0) {

          par.fit[[j]][[i]] <- c(1, rep.int(0, length(par) - 1))

        }

      }
    }

    return(par.fit)
    
  }

#
# build graph
#
  nodes <- c('op', 'ctrl', 'mass', 'form', 'mod', 'rad', 'ref', 'thk')
  dag <- empty.graph(nodes = nodes)

  for (i in 2:length(nodes)) {
    dag <- set.arc(dag, 'op', nodes[i])
    if (i > 2) dag <- set.arc(dag, 'ctrl', nodes[i])
  }

  op.cpt <- ctrl.cpt <- numeric()

  for (i in op) {
    op.cpt <- append(op.cpt, Operation(i))
    for (j in ctrl) {
      ctrl.cpt <- append(ctrl.cpt, Control(i, j))
    }
  }

  op.cpt <- matrix(op.cpt, nrow = 6, ncol = 1, dimnames = list('op' = op))
  ctrl.cpt <- matrix(ctrl.cpt, nrow = 7, ncol = 6, dimnames = list('ctrl' = ctrl, 'op' = op))

#
# fit parameters
#
  mass.cpt <- Parameter(op, ctrl, mass, dist) %>% unlist()
  mass.cpt <- array(mass.cpt, dim = c(length(mass), length(ctrl), length(op)), dimnames = list('mass' = mass, 'ctrl' = ctrl, 'op' = op))

  form.cpt <- Parameter(op, ctrl, form, dist) %>% unlist()
  form.cpt <- array(form.cpt, dim = c(length(form), length(ctrl), length(op)), dimnames = list('form' = form, 'ctrl' = ctrl, 'op' = op))

  mod.cpt <- Parameter(op, ctrl, mod, dist) %>% unlist()
  mod.cpt <- array(mod.cpt, dim = c(length(mod), length(ctrl), length(op)), dimnames = list('mod' = mod, 'ctrl' = ctrl, 'op' = op))

  rad.cpt <- Parameter(op, ctrl, rad, dist) %>% unlist()
  rad.cpt <- array(rad.cpt, dim = c(length(rad), length(ctrl), length(op)), dimnames = list('rad' = rad, 'ctrl' = ctrl, 'op' = op))

  ref.cpt <- Parameter(op, ctrl, ref, dist) %>% unlist()
  ref.cpt <- array(ref.cpt, dim = c(length(ref), length(ctrl), length(op)), dimnames = list('ref' = ref, 'ctrl' = ctrl, 'op' = op))

  thk.cpt <- Parameter(op, ctrl, thk, dist) %>% unlist()
  thk.cpt <- array(thk.cpt, dim = c(length(thk), length(ctrl), length(op)), dimnames = list('thk' = thk, 'ctrl' = ctrl, 'op' = op))

#
# build Bayesian network
#
  bn <- list(
    op = op.cpt,
    ctrl = ctrl.cpt,
    mass = mass.cpt,
    form = form.cpt,
    mod = mod.cpt,
    rad = rad.cpt,
    ref = ref.cpt,
    thk = thk.cpt)
  
  bn <- custom.fit(dag, dist = bn)

  saveRDS(bn, file = paste0(ext.dir, '/', facility, '-', dist, '.RData'))

  return(bn)

}
