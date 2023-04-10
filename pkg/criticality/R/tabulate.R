# tabulate.R
#
#' Tabulate Function
#'
#' This function loads/saves training and test data (imports Scale function).
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param ext.dir External directory (full path)
#' @return A list of centered, scaled, and one-hot-encoded training and test data
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
#' Tabulate(
#'   ext.dir = ext.dir
#' )
#'
#' @import magrittr

Tabulate <- function(
  code = 'mcnp',
  ext.dir) {

  code <- tolower(code)

  dataset.rdata <- list.files(path = ext.dir)[grep(paste0(code, '.*RData$'), list.files(path = ext.dir), ignore.case = TRUE)]

  output.csv <- list.files(path = ext.dir)[grep(paste0(code, '.*csv$'), list.files(path = ext.dir), ignore.case = TRUE)]

  if (file.exists(paste0(ext.dir, '/', dataset.rdata)) && !identical(dataset.rdata, character(0))) {

    load(paste0(ext.dir, '/', dataset.rdata))

  } else {

    if (file.exists(paste0(ext.dir, '/', output.csv)) && !identical(output.csv, character(0))) {

      output <- utils::read.csv(paste0(ext.dir, '/', output.csv), fileEncoding = 'UTF-8-BOM') %>% stats::na.omit()

      output <- output[sample(nrow(output)), ]

      # calculate vol (cc)
      vol <- 4/3 * pi * output$rad^3

      # calculate conc (g/cc)
      conc <- output$mass / vol

      output <- data.frame(
        mass = output$mass,
        form = output$form,
        mod = output$mod,
        rad = output$rad,
        ref = output$ref,
        thk = output$thk,
        shape = output$shape,
        vol = vol,
        conc = conc,
        keff = output$keff,
        sd = output$sd)

      dataset <- Scale(code = code, output = output, ext.dir = ext.dir)

    } else {

      stop('could not find data', call. = FALSE)

    }

  }

  return(dataset)

}
