# tabulate.R
#
# William Zywiec
#
#' Tabulate Function
#'
#' This function imports the Scale function and loads/saves training and test data.
#' @param code Monte Carlo radiation transport code (e.g., "cog", "mcnp")
#' @param ext.dir External directory
#' @export
#' @examples
#' Tabulate(code, ext.dir)

Tabulate <- function(
  code = 'mcnp',
  ext.dir) {

  library(magrittr)

  data.dir <- paste0(ext.dir, '/data')
  dir.create(ext.dir, recursive = TRUE, showWarnings = FALSE)

  setwd(data.dir)

  if (file.exists(paste0(code, '-dataset.RData'))) {

    dataset <- readRDS(paste0(code, '-dataset.RData'))
    cat('Loaded ', code, '-dataset.RData\n', sep = '')

  } else {

    output.files <- list.files(pattern = '\\.o$')

    # load output
    if (file.exists(paste0(code, '-output.csv'))) {

      output <- read.csv(paste0(code, '-output.csv'), fileEncoding = 'UTF-8-BOM') %>% na.omit()

      if (nrow(output) >= length(output.files)) {
        output <- output[sample(nrow(output)), ]
        dataset <- Scale(code, output)
        cat('Loaded ', code, '-dataset.RData\n', sep = '')
      } else {
        remove(output)
      }
      
    } else {

      mass <- rad <- thk <- ht <- vol <- conc <- hd <- keff <- sd <- numeric()
      
      form <- mod <- ref <- shape <- character()

      for (i in 1:length(output.files)) {

        if (any(grep('final result', readLines(output.files[i])))) {

          # set mass (g), form, mod, rad (cm), and ref
          file.name <- gsub('\\.o', '', output.files[i]) %>% strsplit('-') %>% unlist()
          mass[i] <- as.numeric(file.name[1])
          form[i] <- file.name[2]
          mod[i] <- file.name[3]
          rad[i] <- as.numeric(file.name[4])
          ref[i] <- file.name[5]

          # set thk (cm) and shape
          if (ref[i] == 'none') {
            thk[i] <- 0
            shape[i] <- file.name[6]
          } else {
            thk[i] <- as.numeric(file.name[6])
            shape[i] <- file.name[7]
          }

          # set ht (cm)
          if (shape[i] == 'sph') {
            ht[i] <- 2 * rad[i]
          } else if (ref[i] == 'none') {
            ht[i] <- as.numeric(file.name[7])
          } else {
            ht[i] <- as.numeric(file.name[8])
          }

          # calculate vol (cc)
          if (shape[i] == 'sph') {
            vol[i] <- 4/3 * pi * rad[i]^3
          } else if (shape[i] == 'rcc') {
            vol[i] <- pi * rad[i]^2 * ht[i]
          }

          # calculate conc (g/cc) and h/d
          conc[i] <- (mass[i] / vol[i])
          hd[i] <- (ht[i] / (2 * rad[i]))

          # set keff and sd
          final.result <- grep('final result', readLines(output.files[i]), value = TRUE) %>% strsplit('\\s+') %>% unlist()
          keff[i] <- final.result[4]
          sd[i] <- final.result[5]

        }

      }

      output <- data.frame(
        mass = mass,
        form = form,
        mod = mod,
        rad = rad,
        ref = ref,
        thk = thk,
        shape = shape,
        ht = ht,
        vol = vol,
        conc = conc,
        hd = hd,
        keff = keff,
        sd = sd)

      output <- output[sample(nrow(output)), ]
      write.csv(output, file = paste0(code, '-output.csv'), row.names = FALSE)
      
      dataset <- Scale(code, output)
      cat('Loaded ', code, '-dataset.RData\n', sep = '')

    } 

    if (!exists('data') && length(output.files) == 0) {
      stop('Could not find data\n')
    }

  }

  return(dataset)

}
