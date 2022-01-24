#' mcnp-dataset.RData
#'
#' A dataset based on 16,752 MCNP simulations
#' 
#' @format A list of data frames based on the "output" dataset, which has 16,752 rows and 10 variables:
#' \describe{
#'   \item{output}{MCNP simulation output}
#'   \item{training.data}{preprocessed training data (80% of output)}
#'   \item{training.mean}{mean training data values}
#'   \item{training.sd}{standard deviation of mean training data values}
#'   \item{training.df}{processed training data (80% of output, no keff or sd values)}
#'   \item{test.data}{preprocessed test data (20% of output)}
#'   \item{test.df}{processed test data (20% of output, no keff or sd values)}
#' }
#' 
#' @source \url{https://wwww.github.com/willzywiec/criticality/data/mcnp-data.RData}
"mcnp-dataset: dataset"
