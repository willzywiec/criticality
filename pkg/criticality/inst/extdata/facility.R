#' facility.csv
#'
#' A dataset containing 10,000 "facility walkthrough" samples that describe fissile material operations in a nuclear facility
#' 
#' @format A data frame with 10,000 rows and 8 variables:
#' \describe{
#'   \item{op}{fissile material operation}
#'   \item{ctrl}{criticality control set}
#'   \item{mass}{fissile material mass}
#'   \item{form}{fissile material form (e.g., "alpha", "delta", "heu")}
#'   \item{mod}{training moderation}
#'   \item{rad}{radius (cm)}
#'   \item{ref}{reflector}
#'   \item{thk}{reflector thickness (cm)}
#' }
#' 
#' @source \url{https://wwww.github.com/willzywiec/criticality/data/facility.csv}
"facility"
