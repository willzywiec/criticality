% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scale.R
\name{Scale}
\alias{Scale}
\title{Scale Function}
\usage{
Scale(code = "mcnp", dataset = NULL, output, ext.dir = getwd())
}
\arguments{
\item{code}{Monte Carlo radiation transport code (e.g., "cog", "mcnp")}

\item{dataset}{Training and test data}

\item{output}{Processed output from Monte Carlo radiation transport code simulations}

\item{ext.dir}{External directory (full path)}
}
\description{
This function centers, scales, and one-hot encodes variables.
}