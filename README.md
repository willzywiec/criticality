[![R-CMD-check](https://github.com/willzywiec/criticality/workflows/R-CMD-check/badge.svg)](https://github.com/willzywiec/criticality/actions)

# criticality

A collection of functions for modeling fissile material operations in nuclear facilities  
  
The current R package is written using Keras+TensorFlow.  
  
I thought about adding PyTorch support, but the lack of GPU availability is pushing me towards adopting more cutting-edge tools, such as LeFlow (https://arxiv.org/ftp/arxiv/papers/1807/1807.05317.pdf), which leverage cheaper and more readily available FPGAs to train DNNs. I think the overlap in cryptocurrency mining and AI is eventually going to fracture the hardware market and give birth to systems that are optimized to train DNNs.

## Install the latest release from GitHub:  
```r
devtools::install_github('willzywiec/criticality/pkg/criticality')
```

## Install the current release from CRAN:  
```r
install.packages('criticality')
