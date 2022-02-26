# plot.R
#
#' Plot Function
#'
#' This function generates and saves plots and data.
#' @param i Model number
#' @param history Training history
#' @param plot.dir Plot directory (full path)
#' @return No output (generates and saves ggplot2 files and training histories)
#' @export
#' @import ggplot2
#' @import magrittr
#' @import scales

Plot <- function(
  i,
  history = NULL,
  plot.dir) {

  # set theme
  new.theme <- theme_gray() + theme(axis.text = element_text(color = 'black', size = 11), text = element_text(color = 'black', family = 'serif', size = 11))
  theme_set(new.theme)

  if (is.null(history)) {
    history <- utils::read.csv(paste0(plot.dir, '/', i, '.csv'), header = TRUE)
  } else {
    history <- data.frame(
      epoch = 1:length(history$metrics$val_loss),
      val.loss = history$metrics$val_loss,
      val.mae = history$metrics$val_mean_absolute_error,
      loss = history$metrics$loss,
      mae = history$metrics$mean_absolute_error)
    utils::write.csv(history, file = paste0(plot.dir, '/', i, '.csv'), row.names = FALSE)
  }

  ggplot(history, aes_string(x = 'epoch')) +
  geom_line(aes_string(y = 'val.mae', color = shQuote('cross-validation data'))) +
  geom_line(aes_string(y = 'mae', color = shQuote('training data'))) +
  geom_point(aes(x = which.min(history$mae), y = min(history$mae), color = 'training minimum')) +
  guides(color = guide_legend(override.aes = list(linetype = c(1, 1, NA), shape = c(NA, NA, 16)))) +
  scale_color_manual('', breaks = c('training data', 'cross-validation data', 'training minimum'), values = c('black', '#a9a9a9', 'red')) +
  scale_x_continuous(breaks = pretty_breaks()) +
  scale_y_log10(breaks = c(1e-04, 1e-03, 1e-02, 1e-01, 1e+00), limits = c(1e-04, 1e+00)) +
  theme(
    legend.position = 'bottom',
    legend.spacing.x = unit(0.2, 'cm'),
    legend.text = element_text(size = 11),
    legend.title = element_blank()) +
  ylab('mean absolute error') +
  annotate(
    geom = 'text',
    x = which.min(history$mae),
    y = min(history$mae),
    vjust = 1.9,
    label = format(min(history$mae), digits = 3, scientific = TRUE),
    color = 'red',
    family = 'serif',
    size = 3.5)

  ggsave(paste0(plot.dir, '/', i, '.png'), dpi = 1000, height = 4, width = 6.5) %>% suppressMessages() %>% suppressWarnings()

}
