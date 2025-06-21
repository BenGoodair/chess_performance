
if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot)


# Read the data
de_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/DE_daily_stats.csv"))
gb_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/GB_daily_stats.csv"))
fr_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/FR_daily_stats.csv"))
us_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/US_daily_stats.csv"))
in_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/IN_daily_stats.csv"))

# Convert date columns to Date type
de_data$date <- as.Date(de_data$date)
gb_data$date <- as.Date(gb_data$date)
fr_data$date <- as.Date(fr_data$date)
us_data$date <- as.Date(us_data$date)
in_data$date <- as.Date(in_data$date)

# Combine all data
all_data <- rbind(
  data.frame(country = "Germany", de_data),
  data.frame(country = "United Kingdom", gb_data),
  data.frame(country = "France", fr_data),
  data.frame(country = "United States", us_data),
  data.frame(country = "India", in_data)
)

# Define lockdown periods for each country
lockdowns <- data.frame(
  country = c(
    "United Kingdom", "United Kingdom", "United Kingdom",
    "France", "France", "France",
    "Germany", "Germany",
    "United States", "United States",
    "India", "India"
  ),
  start = as.Date(c(
    "2020-03-23", "2020-11-05", "2021-01-06",
    "2020-03-17", "2020-10-30", "2021-04-03",
    "2020-03-22", "2020-11-02",
    "2020-03-19", "2020-11-01",
    "2020-03-25", "2021-04-28"
  )),
  end = as.Date(c(
    "2020-07-04", "2020-12-02", "2021-07-19",
    "2020-05-11", "2020-12-15", "2021-05-03",
    "2020-05-06", "2021-03-07",
    "2020-05-15", "2020-12-15",
    "2020-05-31", "2021-06-30"
  )),
  lockdown_num = c(1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2)
)

# Create a function to add lockdown shading
add_lockdown_shading <- function(plot, country_name) {
  country_lockdowns <- lockdowns[lockdowns$country == country_name, ]
  
  for(i in 1:nrow(country_lockdowns)) {
    plot <- plot + 
      annotate("rect", 
               xmin = country_lockdowns$start[i], 
               xmax = country_lockdowns$end[i],
               ymin = -Inf, ymax = Inf, 
               alpha = 0.2, fill = "red") +
      annotate("text",
               x = country_lockdowns$start[i] + (country_lockdowns$end[i] - country_lockdowns$start[i])/2,
               y = Inf,
               label = paste("Lockdown", country_lockdowns$lockdown_num[i]),
               vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", fontface = "bold")
  }
  return(plot)
}


faceted_plot <- ggplot(all_data, aes(x = date, y = accuracy_mean, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Germany" = "#FF6B6B", "United Kingdom" = "#4ECDC4", "France" = "#45B7D1", "United States" = "#9B59B6", "India" = "#F39C12")) +
  labs(
    title = 'Chess "accuracy" performance by country',
    subtitle = "Mean values",
    x = "Date",
    y = "Mean accuracy",
    color = "Country"
  ) +
  coord_cartesian(xlim = as.Date(c("2019-08-18", "2025-06-18"))) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Add country-specific lockdown shading only for the matching facet
# Create a data frame for lockdown rectangles that can be used with geom_rect
lockdown_rects <- data.frame()
for(country in unique(lockdowns$country)) {
  country_lockdowns <- lockdowns[lockdowns$country == country, ]
  if(nrow(country_lockdowns) > 0) {
    country_rects <- data.frame(
      country = country,
      xmin = country_lockdowns$start,
      xmax = country_lockdowns$end,
      lockdown_num = country_lockdowns$lockdown_num
    )
    lockdown_rects <- rbind(lockdown_rects, country_rects)
  }
}

# Add the rectangles and labels using geom_rect (which respects faceting)
faceted_plot <- faceted_plot +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)+
  theme(plot.title = element_text(hjust = 0.5))


print(faceted_plot)





faceted_plot_elo <- ggplot(all_data, aes(x = date, y = rating_mean, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Germany" = "#FF6B6B", "United Kingdom" = "#4ECDC4", "France" = "#45B7D1", "United States" = "#9B59B6", "India" = "#F39C12")) +
  labs(
    title = "Chess ELO rating performance by country",
    subtitle = "Mean values",
    x = "Date",
    y = "Mean Rating",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Add country-specific lockdown shading only for the matching facet
# Create a data frame for lockdown rectangles that can be used with geom_rect
lockdown_rects <- data.frame()
for(country in unique(lockdowns$country)) {
  country_lockdowns <- lockdowns[lockdowns$country == country, ]
  if(nrow(country_lockdowns) > 0) {
    country_rects <- data.frame(
      country = country,
      xmin = country_lockdowns$start,
      xmax = country_lockdowns$end,
      lockdown_num = country_lockdowns$lockdown_num
    )
    lockdown_rects <- rbind(lockdown_rects, country_rects)
  }
}

# Add the rectangles and labels using geom_rect (which respects faceting)
faceted_plot_elo <- faceted_plot_elo +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)+
  theme(plot.title = element_text(hjust = 0.5))


print(faceted_plot_elo)











faceted_plot_std <- ggplot(all_data, aes(x = date, y = rating_std, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Germany" = "#FF6B6B", "United Kingdom" = "#4ECDC4", "France" = "#45B7D1", "United States" = "#9B59B6", "India" = "#F39C12")) +
  labs(
    title = "",
    subtitle = "Standard deviation values",
    x = "Date",
    y = "Std Rating",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Add country-specific lockdown shading only for the matching facet
# Create a data frame for lockdown rectangles that can be used with geom_rect
lockdown_rects <- data.frame()
for(country in unique(lockdowns$country)) {
  country_lockdowns <- lockdowns[lockdowns$country == country, ]
  if(nrow(country_lockdowns) > 0) {
    country_rects <- data.frame(
      country = country,
      xmin = country_lockdowns$start,
      xmax = country_lockdowns$end,
      lockdown_num = country_lockdowns$lockdown_num
    )
    lockdown_rects <- rbind(lockdown_rects, country_rects)
  }
}

# Add the rectangles and labels using geom_rect (which respects faceting)
faceted_plot_std <- faceted_plot_std +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)+
  theme(plot.title = element_text(hjust = 0.5))


print(faceted_plot_std)









faceted_plot_std_acc <- ggplot(all_data, aes(x = date, y = accuracy_std, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  coord_cartesian(xlim = as.Date(c("2019-08-18", "2025-06-18"))) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Germany" = "#FF6B6B", "United Kingdom" = "#4ECDC4", "France" = "#45B7D1", "United States" = "#9B59B6", "India" = "#F39C12")) +
  labs(
    title = "",
    subtitle = "Standard deviation values",
    x = "Date",
    y = "Std Accuracy",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

# Add country-specific lockdown shading only for the matching facet
# Create a data frame for lockdown rectangles that can be used with geom_rect
lockdown_rects <- data.frame()
for(country in unique(lockdowns$country)) {
  country_lockdowns <- lockdowns[lockdowns$country == country, ]
  if(nrow(country_lockdowns) > 0) {
    country_rects <- data.frame(
      country = country,
      xmin = country_lockdowns$start,
      xmax = country_lockdowns$end,
      lockdown_num = country_lockdowns$lockdown_num
    )
    lockdown_rects <- rbind(lockdown_rects, country_rects)
  }
}

# Add the rectangles and labels using geom_rect (which respects faceting)
faceted_plot_std_acc <- faceted_plot_std_acc +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)+
  theme(plot.title = element_text(hjust = 0.5))


print(faceted_plot_std_acc)





cowplot::plot_grid( faceted_plot_elo,faceted_plot,  faceted_plot_std, faceted_plot_std_acc)

# Create the final combined plot (exactly matching sanctions approach)
divider_quality <- ggdraw() + 
  draw_line(x = c(0.5, 0.5), y = c(0, 1), color = "black", size = 1.5)

# Arrange the plots with the divider in between
final_plot_quality <- plot_grid(
  plot_grid(faceted_plot_elo, faceted_plot_std, ncol = 1),
  divider_quality,
  plot_grid(faceted_plot, faceted_plot_std_acc, ncol = 1),
  ncol = 3,
  rel_widths = c(1, 0.05, 1)
  # Adjust the width of the divider as needed
)+
add_sub(., "Lockdown periods marked in red", x = 0, hjust = 0)





















