
if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot)


# Read the data
de_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/DE_daily_stats.csv"))
gb_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/GB_daily_stats.csv"))
fr_data <- read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/FR_daily_stats.csv"))

# Convert date columns to Date type
de_data$date <- as.Date(de_data$date)
gb_data$date <- as.Date(gb_data$date)
fr_data$date <- as.Date(fr_data$date)

# Combine all data
all_data <- rbind(
  data.frame(country = "Germany", de_data),
  data.frame(country = "United Kingdom", gb_data),
  data.frame(country = "France", fr_data)
)

# Define lockdown periods for each country
lockdowns <- data.frame(
  country = c(
    "United Kingdom", "United Kingdom", "United Kingdom",
    "France", "France", "France",
    "Germany", "Germany"
  ),
  start = as.Date(c(
    "2020-03-23", "2020-11-05", "2021-01-06",
    "2020-03-17", "2020-10-30", "2021-04-03",
    "2020-03-22", "2020-11-02"
  )),
  end = as.Date(c(
    "2020-07-04", "2020-12-02", "2021-07-19",
    "2020-05-11", "2020-12-15", "2021-05-03",
    "2020-05-06", "2021-03-07"
  )),
  lockdown_num = c(1, 2, 3, 1, 2, 3, 1, 2)
)



faceted_plot <- ggplot(all_data, aes(x = date, y = accuracy_mean, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Germany" = "#FF6B6B", "United Kingdom" = "#4ECDC4", "France" = "#45B7D1")) +
  labs(
    title = "Chess accuracy Performance by Country",
    subtitle = "Individual country trends with their specific lockdown periods",
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
            fontface = "bold", inherit.aes = FALSE)

print(faceted_plot)

# 5. Summary statistics during lockdown vs non-lockdown periods
# Create a function to determine if a date is during lockdown
is_lockdown <- function(date, country) {
  country_lockdowns <- lockdowns[lockdowns$country == country, ]
  for(i in 1:nrow(country_lockdowns)) {
    if(date >= country_lockdowns$start[i] & date <= country_lockdowns$end[i]) {
      return(TRUE)
    }
  }
  return(FALSE)
}

# Add lockdown indicator to data
all_data$in_lockdown <- mapply(is_lockdown, all_data$date, all_data$country)

# Calculate summary statistics
summary_stats <- all_data %>%
  filter(!is.na(rating_mean) & !is.na(accuracy_mean)) %>%
  group_by(country, in_lockdown) %>%
  summarise(
    mean_rating = mean(rating_mean, na.rm = TRUE),
    mean_accuracy = mean(accuracy_mean, na.rm = TRUE),
    mean_players = mean(rating_count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(period = ifelse(in_lockdown, "Lockdown", "Normal"))

print("Summary Statistics:")
print(summary_stats)

# Create a comparison plot
comparison_plot <- summary_stats %>%
  filter(!is.na(mean_rating)) %>%
  ggplot(aes(x = country, y = mean_rating, fill = period)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Lockdown" = "#FF6B6B", "Normal" = "#4ECDC4")) +
  labs(
    title = "Average Chess Rating: Lockdown vs Normal Periods",
    x = "Country",
    y = "Mean Rating",
    fill = "Period"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  ) +
  geom_text(aes(label = round(mean_rating, 0)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3)

print(comparison_plot)
