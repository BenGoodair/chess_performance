if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot)

# =============================================================================
# COUNTRY SELECTION - MODIFY THIS SECTION TO CHOOSE COUNTRIES
# =============================================================================

# Available countries: "Germany", "United Kingdom", "France", "United States", "India", "Ukraine", "Russia"
#SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France", "United States", "India")

# Alternative examples:
# SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France")  # Just Europe
 SELECTED_COUNTRIES <- c("United States", "Russia", "Ukraine")   # US + Eastern Europe
# SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France", "United States", "India", "Ukraine", "Russia")  # All countries

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

# Read all available data
country_data <- list(
  "Germany" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/DE_daily_stats.csv")),
  "United Kingdom" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/GB_daily_stats.csv")),
  "France" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/FR_daily_stats.csv")),
  "United States" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/US_daily_stats.csv")),
  "India" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/IN_daily_stats.csv")),
  "Ukraine" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/UA_daily_stats.csv")),
  "Russia" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/RU_daily_stats.csv"))
)

# Convert date columns to Date type for all datasets
for(country in names(country_data)) {
  country_data[[country]]$date <- as.Date(country_data[[country]]$date)
}

# Filter and combine only selected countries
selected_data_list <- country_data[SELECTED_COUNTRIES]
all_data <- do.call(rbind, lapply(names(selected_data_list), function(country) {
  data.frame(country = country, selected_data_list[[country]])
}))

# Define all lockdown periods
all_lockdowns <- data.frame(
  country = c(
    "United Kingdom", "United Kingdom", "United Kingdom",
    "France", "France", "France",
    "Germany", "Germany",
    "United States", "United States",
    "India", "India",
    "Ukraine", "Ukraine",
    "Russia", "Russia", "Russia"
  ),
  start = as.Date(c(
    "2020-03-23", "2020-11-05", "2021-01-06",
    "2020-03-17", "2020-10-30", "2021-04-03",
    "2020-03-22", "2020-11-02",
    "2020-03-19", "2020-11-01",
    "2020-03-25", "2021-04-28",
    "2020-03-17", "2020-10-08",
    "2020-03-30", "2020-10-05", "2021-06-13"
  )),
  end = as.Date(c(
    "2020-07-04", "2020-12-02", "2021-07-19",
    "2020-05-11", "2020-12-15", "2021-05-03",
    "2020-05-06", "2021-03-07",
    "2020-05-15", "2020-12-15",
    "2020-05-31", "2021-06-30",
    "2020-05-11", "2021-01-24",
    "2020-05-12", "2020-12-08", "2021-07-11"
  )),
  lockdown_num = c(1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3)
)

# Filter lockdowns for selected countries only
lockdowns <- all_lockdowns[all_lockdowns$country %in% SELECTED_COUNTRIES, ]

# Define color palette for all countries
all_colors <- c(
  "Germany" = "#FF6B6B", 
  "United Kingdom" = "#4ECDC4", 
  "France" = "#45B7D1", 
  "United States" = "#9B59B6", 
  "India" = "#F39C12", 
  "Ukraine" = "#E74C3C", 
  "Russia" = "#8E44AD"
)

# Filter colors for selected countries
selected_colors <- all_colors[SELECTED_COUNTRIES]

# =============================================================================
# PLOT CREATION FUNCTIONS
# =============================================================================

# Function to create lockdown rectangles data frame
create_lockdown_rects <- function(lockdown_data) {
  lockdown_rects <- data.frame()
  for(country in unique(lockdown_data$country)) {
    country_lockdowns <- lockdown_data[lockdown_data$country == country, ]
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
  return(lockdown_rects)
}

# Create lockdown rectangles
lockdown_rects <- create_lockdown_rects(lockdowns)

# =============================================================================
# PLOT GENERATION
# =============================================================================

# Chess accuracy plot
faceted_plot <- ggplot(all_data, aes(x = date, y = accuracy_mean, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = selected_colors) +
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
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)

# Chess ELO rating plot
faceted_plot_elo <- ggplot(all_data, aes(x = date, y = rating_mean, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = selected_colors) +
  labs(
    title = "Chess ELO rating performance by country",
    subtitle = "Mean values",
    x = "Date",
    y = "Mean Rating",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)

# Rating standard deviation plot
faceted_plot_std <- ggplot(all_data, aes(x = date, y = rating_std, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = selected_colors) +
  labs(
    title = "",
    subtitle = "Standard deviation values",
    x = "Date",
    y = "Std Rating",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)

# Accuracy standard deviation plot
faceted_plot_std_acc <- ggplot(all_data, aes(x = date, y = accuracy_std, color = country)) +
  geom_line(size = 0.8, alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, size = 1.2) +
  coord_cartesian(xlim = as.Date(c("2019-08-18", "2025-06-18"))) +
  facet_wrap(~country, scales = "free_y", ncol = 1) +
  scale_color_manual(values = selected_colors) +
  labs(
    title = "",
    subtitle = "Standard deviation values",
    x = "Date",
    y = "Std Accuracy",
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, color = "gray60"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12)
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  geom_rect(data = lockdown_rects, 
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            alpha = 0.2, fill = "red", inherit.aes = FALSE) +
  geom_text(data = lockdown_rects,
            aes(x = xmin + (xmax - xmin)/2, y = Inf, 
                label = paste("L", lockdown_num)),
            vjust = 1.2, hjust = 0.5, size = 3, color = "darkred", 
            fontface = "bold", inherit.aes = FALSE)

# =============================================================================
# DISPLAY PLOTS
# =============================================================================

# print(faceted_plot)
# print(faceted_plot_elo)
# print(faceted_plot_std)
# print(faceted_plot_std_acc)

# Grid of all plots
cowplot::plot_grid(faceted_plot_elo, faceted_plot, faceted_plot_std, faceted_plot_std_acc)

# Create the final combined plot (exactly matching sanctions approach)
divider_quality <- ggdraw() + 
  draw_line(x = c(0.5, 0.5), y = c(0, 1), color = "black", size = 1.5)

# Arrange the plots with the divider in between
final_plot <- plot_grid(
  plot_grid(faceted_plot_elo, faceted_plot_std, ncol = 1),
  divider_quality,
  plot_grid(faceted_plot, faceted_plot_std_acc, ncol = 1),
  ncol = 3,
  rel_widths = c(1, 0.05, 1)
)

print(final_plot)




####russia invasion ####
if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot, cowplot)

# =============================================================================
# ANALYSIS: IMPACT OF RUSSIAN INVASION ON CHESS PERFORMANCE
# =============================================================================

# Focus on Russia and Ukraine for invasion impact analysis
SELECTED_COUNTRIES <- c("Russia", "Ukraine")

# Read data for Russia and Ukraine
country_data <- list(
  "Ukraine" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/UA_daily_stats.csv")),
  "Russia" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/RU_daily_stats.csv"))
)

# Convert date columns to Date type
for(country in names(country_data)) {
  country_data[[country]]$date <- as.Date(country_data[[country]]$date)
}

# Combine data
all_data <- do.call(rbind, lapply(names(country_data), function(country) {
  data.frame(country = country, country_data[[country]])
}))

# Define invasion date
invasion_date <- as.Date("2022-02-24")

# Add invasion indicator (handle NA dates)
all_data$post_invasion <- ifelse(is.na(all_data$date), NA, all_data$date >= invasion_date)

# Define time periods for analysis (handle NA dates)
all_data$period <- ifelse(is.na(all_data$date), NA,
                          ifelse(all_data$date < as.Date("2020-03-01"), "Pre-COVID",
                                 ifelse(all_data$date < invasion_date, "COVID Era", "Post-Invasion")))

# Define lockdown periods (from your original code)
lockdowns <- data.frame(
  country = c("Ukraine", "Ukraine", "Russia", "Russia", "Russia"),
  start = as.Date(c("2020-03-17", "2020-10-08", "2020-03-30", "2020-10-05", "2021-06-13")),
  end = as.Date(c("2020-05-11", "2021-01-24", "2020-05-12", "2020-12-08", "2021-07-11")),
  lockdown_num = c(1, 2, 1, 2, 3)
)

# Create lockdown rectangles
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

# Define colors
colors <- c("Russia" = "#8E44AD", "Ukraine" = "#E74C3C")

# =============================================================================
# TREND-ADJUSTED ANALYSIS FOR INVASION IMPACT
# =============================================================================

# Function to perform trend-adjusted analysis using regression discontinuity
perform_trend_adjusted_analysis <- function(data, metric) {
  results <- data.frame()
  
  for(country in unique(data$country)) {
    country_data <- data[data$country == country & !is.na(data[[metric]]) & !is.na(data$post_invasion), ]
    
    # Filter to reasonable time window around invasion (18 months before and after)
    analysis_data <- country_data[country_data$date >= as.Date("2020-08-24") & 
                                    country_data$date <= as.Date("2023-08-24"), ]
    
    if(nrow(analysis_data) > 50) {  # Need sufficient data for regression
      # Create time variables
      analysis_data$days_since_start <- as.numeric(analysis_data$date - min(analysis_data$date))
      analysis_data$days_since_invasion <- as.numeric(analysis_data$date - invasion_date)
      analysis_data$post_invasion_numeric <- as.numeric(analysis_data$post_invasion)
      
      # Regression discontinuity model: accounts for pre-existing trend + invasion effect
      # Model: metric ~ trend + invasion_dummy + trend_change_after_invasion
      tryCatch({
        model <- lm(paste(metric, "~ days_since_start + post_invasion_numeric + I(days_since_invasion * post_invasion_numeric)"), 
                    data = analysis_data)
        
        # Extract coefficients
        coeffs <- summary(model)$coefficients
        
        # The invasion effect is the coefficient on post_invasion_numeric
        invasion_effect <- coeffs["post_invasion_numeric", "Estimate"]
        invasion_p_value <- coeffs["post_invasion_numeric", "Pr(>|t|)"]
        
        # Predicted values at invasion date (controlling for trend)
        pre_invasion_predicted <- predict(model, newdata = data.frame(
          days_since_start = as.numeric(invasion_date - min(analysis_data$date)),
          post_invasion_numeric = 0,
          days_since_invasion = 0
        ))
        
        post_invasion_predicted <- predict(model, newdata = data.frame(
          days_since_start = as.numeric(invasion_date - min(analysis_data$date)),
          post_invasion_numeric = 1,
          days_since_invasion = 0
        ))
        
        results <- rbind(results, data.frame(
          country = country,
          metric = metric,
          pre_trend_slope = coeffs["days_since_start", "Estimate"],
          invasion_effect = invasion_effect,
          invasion_p_value = invasion_p_value,
          invasion_significant = invasion_p_value < 0.05,
          r_squared = summary(model)$r.squared,
          pre_invasion_predicted = pre_invasion_predicted,
          post_invasion_predicted = post_invasion_predicted,
          trend_adjusted_difference = post_invasion_predicted - pre_invasion_predicted
        ))
      }, error = function(e) {
        print(paste("Error in regression for", country, metric, ":", e$message))
      })
    }
  }
  return(results)
}

# Alternative: Interrupted Time Series Analysis
perform_its_analysis <- function(data, metric) {
  results <- data.frame()
  
  for(country in unique(data$country)) {
    country_data <- data[data$country == country & !is.na(data[[metric]]) & !is.na(data$post_invasion), ]
    
    # Filter to time window around invasion
    analysis_data <- country_data[country_data$date >= as.Date("2020-08-24") & 
                                    country_data$date <= as.Date("2023-08-24"), ]
    
    if(nrow(analysis_data) > 50) {
      # Create time variables for ITS
      analysis_data$time <- 1:nrow(analysis_data)
      analysis_data$intervention <- as.numeric(analysis_data$post_invasion)
      analysis_data$time_after_intervention <- ifelse(analysis_data$post_invasion, 
                                                      analysis_data$time - which(analysis_data$post_invasion)[1] + 1, 0)
      
      tryCatch({
        # ITS model: metric ~ time + intervention + time_after_intervention
        its_model <- lm(paste(metric, "~ time + intervention + time_after_intervention"), data = analysis_data)
        coeffs <- summary(its_model)$coefficients
        
        # Level change at intervention
        level_change <- coeffs["intervention", "Estimate"]
        level_p_value <- coeffs["intervention", "Pr(>|t|)"]
        
        # Slope change after intervention
        slope_change <- coeffs["time_after_intervention", "Estimate"]
        slope_p_value <- coeffs["time_after_intervention", "Pr(>|t|)"]
        
        results <- rbind(results, data.frame(
          country = country,
          metric = metric,
          pre_trend = coeffs["time", "Estimate"],
          level_change = level_change,
          level_change_p = level_p_value,
          slope_change = slope_change,
          slope_change_p = slope_p_value,
          level_significant = level_p_value < 0.05,
          slope_significant = slope_p_value < 0.05,
          r_squared = summary(its_model)$r.squared
        ))
      }, error = function(e) {
        print(paste("Error in ITS analysis for", country, metric, ":", e$message))
      })
    }
  }
  return(results)
}

# Perform trend-adjusted analyses
print("=== REGRESSION DISCONTINUITY ANALYSIS ===")
rd_rating <- perform_trend_adjusted_analysis(all_data, "rating_mean")
rd_accuracy <- perform_trend_adjusted_analysis(all_data, "accuracy_mean")
rd_rating_std <- perform_trend_adjusted_analysis(all_data, "rating_std")
rd_accuracy_std <- perform_trend_adjusted_analysis(all_data, "accuracy_std")

rd_results <- rbind(rd_rating, rd_accuracy, rd_rating_std, rd_accuracy_std)
print(rd_results)

print("\n=== INTERRUPTED TIME SERIES ANALYSIS ===")
its_rating <- perform_its_analysis(all_data, "rating_mean")
its_accuracy <- perform_its_analysis(all_data, "accuracy_mean")
its_rating_std <- perform_its_analysis(all_data, "rating_std")
its_accuracy_std <- perform_its_analysis(all_data, "accuracy_std")

its_results <- rbind(its_rating, its_accuracy, its_rating_std, its_accuracy_std)
print(its_results)

# Enhanced visualization with trend lines and residuals
create_trend_adjusted_plots <- function(data, metric, title_prefix) {
  
  # Create detrended data for clearer visualization
  detrended_data <- data.frame()
  
  for(country in unique(data$country)) {
    country_data <- data[data$country == country & !is.na(data[[metric]]), ]
    
    if(nrow(country_data) > 50) {
      # Fit pre-invasion trend
      pre_invasion_data <- country_data[country_data$date < invasion_date & 
                                          country_data$date >= as.Date("2020-01-01"), ]
      
      if(nrow(pre_invasion_data) > 20) {
        pre_invasion_data$days <- as.numeric(pre_invasion_data$date - min(pre_invasion_data$date))
        trend_model <- lm(paste(metric, "~ days"), data = pre_invasion_data)
        
        # Apply detrending to all data
        country_data$days <- as.numeric(country_data$date - min(pre_invasion_data$date))
        country_data$predicted_trend <- predict(trend_model, newdata = country_data)
        country_data$detrended <- country_data[[metric]] - country_data$predicted_trend + mean(country_data[[metric]], na.rm = TRUE)
        
        detrended_data <- rbind(detrended_data, country_data)
      }
    }
  }
  
  # Plot original data with trend lines
  p1 <- ggplot(data, aes(x = date, y = .data[[metric]], color = country)) +
    geom_line(alpha = 0.3) +
    geom_smooth(method = "loess", span = 0.3, se = TRUE, alpha = 0.2) +
    facet_wrap(~country, scales = "free_y", ncol = 1) +
    scale_color_manual(values = colors) +
    labs(
      title = paste(title_prefix, "- Original Data with Trends"),
      x = "Date", y = metric
    ) +
    theme_minimal() +
    coord_cartesian(xlim = as.Date(c("2020-01-01", "2023-06-01"))) +
    geom_vline(xintercept = invasion_date, color = "red", linetype = "dashed", size = 1) +
    theme(legend.position = "none", strip.text = element_text(face = "bold"))
  
  # Plot detrended data
  if(nrow(detrended_data) > 0) {
    p2 <- ggplot(detrended_data, aes(x = date, y = detrended, color = country)) +
      geom_line(alpha = 0.5) +
      geom_smooth(method = "loess", span = 0.3, se = FALSE) +
      facet_wrap(~country, scales = "free_y", ncol = 1) +
      scale_color_manual(values = colors) +
      labs(
        title = paste(title_prefix, "- Detrended Data"),
        subtitle = "Pre-invasion trend removed",
        x = "Date", y = paste("Detrended", metric)
      ) +
      theme_minimal() +
      coord_cartesian(xlim = as.Date(c("2020-01-01", "2023-06-01"))) +
      geom_vline(xintercept = invasion_date, color = "red", linetype = "dashed", size = 1) +
      geom_hline(yintercept = 0, color = "gray50", linetype = "dotted") +
      theme(legend.position = "none", strip.text = element_text(face = "bold"))
    
    return(list(original = p1, detrended = p2))
  }
  
  return(list(original = p1, detrended = NULL))
}

# Create trend-adjusted visualizations
rating_plots <- create_trend_adjusted_plots(all_data, "rating_mean", "Chess Rating")
accuracy_plots <- create_trend_adjusted_plots(all_data, "accuracy_mean", "Chess Accuracy")

print(rating_plots$original)
if(!is.null(rating_plots$detrended)) print(rating_plots$detrended)

print(accuracy_plots$original)
if(!is.null(accuracy_plots$detrended)) print(accuracy_plots$detrended)

# =============================================================================
# SUMMARY STATISTICS BY PERIOD
# =============================================================================

# Calculate summary statistics by period and country
summary_stats <- all_data %>%
  group_by(country, period) %>%
  summarise(
    mean_rating = mean(rating_mean, na.rm = TRUE),
    mean_accuracy = mean(accuracy_mean, na.rm = TRUE),
    mean_rating_std = mean(rating_std, na.rm = TRUE),
    mean_accuracy_std = mean(accuracy_std, na.rm = TRUE),
    observations = n(),
    .groups = 'drop'
  )

print("=== SUMMARY STATISTICS BY PERIOD ===")
print(summary_stats)

# Calculate specific before/after invasion comparison (1 year window)
invasion_comparison <- all_data %>%
  filter(date >= as.Date("2021-02-24") & date <= as.Date("2023-02-24") & !is.na(post_invasion)) %>%
  mutate(period_invasion = ifelse(post_invasion, "Post-Invasion", "Pre-Invasion")) %>%
  group_by(country, period_invasion) %>%
  summarise(
    mean_rating = mean(rating_mean, na.rm = TRUE),
    mean_accuracy = mean(accuracy_mean, na.rm = TRUE),
    mean_rating_std = mean(rating_std, na.rm = TRUE),
    mean_accuracy_std = mean(accuracy_std, na.rm = TRUE),
    observations = n(),
    .groups = 'drop'
  )

print("=== INVASION IMPACT COMPARISON (1-year window) ===")
print(invasion_comparison)







if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot, cowplot, bcp)

# =============================================================================
# DIFFERENCE-IN-DIFFERENCES ANALYSIS: INVASION IMPACT ON CHESS PERFORMANCE
# =============================================================================

# Load all countries - treated (Russia, Ukraine) and controls (others)
country_data <- list(
  "Germany" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/DE_daily_stats.csv")),
  "United Kingdom" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/GB_daily_stats.csv")),
  "France" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/FR_daily_stats.csv")),
  "United States" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/US_daily_stats.csv")),
  "India" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/IN_daily_stats.csv")),
  "Ukraine" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/UA_daily_stats.csv")),
  "Russia" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/RU_daily_stats.csv"))
)

# Convert dates and combine data
for(country in names(country_data)) {
  country_data[[country]]$date <- as.Date(country_data[[country]]$date)
}

all_data <- do.call(rbind, lapply(names(country_data), function(country) {
  data.frame(country = country, country_data[[country]])
}))

# Define key dates and treatment
invasion_date <- as.Date("2022-02-24")
covid_start <- as.Date("2020-03-01")

# Create treatment indicators
all_data$treated <- all_data$country %in% c("Russia", "Ukraine")
all_data$post_invasion <- ifelse(is.na(all_data$date), NA, all_data$date >= invasion_date)
all_data$post_covid <- ifelse(is.na(all_data$date), NA, all_data$date >= covid_start)

# Create time trend variables
all_data$days_since_start <- as.numeric(all_data$date - min(all_data$date, na.rm = TRUE))
all_data$days_since_2020 <- as.numeric(all_data$date - as.Date("2020-01-01"))

# Filter to analysis window (2020-2023 for cleaner analysis)
analysis_data <- all_data %>%
  filter(date >= as.Date("2020-01-01") & date <= as.Date("2023-12-31"),
         !is.na(post_invasion),
         rating_count > 100,  # Filter for meaningful sample sizes
         accuracy_count > 100) %>%
  mutate(
    # Log transformations for better model fit
    log_rating_count = log(rating_count + 1),
    log_accuracy_count = log(accuracy_count + 1),
    
    # Detrended metrics using 30-day rolling average
    rating_ma30 = zoo::rollmean(rating_mean, k = 30, fill = NA, align = "center"),
    accuracy_ma30 = zoo::rollmean(accuracy_mean, k = 30, fill = NA, align = "center")
  ) %>%
  group_by(country) %>%
  mutate(
    # Detrend by removing country-specific linear trends
    rating_detrended = rating_mean - predict(lm(rating_mean ~ days_since_2020, na.action = na.exclude)),
    accuracy_detrended = accuracy_mean - predict(lm(accuracy_mean ~ days_since_2020, na.action = na.exclude)),
    
    # Standardize within country (z-scores)
    rating_std_within = scale(rating_mean)[,1],
    accuracy_std_within = scale(accuracy_mean)[,1]
  ) %>%
  ungroup()

# =============================================================================
# DIFFERENCE-IN-DIFFERENCES REGRESSION MODELS
# =============================================================================

# Function to run DiD regression with multiple specifications
run_did_analysis <- function(data, outcome_var, include_controls = TRUE) {
  
  # Basic DiD model
  formula_basic <- as.formula(paste(outcome_var, "~ treated * post_invasion"))
  
  # With time trends
  formula_trends <- as.formula(paste(outcome_var, "~ treated * post_invasion + days_since_2020 + I(days_since_2020^2)"))
  
  # With COVID controls
  formula_covid <- as.formula(paste(outcome_var, "~ treated * post_invasion + post_covid + days_since_2020 + I(days_since_2020^2)"))
  
  # With full controls
  if(include_controls) {
    formula_full <- as.formula(paste(outcome_var, "~ treated * post_invasion + post_covid + days_since_2020 + I(days_since_2020^2) + log_rating_count + log_accuracy_count"))
  } else {
    formula_full <- formula_covid
  }
  
  # Run models
  model_basic <- lm(formula_basic, data = data)
  model_trends <- lm(formula_trends, data = data)
  model_covid <- lm(formula_covid, data = data)
  model_full <- lm(formula_full, data = data)
  
  # Extract DiD coefficients (treated:post_invasion interaction)
  results <- data.frame(
    outcome = outcome_var,
    model = c("Basic", "With Trends", "With COVID", "Full Controls"),
    did_coef = c(
      coef(model_basic)["treatedTRUE:post_invasionTRUE"],
      coef(model_trends)["treatedTRUE:post_invasionTRUE"],
      coef(model_covid)["treatedTRUE:post_invasionTRUE"],
      coef(model_full)["treatedTRUE:post_invasionTRUE"]
    ),
    se = c(
      summary(model_basic)$coefficients["treatedTRUE:post_invasionTRUE", "Std. Error"],
      summary(model_trends)$coefficients["treatedTRUE:post_invasionTRUE", "Std. Error"],
      summary(model_covid)$coefficients["treatedTRUE:post_invasionTRUE", "Std. Error"],
      summary(model_full)$coefficients["treatedTRUE:post_invasionTRUE", "Std. Error"]
    ),
    t_stat = c(
      summary(model_basic)$coefficients["treatedTRUE:post_invasionTRUE", "t value"],
      summary(model_trends)$coefficients["treatedTRUE:post_invasionTRUE", "t value"],
      summary(model_covid)$coefficients["treatedTRUE:post_invasionTRUE", "t value"],
      summary(model_full)$coefficients["treatedTRUE:post_invasionTRUE", "t value"]
    ),
    p_value = c(
      summary(model_basic)$coefficients["treatedTRUE:post_invasionTRUE", "Pr(>|t|)"],
      summary(model_trends)$coefficients["treatedTRUE:post_invasionTRUE", "Pr(>|t|)"],
      summary(model_covid)$coefficients["treatedTRUE:post_invasionTRUE", "Pr(>|t|)"],
      summary(model_full)$coefficients["treatedTRUE:post_invasionTRUE", "Pr(>|t|)"]
    )
  )
  
  results$significant <- results$p_value < 0.05
  results$ci_lower <- results$did_coef - 1.96 * results$se
  results$ci_upper <- results$did_coef + 1.96 * results$se
  
  return(list(results = results, models = list(basic = model_basic, trends = model_trends, 
                                               covid = model_covid, full = model_full)))
}

# Run DiD analysis for key outcomes
print("=== DIFFERENCE-IN-DIFFERENCES ANALYSIS ===")

# Raw metrics
rating_did <- run_did_analysis(analysis_data, "rating_mean")
accuracy_did <- run_did_analysis(analysis_data, "accuracy_mean")

# Detrended metrics
rating_detrended_did <- run_did_analysis(analysis_data, "rating_detrended", include_controls = FALSE)
accuracy_detrended_did <- run_did_analysis(analysis_data, "accuracy_detrended", include_controls = FALSE)

# Standardized metrics
rating_std_did <- run_did_analysis(analysis_data, "rating_std_within", include_controls = FALSE)
accuracy_std_did <- run_did_analysis(analysis_data, "accuracy_std_within", include_controls = FALSE)

# Variability metrics
rating_var_did <- run_did_analysis(analysis_data, "rating_std")
accuracy_var_did <- run_did_analysis(analysis_data, "accuracy_std")

# Combine all results
all_did_results <- rbind(
  rating_did$results,
  accuracy_did$results,
  rating_detrended_did$results,
  accuracy_detrended_did$results,
  rating_std_did$results,
  accuracy_std_did$results,
  rating_var_did$results,
  accuracy_var_did$results
)

print(all_did_results)

# =============================================================================
# SEPARATE ANALYSIS FOR RUSSIA VS UKRAINE
# =============================================================================

# Analyze Russia and Ukraine separately
russia_data <- analysis_data %>% filter(country == "Russia")
ukraine_data <- analysis_data %>% filter(country == "Ukraine")
control_data <- analysis_data %>% filter(!treated)

# Function for single country analysis
analyze_single_country <- function(country_data, control_data, country_name) {
  
  # Combine with controls
  combined_data <- rbind(
    country_data %>% mutate(treated_country = TRUE),
    control_data %>% mutate(treated_country = FALSE)
  )
  
  # Run DiD
  results <- list()
  
  for(outcome in c("rating_mean", "accuracy_mean", "rating_detrended", "accuracy_detrended")) {
    formula <- as.formula(paste(outcome, "~ treated_country * post_invasion + days_since_2020 + post_covid"))
    model <- lm(formula, data = combined_data)
    
    coef_name <- "treated_countryTRUE:post_invasionTRUE"
    if(coef_name %in% names(coef(model))) {
      results[[outcome]] <- data.frame(
        country = country_name,
        outcome = outcome,
        did_coef = coef(model)[coef_name],
        se = summary(model)$coefficients[coef_name, "Std. Error"],
        p_value = summary(model)$coefficients[coef_name, "Pr(>|t|)"],
        significant = summary(model)$coefficients[coef_name, "Pr(>|t|)"] < 0.05
      )
    }
  }
  
  return(do.call(rbind, results))
}

russia_results <- analyze_single_country(russia_data, control_data, "Russia")
ukraine_results <- analyze_single_country(ukraine_data, control_data, "Ukraine")

print("=== INDIVIDUAL COUNTRY ANALYSIS ===")
print("Russia vs Controls:")
print(russia_results)
print("Ukraine vs Controls:")
print(ukraine_results)

# =============================================================================
# ENHANCED VISUALIZATIONS
# =============================================================================

# Create visualization comparing treated vs control countries
viz_data <- analysis_data %>%
  mutate(
    group = ifelse(treated, "Russia + Ukraine", "Control Countries"),
    country_group = ifelse(country %in% c("Russia", "Ukraine"), country, "Control Average")
  )

# Aggregate control countries
control_avg <- viz_data %>%
  filter(!treated) %>%
  group_by(date) %>%
  summarise(
    rating_mean = mean(rating_mean, na.rm = TRUE),
    accuracy_mean = mean(accuracy_mean, na.rm = TRUE),
    rating_detrended = mean(rating_detrended, na.rm = TRUE),
    accuracy_detrended = mean(accuracy_detrended, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(country = "Control Average", treated = FALSE)

# Combine for plotting
plot_data <- rbind(
  viz_data %>% filter(treated) %>% select(date, country, rating_mean, accuracy_mean, rating_detrended, accuracy_detrended, treated),
  control_avg %>% select(date, country, rating_mean, accuracy_mean, rating_detrended, accuracy_detrended, treated)
)

# Colors
colors <- c("Russia" = "#8E44AD", "Ukraine" = "#E74C3C", "Control Average" = "#2ECC71")

# Rating comparison plot
plot_rating_did <- ggplot(plot_data, aes(x = date, y = rating_mean, color = country)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_smooth(method = "loess", span = 0.3, se = TRUE, alpha = 0.2) +
  scale_color_manual(values = colors) +
  labs(
    title = "Chess Rating: Difference-in-Differences Analysis",
    subtitle = "Comparing Russia/Ukraine vs Control Countries",
    x = "Date",
    y = "Mean Rating",
    color = "Country/Group"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "bottom"
  ) +
  geom_vline(xintercept = invasion_date, color = "red", linetype = "dashed", size = 1) +
  geom_text(aes(x = invasion_date, y = max(rating_mean, na.rm = TRUE), label = "Invasion"), 
            vjust = -0.5, hjust = 0, color = "red", fontface = "bold", inherit.aes = FALSE) +
  coord_cartesian(xlim = as.Date(c("2020-01-01", "2023-12-31")))

# Detrended rating plot
plot_rating_detrended <- ggplot(plot_data, aes(x = date, y = rating_detrended, color = country)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_smooth(method = "loess", span = 0.3, se = TRUE, alpha = 0.2) +
  scale_color_manual(values = colors) +
  labs(
    title = "Chess Rating (Detrended): Difference-in-Differences Analysis",
    subtitle = "Removing country-specific time trends",
    x = "Date",
    y = "Detrended Mean Rating",
    color = "Country/Group"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "bottom"
  ) +
  geom_vline(xintercept = invasion_date, color = "red", linetype = "dashed", size = 1) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dotted") +
  coord_cartesian(xlim = as.Date(c("2020-01-01", "2023-12-31")))

# Accuracy comparison
plot_accuracy_did <- ggplot(plot_data, aes(x = date, y = accuracy_mean, color = country)) +
  geom_line(size = 1, alpha = 0.8) +
  geom_smooth(method = "loess", span = 0.3, se = TRUE, alpha = 0.2) +
  scale_color_manual(values = colors) +
  labs(
    title = "Chess Accuracy: Difference-in-Differences Analysis",
    subtitle = "Comparing Russia/Ukraine vs Control Countries",
    x = "Date",
    y = "Mean Accuracy",
    color = "Country/Group"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "bottom"
  ) +
  geom_vline(xintercept = invasion_date, color = "red", linetype = "dashed", size = 1) +
  coord_cartesian(xlim = as.Date(c("2020-01-01", "2023-12-31")))

# Display plots
print(plot_rating_did)
print(plot_rating_detrended)
print(plot_accuracy_did)

# Combined plot
combined_did_plot <- plot_grid(plot_rating_did, plot_rating_detrended, plot_accuracy_did, ncol = 1)
print(combined_did_plot)

# =============================================================================
# SUMMARY AND INTERPRETATION
# =============================================================================

print("=== ANALYSIS SUMMARY ===")
print("Key Findings:")

# Focus on the full control model results
key_results <- all_did_results %>% 
  filter(model == "Full Controls") %>%
  select(outcome, did_coef, se, p_value, significant, ci_lower, ci_upper)

print(key_results)

print("\nInterpretation Guide:")
print("- Negative coefficients indicate performance DECREASED after invasion")
print("- Positive coefficients indicate performance INCREASED after invasion") 
print("- Significant results (p < 0.05) suggest the invasion had a measurable impact")
print("- Detrended results account for underlying time trends")
print("- Results compare Russia+Ukraine vs control countries (Germany, UK, France, US, India)")

# Calculate effect sizes
effect_summary <- key_results %>%
  mutate(
    effect_size = abs(did_coef),
    direction = ifelse(did_coef > 0, "Positive", "Negative"),
    magnitude = case_when(
      effect_size < 0.1 ~ "Small",
      effect_size < 0.5 ~ "Medium", 
      TRUE ~ "Large"
    )
  ) %>%
  select(outcome, did_coef, direction, magnitude, significant)

print("\nEffect Size Summary:")
print(effect_summary)
