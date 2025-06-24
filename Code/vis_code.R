if (!require("pacman")) install.packages("pacman")

pacman::p_load(devtools,np,lazyeval, hmisc,interp, lmtest,gt, modelsummary, dplyr,pdftools, tidyverse,rattle,glmnet,caret, rpart.plot, RcolorBrewer,rpart, tidyr, mice, stringr,randomForest,  curl, plm, readxl, zoo, stringr, patchwork,  sf, clubSandwich, modelsummary, sjPlot)

# =============================================================================
# COUNTRY SELECTION - MODIFY THIS SECTION TO CHOOSE COUNTRIES
# =============================================================================

# Available countries: "Germany", "United Kingdom", "France", "United States", "India", "Ukraine", "Russia"
#SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France", "United States", "India")

# Alternative examples:
 #SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "Canada" ,"France")  # Just Europe
# SELECTED_COUNTRIES <- c("United States", "India", "Russia", "Ukraine")   # US + Eastern Europe
 SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France", "United States", "India", "Ukraine", "Russia", "Brazil", "Canada")  # All countries

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
  "Russia" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/RU_daily_stats.csv")),
  "Canada" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/CA_daily_stats.csv")),
  "Brazil" = read.csv(curl("https://raw.githubusercontent.com/BenGoodair/chess_performance/refs/heads/main/Data/processed/BR_daily_stats.csv"))
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
# Define all lockdown periods including Canada and Brazil
all_lockdowns <- data.frame(
  country = c(
    "United Kingdom", "United Kingdom", "United Kingdom",
    "France", "France", "France",
    "Germany", "Germany",
    "United States", "United States",
    "India", "India",
    "Ukraine", "Ukraine",
    "Russia", "Russia", "Russia",
    "Canada", "Canada", "Canada",
    "Brazil", "Brazil"
  ),
  start = as.Date(c(
    "2020-03-23", "2020-11-05", "2021-01-06",
    "2020-03-17", "2020-10-30", "2021-04-03",
    "2020-03-22", "2020-11-02",
    "2020-03-19", "2020-11-01",
    "2020-03-25", "2021-04-28",
    "2020-03-17", "2020-10-08",
    "2020-03-30", "2020-10-05", "2021-06-13",
    "2020-03-16", "2020-12-26", "2021-04-03",
    "2020-03-16", "2021-01-10"
  )),
  end = as.Date(c(
    "2020-07-04", "2020-12-02", "2021-07-19",
    "2020-05-11", "2020-12-15", "2021-05-03",
    "2020-05-06", "2021-03-07",
    "2020-05-15", "2020-12-15",
    "2020-05-31", "2021-06-30",
    "2020-05-11", "2021-01-24",
    "2020-05-12", "2020-12-08", "2021-07-11",
    "2020-05-15", "2021-03-08", "2021-06-15",
    "2020-05-31", "2021-03-31"
  )),
  lockdown_num = c(1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2)
)

# Display the updated dataset

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
  "Russia" = "#8E44AD",
  "Canada" = "#3498DB",  # A strong blue, works well with the existing tones
  "Brazil" = "#2ECC71"   # A vibrant green, complements the warm tones nicely
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










#### heatwaves####
# =============================================================================
# CHESS PERFORMANCE vs HEATWAVE ANALYSIS - IMPROVED VERSION
# =============================================================================

# Load required packages with better error handling
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  httr, jsonlite, lubridate, dplyr, ggplot2, corrplot, 
  stargazer, viridis, gridExtra, tidyr, broom
)

# Set up global options
options(scipen = 999)  # Disable scientific notation
set.seed(42)  # For reproducibility

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Define countries (assuming these are defined elsewhere in your environment)
if (!exists("SELECTED_COUNTRIES")) {
  SELECTED_COUNTRIES <- c("Germany", "United Kingdom", "France", "United States", "India")
}

# Define capital city coordinates for weather data
city_coords <- data.frame(
  country = c("Germany", "United Kingdom", "France", "United States", 
              "India", "Ukraine", "Russia"),
  city = c("Berlin", "London", "Paris", "Washington DC", 
           "New Delhi", "Kyiv", "Moscow"),
  latitude = c(52.52, 51.51, 48.86, 38.91, 28.61, 50.45, 55.76),
  longitude = c(13.41, -0.13, 2.35, -77.04, 77.21, 30.52, 37.62),
  stringsAsFactors = FALSE
)

# =============================================================================
# WEATHER DATA FUNCTIONS
# =============================================================================

# Improved function to get weather data with better error handling
get_weather_data <- function(latitude, longitude, start_date, end_date, country_name) {
  
  # Convert dates to proper format
  start_str <- format(as.Date(start_date), "%Y-%m-%d")
  end_str <- format(as.Date(end_date), "%Y-%m-%d")
  
  # Construct API URL
  url <- paste0(
    "https://archive-api.open-meteo.com/v1/archive?",
    "latitude=", latitude,
    "&longitude=", longitude,
    "&start_date=", start_str,
    "&end_date=", end_str,
    "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean",
    "&timezone=auto"
  )
  
  cat("Fetching weather data for", country_name, "...\n")
  
  # Make API request with error handling and retry logic
  max_retries <- 3
  retry_delay <- 2
  
  for (attempt in 1:max_retries) {
    tryCatch({
      response <- GET(url, timeout(30))
      
      if (status_code(response) == 200) {
        data <- fromJSON(content(response, "text"))
        
        # Check if data structure is valid
        if (is.null(data$daily) || is.null(data$daily$time)) {
          cat("Invalid data structure for", country_name, "\n")
          return(NULL)
        }
        
        # Extract daily data
        weather_df <- data.frame(
          country = country_name,
          date = as.Date(data$daily$time),
          temp_max = as.numeric(data$daily$temperature_2m_max),
          temp_min = as.numeric(data$daily$temperature_2m_min), 
          temp_mean = as.numeric(data$daily$temperature_2m_mean),
          stringsAsFactors = FALSE
        )
        
        # Remove rows with all NA temperatures
        weather_df <- weather_df[!is.na(weather_df$temp_max) | 
                                   !is.na(weather_df$temp_min) | 
                                   !is.na(weather_df$temp_mean), ]
        
        cat("Successfully fetched", nrow(weather_df), "days of data for", country_name, "\n")
        return(weather_df)
        
      } else if (status_code(response) == 429) {
        cat("Rate limit exceeded for", country_name, ". Retrying in", retry_delay * attempt, "seconds...\n")
        Sys.sleep(retry_delay * attempt)
        next
      } else {
        cat("API request failed for", country_name, "with status:", status_code(response), "\n")
        if (attempt == max_retries) return(NULL)
        Sys.sleep(retry_delay)
      }
      
    }, error = function(e) {
      cat("Error fetching weather data for", country_name, ":", e$message, "\n")
      if (attempt == max_retries) return(NULL)
      Sys.sleep(retry_delay)
    })
  }
  
  return(NULL)
}

# =============================================================================
# HEATWAVE IDENTIFICATION FUNCTIONS
# =============================================================================

# Improved heatwave identification function
identify_heatwaves <- function(weather_df, temp_column = "temp_max", 
                               percentile_threshold = 0.95, min_duration = 3) {
  
  # Check if we have enough data
  if (nrow(weather_df) < 30 || sum(!is.na(weather_df[[temp_column]])) < 30) {
    cat("Insufficient temperature data for heatwave analysis\n")
    return(weather_df)
  }
  
  # Sort by date
  weather_df <- weather_df[order(weather_df$date), ]
  
  # Calculate threshold temperature (95th percentile)
  threshold_temp <- quantile(weather_df[[temp_column]], percentile_threshold, na.rm = TRUE)
  
  # Identify hot days
  weather_df$is_hot_day <- ifelse(is.na(weather_df[[temp_column]]), FALSE,
                                  weather_df[[temp_column]] > threshold_temp)
  
  # Identify consecutive hot days using run-length encoding
  rle_hot <- rle(weather_df$is_hot_day)
  
  # Create group identifiers
  weather_df$hot_day_group <- rep(seq_along(rle_hot$lengths), rle_hot$lengths)
  
  # Identify heatwave groups (consecutive hot days >= min_duration)
  heatwave_groups <- c()
  group_start <- 1
  
  for (i in seq_along(rle_hot$lengths)) {
    if (rle_hot$values[i] && rle_hot$lengths[i] >= min_duration) {
      heatwave_groups <- c(heatwave_groups, i)
    }
  }
  
  # Mark heatwave days
  weather_df$is_heatwave <- weather_df$hot_day_group %in% heatwave_groups & weather_df$is_hot_day
  
  # Create heatwave intensity measure
  weather_df$heatwave_intensity <- ifelse(weather_df$is_heatwave & !is.na(weather_df[[temp_column]]), 
                                          pmax(0, weather_df[[temp_column]] - threshold_temp), 0)
  
  # Add threshold temperature for reference
  weather_df$temp_threshold <- threshold_temp
  
  # Clean up temporary columns
  weather_df$is_hot_day <- NULL
  weather_df$hot_day_group <- NULL
  
  return(weather_df)
}

# =============================================================================
# DATA FETCHING AND PROCESSING
# =============================================================================

# Create sample chess data if all_data doesn't exist
if (!exists("all_data")) {
  cat("Creating sample chess data for demonstration...\n")
  
  # Generate sample data for demonstration
  start_date <- as.Date("2020-01-01")
  end_date <- as.Date("2023-12-31")
  dates <- seq(start_date, end_date, by = "day")
  
  all_data <- data.frame()
  for (country in SELECTED_COUNTRIES) {
    country_data <- data.frame(
      country = country,
      date = dates,
      rating_mean = round(rnorm(length(dates), mean = 1400, sd = 200)),
      accuracy_mean = round(runif(length(dates), min = 0.25, max = 0.45), 3),
      rating_std = round(rnorm(length(dates), mean = 180, sd = 30)),
      accuracy_std = round(runif(length(dates), min = 0.03, max = 0.06), 3)
    )
    all_data <- rbind(all_data, country_data)
  }
  
  cat("Sample chess data created with", nrow(all_data), "observations\n")
}

# Filter coordinates for selected countries
selected_coords <- city_coords[city_coords$country %in% SELECTED_COUNTRIES, ]

# Determine date range from chess data
chess_date_range <- range(all_data$date, na.rm = TRUE)
start_date <- chess_date_range[1]
end_date <- chess_date_range[2]

cat("Fetching weather data from", as.character(start_date), "to", as.character(end_date), "\n")

# Fetch weather data for all selected countries
weather_data_list <- list()
for(i in 1:nrow(selected_coords)) {
  country <- selected_coords$country[i]
  lat <- selected_coords$latitude[i]
  lon <- selected_coords$longitude[i]
  
  weather_data_list[[country]] <- get_weather_data(lat, lon, start_date, end_date, country)
  
  # Add delay to be respectful to the API
  Sys.sleep(2)
}

# Combine all weather data
weather_data <- do.call(rbind, weather_data_list[!sapply(weather_data_list, is.null)])

# Check if we have weather data
if (nrow(weather_data) == 0) {
  cat("No weather data available. Creating synthetic weather data for demonstration...\n")
  
  # Create synthetic weather data
  weather_data <- data.frame()
  for (country in SELECTED_COUNTRIES) {
    country_weather <- data.frame(
      country = country,
      date = seq(start_date, end_date, by = "day"),
      temp_max = round(rnorm(as.numeric(end_date - start_date + 1), mean = 20, sd = 8), 1),
      temp_min = round(rnorm(as.numeric(end_date - start_date + 1), mean = 10, sd = 6), 1),
      temp_mean = round(rnorm(as.numeric(end_date - start_date + 1), mean = 15, sd = 7), 1)
    )
    weather_data <- rbind(weather_data, country_weather)
  }
}

# Apply heatwave identification to each country
weather_with_heatwaves <- data.frame()
for(country in unique(weather_data$country)) {
  country_weather <- weather_data[weather_data$country == country, ]
  country_weather_hw <- identify_heatwaves(country_weather)
  weather_with_heatwaves <- rbind(weather_with_heatwaves, country_weather_hw)
}

# =============================================================================
# MERGE AND CREATE LAGGED VARIABLES
# =============================================================================

# Merge chess performance data with weather data
merged_data <- merge(all_data, weather_with_heatwaves, by = c("country", "date"), all.x = TRUE)%>%
  dplyr::filter(accuracy_mean!=0)

# Create lagged variables function
create_country_lags <- function(df, var_name, lags = 1:7) {
  df_with_lags <- df
  
  for(country in unique(df$country)) {
    country_mask <- df$country == country
    country_data <- df[country_mask, ]
    country_data <- country_data[order(country_data$date), ]
    
    for(lag in lags) {
      lag_col_name <- paste0(var_name, "_lag", lag)
      if (lag <= nrow(country_data)) {
        lagged_values <- c(rep(NA, lag), head(country_data[[var_name]], -lag))
        df_with_lags[country_mask, lag_col_name] <- lagged_values[order(order(country_data$date))]
      }
    }
  }
  
  return(df_with_lags)
}

# Sort data before creating lags
merged_data <- merged_data[order(merged_data$country, merged_data$date), ]

# Create lagged heatwave variables
merged_data <- create_country_lags(merged_data, "is_heatwave", lags = 1:7)
merged_data <- create_country_lags(merged_data, "heatwave_intensity", lags = 1:3)
merged_data <- create_country_lags(merged_data, "temp_max", lags = 1:3)

# Create aggregated heatwave measures
heatwave_cols <- c("is_heatwave", "is_heatwave_lag1", "is_heatwave_lag2")
heatwave_cols <- heatwave_cols[heatwave_cols %in% names(merged_data)]
merged_data$heatwave_last_3days <- rowSums(merged_data[, heatwave_cols, drop = FALSE], na.rm = TRUE)

heatwave_7day_cols <- c("is_heatwave", paste0("is_heatwave_lag", 1:6))
heatwave_7day_cols <- heatwave_7day_cols[heatwave_7day_cols %in% names(merged_data)]
merged_data$heatwave_last_7days <- rowSums(merged_data[, heatwave_7day_cols, drop = FALSE], na.rm = TRUE)

# Create time variables
merged_data$month <- month(merged_data$date)
merged_data$year <- year(merged_data$date)
merged_data$summer_months <- merged_data$month %in% c(6, 7, 8)

# =============================================================================
# DESCRIPTIVE ANALYSIS
# =============================================================================

cat("\n=== HEATWAVE SUMMARY STATISTICS ===\n")

# Heatwave summary by country
heatwave_summary <- merged_data %>%
  group_by(country) %>%
  summarise(
    total_days = n(),
    heatwave_days = sum(is_heatwave %in% TRUE, na.rm = TRUE),
    heatwave_percentage = round(100 * heatwave_days / total_days, 2),
    avg_temp_max = round(mean(temp_max, na.rm = TRUE), 1),
    temp_threshold_95th = round(mean(temp_threshold, na.rm = TRUE), 1),
    max_heatwave_intensity = round(ifelse(heatwave_days > 0, 
                                          max(heatwave_intensity, na.rm = TRUE), 0), 1),
    .groups = 'drop'
  )

print(heatwave_summary)

# Chess performance comparison
cat("\n=== CHESS PERFORMANCE: HEATWAVE vs NORMAL DAYS ===\n")

performance_comparison <- merged_data %>%
  filter(!is.na(rating_mean) & !is.na(is_heatwave)) %>%
  group_by(country, is_heatwave) %>%
  summarise(
    n_days = n(),
    avg_rating = round(mean(rating_mean, na.rm = TRUE), 1),
    avg_accuracy = round(mean(accuracy_mean, na.rm = TRUE), 3),
    rating_std = round(mean(rating_std, na.rm = TRUE), 1),
    accuracy_std = round(mean(accuracy_std, na.rm = TRUE), 3),
    .groups = 'drop'
  ) %>%
  mutate(condition = ifelse(is_heatwave, "Heatwave", "Normal"))

print(performance_comparison)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

# Improved analysis function
analyze_heatwave_effects <- function(country_name) {
  
  cat("\n=== ANALYSIS FOR", toupper(country_name), "===\n")
  
  # Filter data for the country
  country_data <- merged_data %>%
    filter(country == country_name, 
           !is.na(rating_mean), 
           !is.na(is_heatwave),
           !is.na(temp_max))
  
  if(nrow(country_data) < 100) {
    cat("Insufficient data for", country_name, "(", nrow(country_data), "observations)\n")
    return(NULL)
  }
  
  # Check if we have any heatwave days
  heatwave_days <- sum(country_data$is_heatwave, na.rm = TRUE)
  if(heatwave_days < 5) {
    cat("Insufficient heatwave days for", country_name, "(", heatwave_days, "days)\n")
    return(NULL)
  }
  
  # Model 1: Current day heatwave effect
  model1 <- lm(rating_mean ~ is_heatwave + month + year, data = country_data)
  
  # Model 2: Temperature effects (continuous)
  model2 <- lm(rating_mean ~ temp_max + I(temp_max^2) + month + year, data = country_data)
  
  # Model 3: Heatwave intensity
  if("heatwave_intensity" %in% names(country_data)) {
    model3 <- lm(rating_mean ~ heatwave_intensity + month + year, data = country_data)
  } else {
    model3 <- NULL
  }
  
  # Accuracy models
  acc_model1 <- lm(accuracy_mean ~ is_heatwave + month + year, data = country_data)
  
  # Store results
  results <- list(
    country = country_name,
    n_obs = nrow(country_data),
    heatwave_days = heatwave_days,
    rating_models = list(heatwave = model1, temperature = model2, intensity = model3),
    accuracy_models = list(heatwave = acc_model1),
    data = country_data
  )
  
  # Print key results
  cat("Sample size:", nrow(country_data), "days\n")
  cat("Heatwave days:", heatwave_days, "\n")
  
  # Extract and display key coefficients
  if("is_heatwaveTRUE" %in% names(coef(model1))) {
    heatwave_coef <- coef(model1)["is_heatwaveTRUE"]
    heatwave_p <- summary(model1)$coefficients["is_heatwaveTRUE", "Pr(>|t|)"]
    cat("Rating effect (heatwave vs normal):", round(heatwave_coef, 2), "\n")
    cat("P-value:", round(heatwave_p, 4), "\n")
  }
  
  return(results)
}

# Run analysis for countries with data
analysis_results <- list()
for(country in unique(merged_data$country)) {
  if(country %in% SELECTED_COUNTRIES) {
    analysis_results[[country]] <- analyze_heatwave_effects(country)
  }
}

# Remove NULL results
analysis_results <- analysis_results[!sapply(analysis_results, is.null)]

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create visualizations with better error handling
create_visualizations <- function() {
  
  # Filter data for plotting
  plot_data <- merged_data %>%
    filter(!is.na(temp_max), !is.na(accuracy_mean), !is.na(is_heatwave))
  
  if(nrow(plot_data) == 0) {
    cat("No data available for visualization\n")
    return(NULL)
  }
  
  # Plot 1: Time series with dual y-axis
  p1 <- ggplot(plot_data, aes(x = date)) +
    geom_line(aes(y = accuracy_mean, color = "Chess accuracy"), alpha = 0.7, size = 0.5) +
    geom_point(data = filter(plot_data, is_heatwave == TRUE),
               aes(y = accuracy_mean), color = "red", size = 1, alpha = 0.8) +
    facet_wrap(~country, scales = "free") +
    scale_color_manual(values = c("Chess accuracy" = "blue")) +
    labs(
      title = "Chess Performance Over Time",
      subtitle = "Red dots indicate heatwave days",
      x = "Date",
      y = "Chess accuracy",
      color = "Variable"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Plot 2: Box plot comparison
  p2 <- plot_data %>%
    ggplot(aes(x = factor(is_heatwave, labels = c("Normal Days", "Heatwave Days")), 
               y = accuracy_mean, fill = factor(is_heatwave))) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.5) +
    facet_wrap(~country, scales = "free_y") +
    scale_fill_manual(values = c("FALSE" = "lightblue", "TRUE" = "red"), guide = "none") +
    labs(
      title = "Chess accuracy Distribution: Normal vs Heatwave Days",
      x = "Condition",
      y = "Chess accuracy"
    ) +
    theme_minimal()
  
  # Plot 3: Temperature vs Performance scatter
  p3 <- plot_data %>%
    ggplot(aes(x = temp_max, y = accuracy_mean, color = is_heatwave)) +
    geom_point(alpha = 0.6, size = 0.8) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.3) +
    facet_wrap(~country, scales = "free") +
    scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                       labels = c("Normal", "Heatwave")) +
    labs(
      title = "Chess Performance vs Maximum Temperature",
      x = "Maximum Temperature (°C)",
      y = "Chess accuracy",
      color = "Day Type"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  return(list(timeseries = p1, boxplot = p2, scatter = p3))
}

# Create plots
plots <- create_visualizations()

# Display plots if available
if(!is.null(plots)) {
  print(plots$timeseries)
  print(plots$boxplot)
  print(plots$scatter)
}

# =============================================================================
# SUMMARY REPORT
# =============================================================================

cat("\n=== OVERALL HEATWAVE IMPACT SUMMARY ===\n")

# Extract key coefficients from all country analyses
if(length(analysis_results) > 0) {
  summary_results <- data.frame()
  
  for(country in names(analysis_results)) {
    result <- analysis_results[[country]]
    
    if(!is.null(result$rating_models$heatwave)) {
      model_summary <- summary(result$rating_models$heatwave)
      
      if("is_heatwaveTRUE" %in% rownames(model_summary$coefficients)) {
        heatwave_coef <- model_summary$coefficients["is_heatwaveTRUE", ]
        
        summary_results <- rbind(summary_results, data.frame(
          country = country,
          heatwave_effect = round(heatwave_coef["Estimate"], 2),
          std_error = round(heatwave_coef["Std. Error"], 2),
          p_value = round(heatwave_coef["Pr(>|t|)"], 4),
          significant = heatwave_coef["Pr(>|t|)"] < 0.05,
          n_observations = result$n_obs,
          heatwave_days = result$heatwave_days
        ))
      }
    }
  }
  
  if(nrow(summary_results) > 0) {
    print(summary_results)
    
    # Overall conclusions
    cat("\n=== KEY FINDINGS ===\n")
    if(any(summary_results$significant)) {
      sig_countries <- summary_results$country[summary_results$significant]
      cat("Significant heatwave effects found in:", paste(sig_countries, collapse = ", "), "\n")
      
      avg_effect <- mean(summary_results$heatwave_effect[summary_results$significant])
      cat("Average rating change during heatwaves:", round(avg_effect, 2), "points\n")
    } else {
      cat("No statistically significant heatwave effects detected\n")
    }
  } else {
    cat("No valid analysis results available\n")
  }
} else {
  cat("No countries had sufficient data for analysis\n")
}

cat("\n=== ANALYSIS COMPLETE ===\n")











