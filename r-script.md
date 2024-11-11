```markdown
---
title: "SmartFlix: Data-Driven Movie Recommendation System using R"
editor_options: 
  markdown: 
    wrap: 72
---
```
# 1. Title of the Project

Project Title: SmartFlix: Data-Driven Movie Recommendation System using R
      #Github: https://github.com/kawas8516/movie-recommend-r

# 2. Problem Identification

In the entertainment industry, personalized recommendations enhance user engagement and satisfaction by aligning suggested content with user preferences. The SmartFlix project aims to develop a movie recommendation system using R, addressing the challenge of recommending movies based on user preferences, viewing history, and genre selection.

# 3. Dataset Preparation

The dataset includes user movie ratings, genres, and viewing history, sourced from publicly available movie databases such as MovieLens or IMDb. Data Preparation involves:

- Data Cleaning: Removing duplicates, handling missing values, and ensuring consistent data types.
- Normalization: Standardizing ratings to improve model efficiency.

```r
# Install necessary packages if not already installed
# install.packages(c("dplyr", "tidyr", "ggplot2", "caret", "janitor", "purrr"))

# Load required libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(janitor)
library(readr)
library(purrr)

# Load data
file_path <- "C:/Users/kaust/Envs/SmartFlix/data/movie_reviews_clean.csv"
movies <- read_csv(file_path)

# Clean column names to avoid issues (e.g., spaces, case sensitivity)
movies <- clean_names(movies)

# Basic data manipulation - select relevant fields and handle missing data
movies_clean <- movies %>%
  select(movie_title, genres, tomatometer_rating, audience_rating, critics_consensus, content_rating, actors, runtime) %>%
  filter(!is.na(tomatometer_rating) & !is.na(audience_rating))

# Convert genres into individual rows and create dummy variables
movies_clean <- movies_clean %>%
  separate_rows(genres, sep = ",") %>%
  mutate(genre = trimws(tolower(genres))) %>%  # Standardize genre names (lowercase and remove whitespace)
  filter(genre != "" & !is.na(genre)) %>%      # Remove empty or NA genres
  mutate(genre_dummy = 1) %>%
  pivot_wider(names_from = genre, values_from = genre_dummy, values_fill = 0)
```

# 4. Exploratory Data Analysis (EDA)

EDA is conducted using ggplot2 to understand the data's structure and uncover patterns:

- Visualization of Rating Distributions: Plotting histograms and genre-wise average ratings.
- User Analysis: Examining user engagement and trends in movie ratings.

```r
# Distribution of tomatometer ratings
ggplot(movies_clean, aes(x = tomatometer_rating)) +
  geom_histogram(binwidth = 5, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Tomatometer Ratings", x = "Tomatometer Rating", y = "Count")

# Relationship between audience rating and tomatometer rating
ggplot(movies_clean, aes(x = audience_rating, y = tomatometer_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", col = "red") +
  labs(title = "Audience Rating vs Tomatometer Rating", x = "Audience Rating", y = "Tomatometer Rating")
```

# 5. Statistical Analysis

This phase uses the `stats` package to perform statistical tests:

- Correlation Analysis: Checking relationships between variables like genre popularity and rating scores.

```r
# Calculate correlation between audience rating and tomatometer rating
correlation <- cor(movies_clean$audience_rating, movies_clean$tomatometer_rating, use = "complete.obs")
print(paste("Correlation between Audience Rating and Tomatometer Rating:", round(correlation, 2)))
```

# 6. Functional Programming and Clustering

Using `purrr` and `caret`, the project leverages functional programming and machine learning for clustering movies based on features:

```r
# Prepare data for machine learning
set.seed(123)

# Select relevant features for clustering
features <- movies_clean %>%
  select(tomatometer_rating, audience_rating, runtime, starts_with("genre_"))

# Normalize the features
preproc <- preProcess(features, method = c("center", "scale"))
features_normalized <- predict(preproc, features)

# Use k-means clustering to create movie clusters
set.seed(123)
k_clusters <- kmeans(features_normalized, centers = 5, nstart = 20)

# Add cluster information to the original data
movies_clean$cluster <- k_clusters$cluster

# Print a summary of movies in each cluster
cluster_summary <- movies_clean %>%
  group_by(cluster) %>%
  summarize(
    avg_tomatometer = mean(tomatometer_rating, na.rm = TRUE),
    avg_audience = mean(audience_rating, na.rm = TRUE),
    count = n()
  )
print(cluster_summary)

# Function to recommend movies from the same cluster
recommend_movies <- function(movie_title, data, n_recommendations = 5) {
  cluster_num <- data %>%
    filter(movie_title == !!movie_title) %>%
    pull(cluster)
  
  recommendations <- data %>%
    filter(cluster == cluster_num, movie_title != movie_title) %>%
    select(movie_title, tomatometer_rating, audience_rating) %>%
    head(n_recommendations)
  
  return(recommendations)
}

# Example: Recommend movies similar to a given title
print(recommend_movies("Inception", movies_clean))
```

# 7. Conclusion

The SmartFlix recommendation system successfully integrates user data and machine learning to deliver personalized movie recommendations. This modular R-based solution demonstrates the effectiveness of data cleaning, EDA, statistical testing, and functional programming in building a scalable recommendation system.