##########################################################
### Part 1. Win Prediction for Baseball
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Lahman)) install.packages("Lahman", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(Lahman)
library(randomForest)
library(ggpubr)
library(dplyr)
library(plotly)

################### DATA PREPROCESSING ########################

# Group 'Teams' data by 30 years
Teams <- Teams %>% mutate(group = case_when(
  yearID %in% c(1871:1900) ~ "1871~1900",
  yearID %in% c(1901:1930) ~ "1901~1930",
  yearID %in% c(1931:1960) ~ "1931~1960",
  yearID %in% c(1961:1990) ~ "1961~1990",
  yearID %in% c(1991:2019) ~ "1991~2019"
))

# Add columns for X1B (single), X123BA (Non-HR hits allowed), and SH (sacrifice hits)
SH_table <- Batting %>% 
  group_by(teamID, yearID) %>%
  summarize(SH = sum(SH)) %>%
  filter(!is.na(SH))

Teams <- Teams %>% 
  filter(yearID %in% c(1871:2019)) %>%
  mutate(X1B = H - HR - X2B - X3B,
         X123BA = HA - HRA) %>%
  left_join(SH_table, by = c("teamID", "yearID")) %>%
  select(yearID, W, R, X1B, X2B, X3B, HR, BB, SO, SB, HBP, SH, RA, X123BA, HRA, BBA, SOA, E, group)

# To see if there is anything noticeable, maybe outliers?
p_outliers <- Teams %>% 
  filter(yearID %in% c(1871:2019)) %>% 
  ggplot(aes(X1B, W, color = group)) + geom_point() +
  ggtitle("Singles(X1B) vs. Wins(W)")

# Exclude years 1871~1900 from dataset and from year group
Teams <- Teams %>%
  filter(yearID %in% c(1901:2019)) %>%
  filter(!is.na(SO))  # filter out 16 NA entries for 'SO'

# Correlation between R - RA and W
p_rdiff <- Teams %>%
  ggplot(aes(R-RA, W)) + 
  geom_point() + 
  ggtitle("Run differential vs. Win")

# Correlation between variables and W
p <- Teams %>% 
  ggplot(aes(X1B, W, color = group)) + geom_point() + theme(legend.title = element_blank())

legend_group <- as_ggplot(get_legend(p))

p1 <- Teams %>% 
  ggplot(aes(X1B, W, color = group)) + geom_point(show.legend = FALSE)

p2 <- Teams %>% 
  ggplot(aes(X2B, W, color = group)) + geom_point(show.legend = FALSE)

p3 <- Teams %>% 
  ggplot(aes(X3B, W, color = group)) + geom_point(show.legend = FALSE)

p4 <- Teams %>% 
  ggplot(aes(HR, W, color = group)) + geom_point(show.legend = FALSE)

p5 <- Teams %>% 
  ggplot(aes(BB, W, color = group)) + geom_point(show.legend = FALSE)

p6 <- Teams %>% 
  ggplot(aes(SO, W, color = group)) + geom_point(show.legend = FALSE)
 
p7 <- Teams %>% 
  ggplot(aes(SB, W, color = group)) + geom_point(show.legend = FALSE)

p8 <- Teams %>% 
  ggplot(aes(SH, W, color = group)) + geom_point(show.legend = FALSE)

p9 <- Teams %>% 
  ggplot(aes(HBP, W, color = group)) + geom_point(show.legend = FALSE)

p10 <- Teams %>% 
  ggplot(aes(X123BA, W, color = group)) + geom_point(show.legend = FALSE)

p11 <- Teams %>% 
  ggplot(aes(HRA, W, color = group)) + geom_point(show.legend = FALSE)

p12 <- Teams %>% 
  ggplot(aes(BBA, W, color = group)) + geom_point(show.legend = FALSE)

p13 <- Teams %>% 
  ggplot(aes(SOA, W, color = group)) + geom_point(show.legend = FALSE)

p14 <- Teams %>% 
  ggplot(aes(E, W, color = group)) + geom_point(show.legend = FALSE)

p_collect_1 <- ggarrange(legend_group,p1,p2,p3,p4,p5, nrow = 3, ncol = 2)
p_collect_2 <- ggarrange(p6,p7,p8,p9,p10,p11, nrow = 3, ncol = 2)
p_collect_3 <- ggarrange(legend_group,p12,p13,p14, nrow = 3, ncol = 2)

p_collect_1
p_collect_2
p_collect_3

# What happened in 1981/1994? 713 games canceled in 1981, 948 games canceled in 1994-1995
p_1981 <- Teams %>% filter(yearID %in% c(1981:2000)) %>%
  ggplot(aes(X1B,W)) +
  geom_point(aes(color = ifelse(yearID %in% c(1981,1994), 'red', 'black'))) + 
  scale_color_identity() +
  labs(title = "What happened in 1981 and 1994-95?", x = "Singles (1B)", y = "Wins (W)")

# Final 'Teams' dataset
Teams <- Teams %>% select(-HBP)

######################### Model Building ###############################

# Create prediction set and validation set
set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = Teams$W, times = 1, p = 0.3, list = FALSE)
prediction_set <- Teams[-validation_index,]
validation_set <- Teams[validation_index,]

# Create train set and test set
set.seed(25, sample.kind="Rounding")
test_index <- createDataPartition(y = prediction_set$W, times = 1, p = 0.3, list = FALSE)
train_set <- prediction_set[-test_index,]
test_set <- prediction_set[test_index,]

# Model using only batting stats
fit <- lm(W ~ X1B + X2B + X3B + HR + BB + SO + SB + SH, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_0 <- sqrt(mean((y_hat - test_set$W)^2)) #> 10.7529
summary_0 <- summary(fit)

# Model using all variables
fit <- lm(W ~ X1B + X2B + X3B + HR + BB + SO + SB + SH + X123BA + HRA + BBA + SOA + E, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_1 <- sqrt(mean((y_hat - test_set$W)^2)) #> 6.3000
summary_1 <- summary(fit)

# Model after removing SO
fit <- lm(W ~ X1B + X2B + X3B + HR + BB + SB + SH + X123BA + HRA + BBA + SOA + E, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_2 <- sqrt(mean((y_hat - test_set$W)^2)) #> 6.2859
summary_2 <- summary(fit)

# Just checking how random forest performs
fit <- randomForest(W ~ X1B + X2B + X3B + HR + BB + SB + SH + X123BA + HRA + BBA + SOA + E, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_3 <- sqrt(mean((y_hat - test_set$W)^2)) #> 7.7628

# Model including R and RA
fit <- lm(W ~ R + RA + X1B + X2B + X3B + HR + BB + SB + X123BA + HRA + BBA + SOA + E, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_4 <- sqrt(mean((y_hat - test_set$W)^2)) #> 4.6790
summary_4 <- summary(fit)
varimp <- varImp(fit)

# Data manipulation to adjust for the mean differences among year groups 

## X1B ##
means_X1B <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(X1B)) %>%
  mutate(adjust_X1B = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_X1B) %>% 
  mutate(X1B = X1B + adjust_X1B) %>%
  select(-adjust_X1B)

## X2B ##
means_X2B <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(X2B)) %>%
  mutate(adjust_X2B = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_X2B) %>% 
  mutate(X2B = X2B + adjust_X2B) %>%
  select(-adjust_X2B)

## X3B ##
means_X3B <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(X3B)) %>%
  mutate(adjust_X3B = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_X3B) %>% 
  mutate(X3B = X3B + adjust_X3B) %>%
  select(-adjust_X3B)

## HR ##
means_HR <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(HR)) %>%
  mutate(adjust_HR = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_HR) %>% 
  mutate(HR = HR + adjust_HR) %>%
  select(-adjust_HR)

## BB ##
means_BB <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(BB)) %>%
  mutate(adjust_BB = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_BB) %>% 
  mutate(BB = BB + adjust_BB) %>%
  select(-adjust_BB)

## SB ##
means_SB <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(SB)) %>%
  mutate(adjust_SB = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_SB) %>% 
  mutate(SB = SB + adjust_SB) %>%
  select(-adjust_SB)

## SH ##
means_SH <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(SH)) %>%
  mutate(adjust_SH = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_SH) %>% 
  mutate(SH = SH + adjust_SH) %>%
  select(-adjust_SH)

## X123BA ##
means_X123BA <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(X123BA)) %>%
  mutate(adjust_X123BA = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_X123BA) %>% 
  mutate(X123BA = X123BA + adjust_X123BA) %>%
  select(-adjust_X123BA)

## HRA ##
means_HRA <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(HRA)) %>%
  mutate(adjust_HRA = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_HRA) %>% 
  mutate(HRA = HRA + adjust_HRA) %>%
  select(-adjust_HRA)

## BBA ##
means_BBA <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(BBA)) %>%
  mutate(adjust_BBA = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_BBA) %>% 
  mutate(BBA = BBA + adjust_BBA) %>%
  select(-adjust_BBA)

## SOA ##
means_SOA <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(SOA)) %>%
  mutate(adjust_SOA = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_SOA) %>% 
  mutate(SOA = SOA + adjust_SOA) %>%
  select(-adjust_SOA)

## E ##
means_E <- Teams %>% 
  group_by(group) %>%
  summarize(group_mean = mean(E)) %>%
  mutate(adjust_E = round(group_mean[4] - group_mean)) %>%
  select(-group_mean)

New_Teams <- Teams %>%
  left_join(means_E) %>% 
  mutate(E = E + adjust_E) %>%
  select(-adjust_E)

# HR vs. W of New_Teams to see how it changed
pp2 <- New_Teams %>%
  ggplot(aes(HR, W, color = group)) + geom_point() + ggtitle("New_Teams Dataset") +
  theme(legend.position = c(0.85,0.15),
        legend.title = element_blank())
p4 <- p4 + ggtitle("Teams Dataset")
ggarrange(p4, pp2, ncol = 2)

# Re-create prediction set and validation set
set.seed(5, sample.kind="Rounding")
validation_index <- createDataPartition(y = New_Teams$W, times = 1, p = 0.3, list = FALSE)
prediction_set <- New_Teams[-validation_index,]
validation_set <- New_Teams[validation_index,]

# Re-create train set and test set
set.seed(50, sample.kind="Rounding")
test_index <- createDataPartition(y = prediction_set$W, times = 1, p = 0.3, list = FALSE)
train_set <- prediction_set[-test_index,]
test_set <- prediction_set[test_index,]

# Data-manipulated model 
fit <- lm(W ~ R + RA + X1B + X2B + X3B + HR + BB + SB + X123BA + HRA + BBA + SOA + E, data = train_set)
y_hat <- predict(fit, test_set)
RMSE_5 <- sqrt(mean((y_hat - test_set$W)^2)) #> 4.4407
summary_5 <- summary(fit)

# Final test using validation set
fit <- lm(W ~ R + RA + X1B + X2B + X3B + HR + BB + SB + X123BA + HRA + BBA + SOA + E, data = prediction_set)
y_hat <- predict(fit, validation_set)
RMSE_val <- sqrt(mean((y_hat - validation_set$W)^2)) #> 4.5805
summary_val <- summary(fit)

####################################################
### Part 2 (Bonus). Batted Ball Prediction
####################################################

# Data Processing
# 
# filenames <- list.files(path = "statcast_data", full.names=TRUE)
# data_raw <- map_df(filenames, read_csv)
# data <- data_raw[,c("launch_speed","launch_angle","events","bb_type")]
# data <- na.omit(data)

# Read pre-processed data (as shown above) from Github repository
data <- read_csv("https://raw.githubusercontent.com/chkim15/Win-Prediction-for-Baseball/main/statcast_data.csv")

# Data contains four columns ('launch_speed', 'launch_angle', 'events', 'bb_type')
str(data)

# 16 different categories for 'events'
unique(data$events)

# Reset 'events' into three categories ('Hit', 'Out', 'HR')
data$events <- recode(data$events, "single"="Hit", "double"="Hit","triple"="Hit",
                      "home_run"="HR",
                      "field_out"="Out", "force_out"="Out", "grounded_into_double_play"="Out",
                      "sac_fly"="Out", "fielders_choice"="Out", "double_play"="Out",
                      "fielders_choice_out"="Out","sac_fly_double_play"="Out", "triple_play"="Out",
                      "sac_bunt"="Out", "field_error"="Out", "catcher_interf"="Out")

# Reset the names for 'bb_type' (batted-ball type)
data$bb_type <- recode(data$bb_type, "fly_ball"="Fly", "ground_ball"="GB", "popup"="Pop", "line_drive"="Liner")

# Create 'new_events' column containing 'Fly/GB/Pop/Liner/HR'
data <- data %>% mutate(new_events = ifelse(events=="HR","HR",bb_type))

# Make sure 'events' and 'new_events' are factors
data$events <- factor(data$events)
data$new_events <- factor(data$new_events, level = c("GB","Liner","Fly","Pop","HR"))


########### GB/Liner/Fly/Pop/HR Prediction ################

# Set training set and testing set
# set.seed()
test_index <- createDataPartition(data$new_events,times=1,p=0.3,list=FALSE)
train_set <- data[-test_index,]
test_set <- data[test_index,]

# Train the model (Random Forest)
ctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 5)
fit <- train(new_events ~ launch_speed + launch_angle, method='rf', data=train_set, trControl=ctrl)

# Run the model on the test set
predicted <- predict(fit, test_set)
# Check out the confusion matrix
confusionMatrix(predicted, test_set$new_events)

## Visualization using Plotly
# Exit velocities from 40 to 120
x <- seq(40,120,by=1)

# Hit angles from 0 to 60
y <- seq(0,60,by=1)

# Make a data frame of the relevant x and y values
plotDF <- data.frame(expand.grid(x,y))

# Add the column names
colnames(plotDF) <- c('launch_speed','launch_angle')

# Add the classification
plotPredictions <- predict(fit,newdata=plotDF)
plotDF$pred <- plotPredictions

# The plot of GB/Liner/Fly/Pop/HR based on 'launch_speed' and 'launch_angle'
p_bb <- plot_ly(data=plotDF, x=~launch_speed, y = ~launch_angle, color=~pred, type="scatter", mode="markers") %>%
  layout(title = "Exit Velocity vs. Launch Angle for GB/Liner/Fly/Pop/HR",
         xaxis = list(title = "Exit Velocity"),
         yaxis = list(title = "Launch Angle"))
p_bb


########### Hit/Out/HR Prediction ################
# test_index <- createDataPartition(data$events,times=1,p=0.3,list=FALSE)
# train_set <- data[-test_index,]
# test_set <- data[test_index,]


# train the model
ctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 5)
fit <- train(events ~ launch_speed + launch_angle, method='rf', data=train_set, trControl=ctrl)

# Run the model on the test set
predicted <- predict(fit, test_set)
# Check out the confusion matrix
confusionMatrix(predicted, test_set$events)


# Exit velocities from 40 to 120
x <- seq(40,120,by=1)

# Hit angles from 0 to 60
y <- seq(0,60,by=1)

# Make a data frame of the relevant x and y values
plotDF <- data.frame(expand.grid(x,y))

# Add the correct column names
colnames(plotDF) <- c('launch_speed','launch_angle')

# Add the classification
plotPredictions <- predict(fit,newdata=plotDF)
plotDF$pred <- plotPredictions

p2_bb <- plot_ly(data=plotDF, x=~launch_speed, y = ~launch_angle, color=~pred, type="scatter", mode="markers") %>%
  layout(title = "Exit Velocity vs. Launch Angle for Hit/Out/HR",
         xaxis = list(title = "Exit Velocity"),
         yaxis = list(title = "Launch Angle"))
p2_bb

