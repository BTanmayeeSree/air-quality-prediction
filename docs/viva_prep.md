# Viva Preparation

**1. What is AQI?**
A single number (with a category like Good/Moderate/Severe) that
summarizes how polluted the air is, based on pollutant concentrations.

**2. What is air pollution?**
The presence of harmful substances (like PM2.5, NO2, CO) in the air at
levels that can harm health or the environment.

**3. Why did you select this project?**
Air quality is a real, high-impact problem, and it's a good fit for
regression — clear numeric target, real-world data, genuine
model-comparison opportunity.

**4. Why is this a machine-learning problem?**
Because AQI depends on multiple weather variables in a non-linear way
that's hard to capture with fixed rules — a model can learn the
relationship from historical data instead.

**5. Is this classification or regression?**
Regression — we predict a continuous AQI number. (A category label is
then derived from that number afterward, but that's a lookup, not a
separate classifier.)

**6. Why is AQI the target?**
It's the standard way air quality is communicated to the public, and
it's directly derivable from the PM2.5 data available.

**7. What features are used?**
Eight weather variables: mean/max/min temperature, sea-level pressure,
humidity, visibility, and mean/max wind speed. No pollutant
concentrations are used as inputs (only PM2.5 is used to build the
target).

**8. Why did you choose Random Forest?**
The tuned Random Forest had the highest mean 5-fold CV R² (0.6826) on
the training partition. The final test set was not used to select it.

**9. What is Linear Regression?**
A model that fits a straight-line (weighted sum) relationship between
inputs and the output.

**10. What is a Decision Tree?**
A model that splits data into branches based on feature thresholds to
reach a prediction; easy to interpret but prone to overfitting.

**11. What is Random Forest?**
An ensemble of many decision trees, each trained on a random subset of
data/features, whose predictions are averaged — reduces overfitting.

**12. What is Gradient Boosting?**
An ensemble that builds trees sequentially, each one correcting the
errors of the previous ones.

**13. What is MAE?**
Mean Absolute Error — the average absolute difference between predicted
and actual values.

**14. What is RMSE?**
Root Mean Squared Error — like MAE, but squares errors first (penalizing
large errors more) before taking the square root.

**15. What is R²?**
The proportion of variance in the target that the model explains; 1.0 is
perfect, 0 means no better than predicting the mean.

**16. What is overfitting?**
When a model learns the training data (including its noise) too well and
performs poorly on new, unseen data. Here the single Decision Tree had
the lowest test R² (0.3605), although it remained positive.

**17. What is underfitting?**
When a model is too simple to capture the real pattern, performing
poorly on both training and test data.

**18. What is feature engineering?**
Creating or selecting input variables that help a model perform better
— here, mainly deciding which raw weather columns to use, since the
dataset had no date/time to build lag or seasonal features from.

**19. What is data leakage?**
When information that wouldn't be available at prediction time
(including data derived from the target) leaks into the training
features, making test performance look better than it really would be
in production. This project avoids it: none of the model's input
features are pollutant concentrations.

**20. Why shouldn't future information be used?**
Because in real deployment, you only have information available up to
the moment of prediction — using future data during training gives an
unrealistic, inflated performance estimate.

**21. How did you handle missing values?**
One row had a missing PM2.5 target and was dropped. The 22 exact-zero
PM2.5 values cannot be verified without sensor metadata; they are treated
as unverified placeholders and excluded, while original CSVs are kept.
Weather values that are missing or physically invalid are median-imputed
inside the model pipeline.

**22. How did you handle outliers?**
Extreme PM2.5/AQI values (severe pollution days) were kept, since they
are real events, not data errors — removing them would bias the model
against exactly the days it matters most to predict well.

**23. How did you split the data?**
A random 80/20 train/test split with a fixed seed, since the dataset has
no reliable date column to justify a chronological split.

**24. Which model was selected?**
Random Forest (tuned), with 200 estimators, max depth 12, and minimum
leaf size 2. It was selected by training CV R², not by test metrics.

**25. Why did it perform best?**
 It had the highest mean 5-fold CV R² on training data (0.6826). Its
 untouched-test MAE is 55.7229, MSE 5661.8129, RMSE 75.2450, and R² 0.6474.

**26. What are the limitations?**
 The inputs omit traffic, construction, crop-burning, and other emissions;
 the target is PM2.5-only; data covers one city and has no reliable dates.
 It cannot support a future-forecasting claim.

**27. What is the future scope?**
Multi-pollutant data, live weather/AQI APIs, true time-series
forecasting with dated records, multi-city support.
