# Mashable News Popularity Analysis

Predicting the virality of online news articles using the [UCI Online News Popularity dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (Mashable, 2015).

## Overview

The dataset contains 58 features describing ~40,000 Mashable articles — content length, media count (images/videos), publishing channel, keyword quality, timing, and sentiment. The target variable is `shares` (total share count).

The analysis covers:

- **EDA** — distribution of shares (heavy right skew), per-channel and per-day breakdowns, top correlated features
- **Regression** — predicting share count with Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost; evaluated with RMSE and 5-fold cross-validation
- **Binary Classification** — predicting whether an article is "popular" (≥ 1400 shares, the median); best models reach ~66% accuracy and ~0.73 AUC
- **Prescriptive Analysis** — actionable writing recommendations derived from feature importance (optimal word count, image count, keyword strength, publish day, topic channel)

## Key Findings

- LightGBM and CatBoost outperform linear models; boosting captures the non-linear patterns in the data.
- R² is low (~0.15) because virality depends heavily on factors outside the dataset (influencer reach, timing luck).
- Top predictive features: `kw_avg_avg`, `self_reference_avg_sharess`, `num_imgs`.

## Setup

```bash
conda env create -f environment.yml
conda activate mashable
jupyter notebook news_popularity_analysis.ipynb
```
