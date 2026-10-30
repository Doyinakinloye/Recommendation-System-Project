# Amazon Reviews Recommendation System

A content-based recommendation system built on Amazon customer reviews. The notebook cleans a reviews dataset, converts review text into TF-IDF vectors, and uses cosine similarity to recommend reviews that are textually similar to a chosen review.

Notebook: `Recommendation_system.ipynb`

## Overview

1. **Load** the Amazon reviews CSV.
2. **Explore and clean** the data (ratings, dates, missing values).
3. **Vectorize** the combined review title and text with TF-IDF.
4. **Compute** pairwise cosine similarity between all reviews.
5. **Recommend** the top-N most similar reviews for any review index.

## Dataset

The notebook expects a file named `Amazon_Reviews.csv`, loaded from Google Drive:

```
/content/drive/MyDrive/Amazon recommendation system/Amazon_Reviews.csv
```

Columns used or referenced in the notebook:

| Column | Description |
| --- | --- |
| `Reviewer Name` | Name of the reviewer |
| `Profile Link` | Reviewer profile URL (dropped) |
| `Country` | Reviewer country |
| `Review Count` | Number of reviews by the reviewer, stored as text (parsed to `Review_Count_numeric`, then dropped) |
| `Rating` | Text such as "Rated 4 out of 5 stars" (parsed to `Rating_numeric`, then dropped) |
| `Review Date` | Date the review was posted |
| `Date of Experience` | Date of the customer's experience |
| `Review Title` | Short review headline |
| `Review Text` | Full review body |

The CSV is read with `engine="python"` and `on_bad_lines="skip"`, so malformed rows are silently dropped.

## Pipeline

### Data cleaning
- Extract numeric star ratings from the `Rating` text with a regex.
- Extract numeric review counts from `Review Count`.
- Drop `Profile Link`, `Review Count` and `Rating`.
- Parse `Date of Experience` and `Review Date` as dates.
- Drop rows with no valid rating.
- Fill missing `Date of Experience` values with `Review Date`.
- Fill missing `Country` values.

### Exploratory analysis
- Dataset info, missing-value counts and descriptive statistics.
- Most common review titles.
- Rating distribution (seaborn count plot). Most reviews are 1-star.

### Feature engineering
- `content` = `Review Title` + `Review Text`, to give the vectorizer more context.
- `TfidfVectorizer(stop_words="english", max_features=1000)` produces the TF-IDF matrix.

### Similarity and recommendation
- `cosine_similarity(tfidf_matrix)` gives an N x N similarity matrix.
- `recommend(review_index, n=5)` sorts one row of that matrix and returns the top `n` similar reviews (excluding the review itself), showing `Review Title` and `Rating_numeric`.

## Usage

```python
# Recommend the 5 reviews most similar to the review at position 5
recommend(5)

# Inspect the original review text
df_clean['content'][5]

# Get 10 recommendations instead
recommend(5, n=10)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

The notebook was written for Google Colab (it reads from Google Drive). To run it locally, change the CSV path in the data-loading cell.

## How to run

1. Place `Amazon_Reviews.csv` where the notebook can read it, and update the path if needed.
2. Open `Recommendation_system.ipynb` in Colab or Jupyter.
3. Run all cells in order.
4. Call `recommend(<index>)` with any row position.


## Project structure

```
.
├── Recommendation_system.ipynb
├── README.md
└── Amazon_Reviews.csv   # not included; supply your own
```
