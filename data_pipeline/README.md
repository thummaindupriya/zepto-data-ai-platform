# Data Pipeline

## Overview

This module scrapes book data from Books to Scrape, cleans the collected data, converts prices from GBP to INR, stores the data in a normalized SQLite database, and demonstrates SQL and pandas-based analysis.

## Source

Website: Books to Scrape

The scraper automatically collects the first 3 pages of books, resulting in 60 books across multiple categories.

## Files

* scrape_books.py — scrapes and cleans the book data
* books_cleaned.csv — cleaned dataset
* database.py — creates and populates the SQLite database
* books.db — SQLite database
* queries.py — required SQL queries and pandas JOIN reproduction

## Cleaning Decisions

The following transformations are applied:

1. price_gbp

   * Removes the GBP currency symbol and encoding artifact.
   * Converts the value to a numeric float.

2. price_inr

   * Uses the fixed assignment exchange rate:
   * **1 GBP = 105.50 INR**
   * No external currency API is required.

3. rating

   * Converts star-rating text:

     * One → 1
     * Two → 2
     * Three → 3
     * Four → 4
     * Five → 5

4. in_stock

   * "In stock" is converted to True.
   * Other availability values would be represented as False.

5. Robust parsing

   * Missing essential fields are detected.
   * Invalid books are skipped instead of crashing the complete scraping process.
   * Missing category information is represented as "Unknown".

### Data Quality Handling

The scraper validates required fields before adding a book to the final dataset. If a required field is missing or a request fails for an individual book, that book is skipped and the remaining scraping process continues.

## Final Dataset

The cleaned dataset contains:

* 60 books
* 6 final columns:

  * title
  * price_gbp
  * price_inr
  * rating
  * in_stock
  * category

## Database Design

The SQLite database uses two normalized tables.

### categories

| Column          | Type    | Description          |
| --------------- | ------- | -------------------- |
| category_id     | INTEGER | Primary key          |
| category_name   | TEXT    | Unique category name |

### books

| Column        | Type    | Description                                   |
| ------------- | ------- | --------------------------------------------- |
|   book_id     | INTEGER | Primary key                                   |
|  title        | TEXT    | Book title                                    |
|  price_gbp    | REAL    | Price in GBP                                  |
|  price_inr    | REAL    | Converted price in INR                        |
|  rating       | INTEGER | Rating from 1 to 5                            |
|  in_stock     | INTEGER | SQLite representation of boolean stock status |
|  category_id  | INTEGER | Foreign key to categories                     |

The books.category_id foreign key connects each book to its category.

## SQL Queries

Five SQL queries are implemented in queries.py:

1. SELECT with WHERE — filters books with rating 4 or higher.
2. ORDER BY with LIMIT — finds the 10 most expensive books.
3. DISTINCT — lists distinct ratings.
4. BETWEEN — filters books within a GBP price range.
5. JOIN — combines books with their category names.

The JOIN result is also reproduced using pandas.merge().

## Pandas Verification

Two SQL results are loaded using pd.read_sql().

The books and categories DataFrames are then joined in memory using:

```python
pd.merge(
    top_rated_df,
    category_df,
    on="category_id",
    how="inner"
)
```

The resulting records match the SQL JOIN output.

## How to Run

From the data_pipeline directory:

```powershell
python scrape_books.py
```

This creates/updates:

```text
books_cleaned.csv
```

Then create the SQLite database:

```powershell
python database.py
```

Finally run the SQL and pandas verification:

```powershell
python queries.py
```
## Validation

The pipeline was validated end-to-end with 60 scraped books. The SQLite database contains 25 categories and 60 books, and the SQL JOIN output was reproduced using `pandas.merge()`.
## Reproducibility

The complete pipeline can be regenerated without manually copying data from the website. The scraper obtains the source data programmatically, applies the documented cleaning rules, creates the cleaned CSV, and the database and query scripts use that generated dataset.
