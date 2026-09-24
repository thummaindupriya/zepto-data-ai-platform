import sqlite3
import pandas as pd

DB_NAME = "books.db"

# Load cleaned dataset
df = pd.read_csv("books_cleaned.csv")

# Connect to SQLite
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Create normalized categories table
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

# Create books table
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

# Insert categories
for category in df["category"].dropna().unique():
    cursor.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (category,)
    )

# Insert books
for _, row in df.iterrows():
    category_id = cursor.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (row["category"],)
    ).fetchone()[0]

    cursor.execute("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["price_inr"],
        row["rating"],
        int(row["in_stock"]),
        category_id
    ))

conn.commit()

# Verify
print("Database created successfully!")

print("\nCategories:")
print(pd.read_sql("SELECT * FROM categories", conn))

print("\nNumber of books:")
print(pd.read_sql("SELECT COUNT(*) AS book_count FROM books", conn))

conn.close()