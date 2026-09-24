import sqlite3
import pandas as pd

conn = sqlite3.connect("books.db")

queries = {
    "query_1_select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """,

    "query_2_order_by": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """,

    "query_3_distinct": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating
    """,

    "query_4_between": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
    """,

    "query_5_join": """
        SELECT
            b.title,
            b.rating,
            b.price_gbp,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC
        LIMIT 10
    """
}

for name, query in queries.items():
    print(f"\n{'=' * 60}")
    print(name.upper())
    print("=" * 60)

    result = pd.read_sql(query, conn)
    print(result)

# ---------------------------------------------------------
# Pandas read_sql + pandas merge equivalent of the JOIN
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PANDAS READ_SQL + MERGE")
print("=" * 60)

# Read two SQL query results into pandas
top_rated_df = pd.read_sql("""
    SELECT title, rating, price_gbp, category_id
    FROM books
    WHERE rating >= 4
""", conn)

category_df = pd.read_sql("""
    SELECT category_id, category_name
    FROM categories
""", conn)

# Reproduce the SQL JOIN using pandas merge
merged_df = pd.merge(
    top_rated_df,
    category_df,
    on="category_id",
    how="inner"
)

# Display equivalent result
merged_df = merged_df[
    ["title", "rating", "price_gbp", "category_name"]
].sort_values(
    ["rating", "price_gbp"],
    ascending=[False, False]
).head(10)

print("\nSQL-style result reproduced using pandas.merge:")
print(merged_df)
conn.close()