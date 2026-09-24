import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
URL = "https://books.toscrape.com/"

response = requests.get(URL, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

print("Website connected successfully!")
print("Page title:", soup.title.get_text(strip=True))
books = soup.select("article.product_pod")

print("Books found on this page:", len(books))
def extract_book(book, page_url):
    try:
        title_tag = book.select_one("h3 a")
        price_tag = book.select_one(".price_color")
        rating_tag = book.select_one("p.star-rating")
        availability_tag = book.select_one(".availability")

        # Skip the book if essential fields are missing
        if not all([title_tag, price_tag, rating_tag, availability_tag]):
            print("Skipping book: required field missing")
            return None

        title = title_tag.get("title", "").strip()
        price = price_tag.get_text(strip=True)
        rating_classes = rating_tag.get("class", [])
        availability = availability_tag.get_text(" ", strip=True)

        # Rating class should contain something like One, Two, Three...
        rating = next(
            (value for value in rating_classes
             if value in ["One", "Two", "Three", "Four", "Five"]),
            None
        )

        if not title or not price or not rating:
            print("Skipping book: invalid data")
            return None

        book_url = title_tag.get("href")

        if not book_url:
            print(f"Skipping book: URL missing for {title}")
            return None

        detail_url = urljoin(page_url, book_url)

        detail_response = requests.get(detail_url, timeout=10)
        detail_response.raise_for_status()

        detail_soup = BeautifulSoup(detail_response.text, "html.parser")

        category_tag = detail_soup.select_one(
            "ul.breadcrumb li:nth-of-type(3)"
        )

        if category_tag:
            category = category_tag.get_text(strip=True)
        else:
            category = "Unknown"

        return {
            "title": title,
            "price": price,
            "rating": rating,
            "availability": availability,
            "category": category,
        }

    except (requests.RequestException, AttributeError, KeyError) as error:
        print(f"Skipping book because of parsing/request error: {error}")
        return None
all_books = []

page_url = URL

for page_number in range(1, 4):
    print(f"Scraping page {page_number}...")

    response = requests.get(page_url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.select("article.product_pod")

    for book in books:
        book_data = extract_book(book, page_url)
        if book_data is not None:
            all_books.append(book_data)
    next_button = soup.select_one("li.next a")

    if next_button:
        page_url = urljoin(page_url, next_button["href"])
    else:
        break

print("Total books scraped:", len(all_books))
import pandas as pd

df = pd.DataFrame(all_books)

print(df.head())
print(df.shape)
df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .str.replace("Â", "", regex=False)
    .astype(float)
)

df["price_inr"] = df["price_gbp"] * 105.50
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

df["rating"] = df["rating"].map(rating_map)
df["in_stock"] = df["availability"].str.strip().str.lower().eq("in stock")

print(df[["availability", "in_stock"]].head())
print(df["in_stock"].value_counts())
print(df[["rating"]].head())
print(df[["price", "price_gbp", "price_inr"]].head())
# Keep only the cleaned columns needed for the final dataset
cleaned_df = df[
    ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
].copy()

# Save cleaned dataset
cleaned_df.to_csv("books_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")
print("Final shape:", cleaned_df.shape)
print(cleaned_df.head())