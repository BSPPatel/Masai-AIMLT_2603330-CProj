-- Query 1: Top 5 Most Expensive In-Stock Books (Demonstrates SELECT, WHERE, ORDER BY, LIMIT)
-- Returns the top 5 highest priced items that are currently available.
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating
FROM books
WHERE in_stock = 1
ORDER BY price_inr DESC
LIMIT 5;

-- Query 2: Distinct Star Ratings for Premium Books (Demonstrates DISTINCT, WHERE)
-- Identifies the unique rating levels among books priced over 2,500 INR.
SELECT DISTINCT 
    rating
FROM books
WHERE price_inr > 2500.00
ORDER BY rating ASC;

-- Query 3: Affordable Highly-Rated Books (Demonstrates BETWEEN, AND, ORDER BY)
-- Finds high-quality books (rating 4 or 5) priced in the mid-range between 1,500 and 4,000 INR.
SELECT 
    book_id,
    title,
    rating,
    price_inr
FROM books
WHERE price_inr BETWEEN 1500.00 AND 4000.00
  AND rating >= 4
ORDER BY rating DESC, price_inr ASC;

-- Query 4: Books Filtered by Specific Categories (Demonstrates IN, JOIN, WHERE)
-- Retrieves books belonging to targeted customer-favorite genres.
SELECT 
    b.book_id,
    b.title,
    c.category_name,
    b.price_inr,
    b.rating
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name IN ('Mystery', 'Historical Fiction', 'Travel')
ORDER BY c.category_name, b.price_inr DESC;

-- Query 5: Relational JOIN of Books with Category Information (Demonstrates JOIN, ORDER BY)
-- Core required relational join query linking books with parent categories for top-rated books.
SELECT 
    b.book_id,
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
WHERE b.rating = 5
ORDER BY b.price_inr DESC;

-- Query 6: Category Aggregation & Health Summary (Demonstrates JOIN, GROUP BY, Aggregates)
-- Produces executive summary by category with inventory counts and average price metrics.
SELECT 
    c.category_name,
    COUNT(b.book_id) AS total_books,
    SUM(b.in_stock) AS in_stock_count,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    MIN(b.price_inr) AS min_price_inr,
    MAX(b.price_inr) AS max_price_inr
FROM categories c
LEFT JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_books DESC;
