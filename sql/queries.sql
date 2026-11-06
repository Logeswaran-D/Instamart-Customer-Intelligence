-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2 three-table join: revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3 scalar subquery
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4 LEFT JOIN: customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5 CTEs: customers above average spend
WITH spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend, ROUND(a.avg_total, 2) AS avg_total
FROM spend s CROSS JOIN avg_spend a
WHERE s.total_spend > a.avg_total;

-- Q6 ranking window functions per city
WITH spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7 daily revenue with running total
WITH daily AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
