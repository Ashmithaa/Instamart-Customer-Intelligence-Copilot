-- Q1
SELECT customer_id, COUNT(*) AS order_count,
              ROUND(SUM(order_amount), 2) AS total_spending
       FROM orders
       WHERE status='completed' AND order_amount > 500
       GROUP BY customer_id
       HAVING SUM(order_amount) > 8000
       ORDER BY total_spending DESC;

-- Q2
SELECT c.city, p.category, COUNT(o.order_id) AS completed_orders,
              ROUND(SUM(o.order_amount), 2) AS total_revenue
       FROM orders o
       JOIN customers c ON o.customer_id=c.customer_id
       JOIN products p ON o.product_id=p.product_id
       WHERE o.status='completed'
       GROUP BY c.city, p.category
       ORDER BY c.city, total_revenue DESC;

-- Q3
SELECT product_id, product_name, category, price_inr
       FROM products
       WHERE price_inr > (SELECT AVG(price_inr) FROM products)
       ORDER BY price_inr DESC;

-- Q4
SELECT c.customer_id, c.city, c.gender
       FROM customers c
       LEFT JOIN orders o ON c.customer_id=o.customer_id
                         AND o.status='completed'
       WHERE o.order_id IS NULL
       ORDER BY c.customer_id;

-- Q5
WITH customer_spending AS (
           SELECT customer_id, COUNT(*) AS completed_orders,
                  SUM(order_amount) AS total_spending
           FROM orders WHERE status='completed'
           GROUP BY customer_id
       ),
       average_spending AS (
           SELECT AVG(total_spending) AS avg_spending FROM customer_spending
       )
       SELECT cs.customer_id, cs.completed_orders,
              ROUND(cs.total_spending, 2) AS total_spending,
              ROUND(a.avg_spending, 2) AS average_spending
       FROM customer_spending cs CROSS JOIN average_spending a
       WHERE cs.total_spending > a.avg_spending
       ORDER BY cs.total_spending DESC;

-- Q6
WITH customer_spending AS (
           SELECT c.customer_id, c.city, COUNT(o.order_id) AS completed_orders,
                  SUM(o.order_amount) AS total_spending
           FROM customers c JOIN orders o ON c.customer_id=o.customer_id
           WHERE o.status='completed'
           GROUP BY c.customer_id, c.city
       )
       SELECT customer_id, city, completed_orders,
              ROUND(total_spending, 2) AS total_spending,
              RANK() OVER (PARTITION BY city ORDER BY total_spending DESC) AS spending_rank,
              DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spending DESC) AS dense_rank,
              ROW_NUMBER() OVER (
                  PARTITION BY city ORDER BY total_spending DESC, customer_id
              ) AS row_number
       FROM customer_spending
       ORDER BY city, spending_rank, customer_id;

-- Q7
WITH daily_revenue AS (
           SELECT DATE(order_ts) AS order_date, SUM(order_amount) AS daily_revenue
           FROM orders WHERE status='completed'
           GROUP BY DATE(order_ts)
       )
       SELECT order_date, ROUND(daily_revenue, 2) AS daily_revenue,
              ROUND(SUM(daily_revenue) OVER (
                  ORDER BY order_date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
              ), 2) AS running_revenue
       FROM daily_revenue ORDER BY order_date;
