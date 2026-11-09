#!/usr/bin/env python
# coding: utf-8

# # Part A — Data Foundation
# 
# **Instamart Customer Intelligence & Support Copilot**
# 

# ##  Project setup and load raw data

# In[1]:


from pathlib import Path
import sqlite3
import math
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

CLEAN.mkdir(parents=True, exist_ok=True)
SQL_PATH.parent.mkdir(parents=True, exist_ok=True)

required_files = ["customers.csv", "products.csv", "orders.csv"]
missing_files = [name for name in required_files if not (RAW / name).exists()]
if missing_files:
    raise FileNotFoundError(
        f"Missing raw files in {RAW}: {missing_files}. "
        "Save this notebook in the project root and check data/raw."
    )

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

print("Project root:", ROOT)
print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)


# ##  Raw data audit

# In[2]:


print("RAW DATA SHAPES")
for name, frame in {
    "customers": customers, "products": products, "orders": orders
}.items():
    print(f"{name}: {frame.shape}")

print("\\nExact duplicate rows")
print({
    "customers": int(customers.duplicated().sum()),
    "products": int(products.duplicated().sum()),
    "orders": int(orders.duplicated().sum())
})

print("\\nCustomer audit")
age_numeric = pd.to_numeric(customers["age"], errors="coerce")
print("Blank ages:", int(customers["age"].astype(str).str.strip().eq("").sum()))
print("Invalid/out-of-range ages (10-100 rule):",
      int(((age_numeric < 10) | (age_numeric > 100)).sum()))
print("Distinct raw city spellings:", customers["city"].nunique())
print("Raw gender values:", customers["gender"].value_counts(dropna=False).to_dict())

print("\\nOrder audit")
amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Rows:", len(orders))
print("Unique order IDs:", orders["order_id"].nunique())
print("Invalid/non-numeric amounts:", int(amount_numeric.isna().sum()))
print("Negative delivery times:", int((delivery_numeric < 0).sum()))
print("Blank delivery times:", int(orders["delivery_minutes"].astype(str).str.strip().eq("").sum()))
print("Status counts:", orders["status"].value_counts(dropna=False).to_dict())


# ##  Clean customer data

# In[3]:


customers_clean = customers.copy()

customers_clean["city"] = customers_clean["city"].astype(str).str.strip().str.title()

gender_map = {
    "f": "female", "female": "female",
    "m": "male", "male": "male"
}
customers_clean["gender"] = (
    customers_clean["gender"].astype(str).str.strip().str.lower()
    .map(gender_map).fillna("unknown")
)

customers_clean["age"] = pd.to_numeric(customers_clean["age"], errors="coerce")
invalid_age = (customers_clean["age"] < 10) | (customers_clean["age"] > 100)
customers_clean.loc[invalid_age, "age"] = np.nan

city_medians = customers_clean.groupby("city")["age"].transform("median")
customers_clean["age"] = customers_clean["age"].fillna(city_medians)
customers_clean["signup_date"] = pd.to_datetime(
    customers_clean["signup_date"], errors="coerce"
)

print("Clean customer rows:", len(customers_clean))
print("Missing values:")
display(customers_clean.isna().sum().to_frame("missing_values"))
print("City counts:")
display(customers_clean["city"].value_counts())
print("Gender counts:")
display(customers_clean["gender"].value_counts())
print("Age range:", customers_clean["age"].min(), "to", customers_clean["age"].max())


# ##  Clean product data

# In[4]:


products_clean = products.copy()
products_clean["price_inr"] = pd.to_numeric(products_clean["price_inr"], errors="coerce")

print("Clean product rows:", len(products_clean))
print("Duplicate product IDs:", int(products_clean["product_id"].duplicated().sum()))
print("Missing values:")
display(products_clean.isna().sum().to_frame("missing_values"))
display(products_clean.head())


# ##  Clean order data

# In[5]:


orders_clean = orders.copy()
print("Original order rows:", len(orders_clean))

orders_clean = orders_clean.drop_duplicates().copy()
print("After exact duplicate removal:", len(orders_clean))

orders_clean["order_ts"] = pd.to_datetime(orders_clean["order_ts"], errors="coerce")
orders_clean = orders_clean.sort_values("order_ts", na_position="first")
orders_clean = orders_clean.drop_duplicates(subset="order_id", keep="last").copy()

orders_clean["quantity"] = pd.to_numeric(orders_clean["quantity"], errors="coerce")
orders_clean["discount_pct"] = pd.to_numeric(orders_clean["discount_pct"], errors="coerce")
orders_clean["order_amount"] = pd.to_numeric(orders_clean["order_amount"], errors="coerce")
orders_clean["delivery_minutes"] = pd.to_numeric(orders_clean["delivery_minutes"], errors="coerce")

orders_clean.loc[orders_clean["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

orders_clean = orders_clean.merge(
    products_clean[["product_id", "price_inr", "category"]],
    on="product_id", how="left", validate="many_to_one"
)

invalid_amount = orders_clean["order_amount"].isna()
orders_clean.loc[invalid_amount, "order_amount"] = (
    orders_clean.loc[invalid_amount, "price_inr"]
    * orders_clean.loc[invalid_amount, "quantity"]
    * (1 - orders_clean.loc[invalid_amount, "discount_pct"] / 100)
)

category_medians = (
    orders_clean.loc[
        (orders_clean["status"] == "completed") &
        orders_clean["delivery_minutes"].notna()
    ].groupby("category")["delivery_minutes"].median()
)
missing_completed = (
    (orders_clean["status"] == "completed") &
    orders_clean["delivery_minutes"].isna()
)
orders_clean.loc[missing_completed, "delivery_minutes"] = (
    orders_clean.loc[missing_completed, "category"].map(category_medians)
)

orders_clean = orders_clean.drop(columns=["price_inr", "category"], errors="ignore")

print("Clean order rows:", len(orders_clean))
print("Unique order IDs:", orders_clean["order_id"].nunique())
print("Status counts:")
display(orders_clean["status"].value_counts())
print("Missing values:")
display(orders_clean.isna().sum().to_frame("missing_values"))


# ##  Save cleaned CSV files

# In[6]:


customers_clean.to_csv(CLEAN / "customers.csv", index=False)
products_clean.to_csv(CLEAN / "products.csv", index=False)
orders_clean.to_csv(CLEAN / "orders.csv", index=False)

for filename in ["customers.csv", "products.csv", "orders.csv"]:
    saved = pd.read_csv(CLEAN / filename)
    print(f"{filename}: {saved.shape[0]} rows × {saved.shape[1]} columns")


# ##  Create the SQLite database

# In[7]:


with sqlite3.connect(DB_PATH) as conn:
    customers_clean.to_sql("customers", conn, if_exists="replace", index=False)
    products_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)

    table_names = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    print("Database:", DB_PATH)
    display(table_names)
    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(f"SELECT COUNT(*) AS rows FROM {table}", conn)
        print(f"{table}: {int(count.loc[0, 'rows'])} rows")


# ##  Seven required SQL questions

# In[8]:


sql_queries = [
    # Q1
    """SELECT customer_id, COUNT(*) AS order_count,
              ROUND(SUM(order_amount), 2) AS total_spending
       FROM orders
       WHERE status='completed' AND order_amount > 500
       GROUP BY customer_id
       HAVING SUM(order_amount) > 8000
       ORDER BY total_spending DESC""",

    # Q2
    """SELECT c.city, p.category, COUNT(o.order_id) AS completed_orders,
              ROUND(SUM(o.order_amount), 2) AS total_revenue
       FROM orders o
       JOIN customers c ON o.customer_id=c.customer_id
       JOIN products p ON o.product_id=p.product_id
       WHERE o.status='completed'
       GROUP BY c.city, p.category
       ORDER BY c.city, total_revenue DESC""",

    # Q3
    """SELECT product_id, product_name, category, price_inr
       FROM products
       WHERE price_inr > (SELECT AVG(price_inr) FROM products)
       ORDER BY price_inr DESC""",

    # Q4
    """SELECT c.customer_id, c.city, c.gender
       FROM customers c
       LEFT JOIN orders o ON c.customer_id=o.customer_id
                         AND o.status='completed'
       WHERE o.order_id IS NULL
       ORDER BY c.customer_id""",

    # Q5
    """WITH customer_spending AS (
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
       ORDER BY cs.total_spending DESC""",

    # Q6
    """WITH customer_spending AS (
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
       ORDER BY city, spending_rank, customer_id""",

    # Q7
    """WITH daily_revenue AS (
           SELECT DATE(order_ts) AS order_date, SUM(order_amount) AS daily_revenue
           FROM orders WHERE status='completed'
           GROUP BY DATE(order_ts)
       )
       SELECT order_date, ROUND(daily_revenue, 2) AS daily_revenue,
              ROUND(SUM(daily_revenue) OVER (
                  ORDER BY order_date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
              ), 2) AS running_revenue
       FROM daily_revenue ORDER BY order_date"""
]

sql_text = "\n\n".join(
    f"-- Q{i}\n{query.strip()};" for i, query in enumerate(sql_queries, 1)
)
SQL_PATH.write_text(sql_text, encoding="utf-8")
print("Saved seven queries to:", SQL_PATH)

query_results = []
with sqlite3.connect(DB_PATH) as conn:
    for i, query in enumerate(sql_queries, 1):
        result = pd.read_sql_query(query, conn)
        query_results.append(result)
        print(f"Q{i}: {len(result)} rows")
        display(result.head(3))


# ##  SQL result cross-checks

# In[9]:


with sqlite3.connect(DB_PATH) as conn:
    total_revenue = pd.read_sql_query(
        "SELECT SUM(order_amount) AS revenue FROM orders WHERE status='completed'", conn
    ).iloc[0, 0]

    average_spending = pd.read_sql_query(
        """SELECT AVG(total_spending) FROM (
               SELECT customer_id, SUM(order_amount) AS total_spending
               FROM orders WHERE status='completed' GROUP BY customer_id
           )""", conn
    ).iloc[0, 0]

    ranking_ties = pd.read_sql_query(
        """WITH spending AS (
               SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spending
               FROM customers c JOIN orders o ON c.customer_id=o.customer_id
               WHERE o.status='completed'
               GROUP BY c.customer_id, c.city
           )
           SELECT RANK() OVER (PARTITION BY city ORDER BY total_spending DESC) AS rnk,
                  ROW_NUMBER() OVER (
                      PARTITION BY city ORDER BY total_spending DESC, customer_id
                  ) AS rn
           FROM spending""", conn
    )

print("Completed revenue:", round(total_revenue, 2))
print("Average customer spending:", round(average_spending, 2))
print("Ranking ties:", int((ranking_ties["rnk"] != ranking_ties["rn"]).sum()))
print("SQL query row counts:", [len(x) for x in query_results])


# ##  Pandas analysis — city and category

# In[10]:


completed_orders = orders_clean.loc[orders_clean["status"] == "completed"].copy()
analysis_df = (
    completed_orders
    .merge(customers_clean[["customer_id", "city"]], on="customer_id", how="left")
    .merge(products_clean[["product_id", "category"]], on="product_id", how="left")
)

revenue_pivot = pd.pivot_table(
    analysis_df, values="order_amount", index="city", columns="category",
    aggfunc="sum", fill_value=0, margins=True, margins_name="Total"
)
print("Revenue by city and category (INR):")
display(revenue_pivot.round(2))

city_summary = analysis_df.groupby("city").agg(
    completed_orders=("order_id", "count"),
    unique_customers=("customer_id", "nunique"),
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean")
).reset_index().sort_values("total_revenue", ascending=False)

print("City-wise summary:")
display(city_summary.round(2))


# ##  Exploratory data analysis (EDA)

# In[11]:


amount = analysis_df["order_amount"].dropna()
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("Order amount statistics")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Skewness:", round(amount.skew(), 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", int((amount < lower_fence).sum()))
print("Outliers above upper fence:", int((amount > upper_fence).sum()))

numeric_cols = analysis_df.select_dtypes(include="number")
correlations = numeric_cols.corr()
print("\\nCorrelation matrix:")
display(correlations.round(3))

if "discount_pct" in correlations.columns:
    print("Order amount vs discount_pct correlation:",
          round(correlations.loc["order_amount", "discount_pct"], 3))


# ##  Final validation

# In[12]:


checks = {
    "1. 1,200 customers": len(customers_clean) == 1200,
    "2. 40 products": len(products_clean) == 40,
    "3. 12,309 orders": len(orders_clean) == 12309,
    "4. Unique customer IDs": customers_clean["customer_id"].is_unique,
    "5. Unique product IDs": products_clean["product_id"].is_unique,
    "6. Unique order IDs": orders_clean["order_id"].is_unique,
    "7. Customer ages between 18 and 60": customers_clean["age"].between(18, 60).all(),
    "8. 11,110 completed orders": int((orders_clean["status"] == "completed").sum()) == 11110,
    "9. 1,199 cancelled orders": int((orders_clean["status"] == "cancelled").sum()) == 1199,
    "10. Completed revenue is INR 2,998,668.80":
        math.isclose(float(total_revenue), 2998668.80, abs_tol=0.01),
    "11. Q1 returns 30 rows": len(query_results[0]) == 30,
    "12. Q2 returns 30 rows": len(query_results[1]) == 30,
    "13. Q3 returns 11 rows": len(query_results[2]) == 11,
    "14. Q4 returns 25 rows": len(query_results[3]) == 25,
    "15. Q5 returns 550 rows": len(query_results[4]) == 550,
    "16. Q6 returns 1,175 rows": len(query_results[5]) == 1175,
    "17. Q7 returns 183 rows": len(query_results[6]) == 183,
    "18. Average customer spending is INR 2,552.06":
        math.isclose(float(average_spending), 2552.06, abs_tol=0.01),
}

print("PART A VALIDATION RESULTS")
print("-------------------------")
for label, passed in checks.items():
    print(f"{label}: {'PASSED' if passed else 'FAILED'}")

print("-------------------------")
print(f"Passed: {sum(checks.values())}/{len(checks)}")
if all(checks.values()):
    print("ALL CHECKPOINTS PASSED")
else:
    failed = [label for label, passed in checks.items() if not passed]
    raise AssertionError("Validation failed: " + "; ".join(failed))

