import sqlite3
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Create a new SQLite database (or connect to an existing one)
conn = sqlite3.connect('ecommerce.db')
cursor = conn.cursor()

# Create Users Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Create Products Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    price REAL,
    stock INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Create Orders Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    total REAL,
    status TEXT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
)
''')

# Create Cart Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Cart (
    cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
)
''')

# Create OrderItems Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS OrderItems (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    price REAL,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
)
''')

# Generate Dummy Data
# Create Users
users = []
for _ in range(10):
    users.append((fake.user_name(), fake.email(), fake.password()))

cursor.executemany('INSERT INTO Users (username, email, password) VALUES (?, ?, ?)', users)

# Create Products
products = []
for _ in range(20):
    products.append((fake.word(), fake.text(max_nb_chars=200), round(random.uniform(5.0, 100.0), 2), random.randint(1, 100)))

cursor.executemany('INSERT INTO Products (name, description, price, stock) VALUES (?, ?, ?, ?)', products)

# Create Carts and Orders
for user_id in range(1, 11):
    # Create a cart for each user
    cursor.execute('INSERT INTO Cart (user_id) VALUES (?)', (user_id,))
    
    # Create an order for each user
    order_total = 0
    cursor.execute('INSERT INTO Orders (user_id, total) VALUES (?, ?)', (user_id, order_total))
    order_id = cursor.lastrowid

    # Generate OrderItems
    for _ in range(random.randint(1, 5)):  # Each order can have 1 to 5 items
        product_id = random.randint(1, 20)  # Random product from the product table
        quantity = random.randint(1, 3)  # Random quantity
        price = cursor.execute('SELECT price FROM Products WHERE product_id = ?', (product_id,)).fetchone()[0]
        order_item_total = price * quantity
        
        cursor.execute('INSERT INTO OrderItems (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)', 
                       (order_id, product_id, quantity, price))

        # Update the order total
        order_total += order_item_total

    # Update the order total in Orders table
    cursor.execute('UPDATE Orders SET total = ? WHERE order_id = ?', (order_total, order_id))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database created and populated with dummy data.")
