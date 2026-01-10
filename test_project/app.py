# Test vulnerable application
import os
import sqlite3

def get_user(username):
    # SQL Injection vulnerability
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def execute_command(cmd):
    # Command injection vulnerability
    os.system(cmd)

if __name__ == "__main__":
    user = get_user(input("Username: "))
    print(user)
