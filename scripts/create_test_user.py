#!/usr/bin/env python3
"""Create a test user with a copy of an existing user's closet."""

import os
import sys
import bcrypt
import psycopg2

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgres://postgres:~1eEQeCzn4cNvsKnN6kEYHQICalVId6D@shinkansen.proxy.rlwy.net:38559/railway")

# Test user credentials
TEST_EMAIL = "test@email.com"
TEST_PASSWORD = "test@email"
SOURCE_USER_ID = 1  # Your user ID

def main():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    try:
        # Hash the password
        password_hash = bcrypt.hashpw(TEST_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Check if test user already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (TEST_EMAIL,))
        existing = cursor.fetchone()
        
        if existing:
            test_user_id = existing[0]
            print(f"Test user already exists with ID {test_user_id}")
            # Update password
            cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s", (password_hash, test_user_id))
            # Clear existing items
            cursor.execute("DELETE FROM user_closet_items WHERE user_id::text = %s::text", (test_user_id,))
            print("Cleared existing closet items")
        else:
            # Create new test user
            cursor.execute(
                """INSERT INTO users (email, password_hash, name, created_at)
                   VALUES (%s, %s, %s, NOW()) RETURNING id""",
                (TEST_EMAIL, password_hash, "Test User")
            )
            test_user_id = cursor.fetchone()[0]
            print(f"Created test user with ID {test_user_id}")
        
        # Copy closet items from source user (cast to handle text/int mismatch)
        cursor.execute(
            """INSERT INTO user_closet_items 
               (user_id, name, category, image_url, primary_color, secondary_colors, 
                material, style_tags, occasion_tags, season_tags, embedding, created_at)
               SELECT %s, name, category, image_url, primary_color, secondary_colors,
                      material, style_tags, occasion_tags, season_tags, embedding, NOW()
               FROM user_closet_items 
               WHERE user_id::text = %s::text""",
            (test_user_id, SOURCE_USER_ID)
        )
        
        items_copied = cursor.rowcount
        print(f"Copied {items_copied} closet items to test user")
        
        conn.commit()
        print(f"\n✅ Test account ready!")
        print(f"   Email: {TEST_EMAIL}")
        print(f"   Password: {TEST_PASSWORD}")
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()

