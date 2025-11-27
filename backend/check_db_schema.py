#!/usr/bin/env python3
"""
Check database schema to verify HuggingFace columns exist.
"""

import sqlite3
import os


def check_schema():
    db_path = "tinker_platform.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check users table schema
    cursor.execute("PRAGMA table_info(users)")
    users_columns = cursor.fetchall()

    print("Users table columns:")
    for col in users_columns:
        print(f"  {col[1]} - {col[2]}")

    # Check if HF columns exist
    column_names = [col[1] for col in users_columns]
    hf_columns = ["hf_token_encrypted", "hf_username", "hf_token_last_verified"]

    print("\nHuggingFace columns check:")
    for col in hf_columns:
        if col in column_names:
            print(f"  ✓ {col} exists")
        else:
            print(f"  ✗ {col} missing")

    # Check deployments table
    try:
        cursor.execute("PRAGMA table_info(deployments)")
        deployments_columns = cursor.fetchall()
        print(f"\nDeployments table exists with {len(deployments_columns)} columns")
    except sqlite3.OperationalError:
        print("\n✗ Deployments table does not exist")

    conn.close()


if __name__ == "__main__":
    check_schema()
