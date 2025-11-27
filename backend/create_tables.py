#!/usr/bin/env python3
"""
Create database tables if they don't exist.
"""

from database import Base, engine


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully!")


if __name__ == "__main__":
    create_tables()
