"""
Database migration to add HuggingFace integration fields.
Run this once to add the new columns to existing database.
"""

from database import engine
from sqlalchemy import text

def migrate():
    """Add HuggingFace fields to database."""

    with engine.connect() as conn:
        # Add fields to users table
        try:
            conn.execute(text("""
                ALTER TABLE users ADD COLUMN hf_token_encrypted TEXT
            """))
            conn.commit()
            print("✓ Added hf_token_encrypted to users")
        except Exception as e:
            print(f"hf_token_encrypted might already exist: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE users ADD COLUMN hf_username TEXT
            """))
            conn.commit()
            print("✓ Added hf_username to users")
        except Exception as e:
            print(f"hf_username might already exist: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE users ADD COLUMN hf_token_last_verified TIMESTAMP
            """))
            conn.commit()
            print("✓ Added hf_token_last_verified to users")
        except Exception as e:
            print(f"hf_token_last_verified might already exist: {e}")

        # Add fields to checkpoints table
        try:
            conn.execute(text("""
                ALTER TABLE checkpoints ADD COLUMN hf_repo_url TEXT
            """))
            conn.commit()
            print("✓ Added hf_repo_url to checkpoints")
        except Exception as e:
            print(f"hf_repo_url might already exist: {e}")

        try:
            conn.execute(text("""
                ALTER TABLE checkpoints ADD COLUMN hf_deployed_at TIMESTAMP
            """))
            conn.commit()
            print("✓ Added hf_deployed_at to checkpoints")
        except Exception as e:
            print(f"hf_deployed_at might already exist: {e}")

        # Create deployments table
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    hf_repo_name TEXT NOT NULL,
                    hf_repo_url TEXT NOT NULL,
                    hf_model_id TEXT NOT NULL,
                    is_private INTEGER DEFAULT 0,
                    merged_weights INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    deployed_at TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """))
            conn.commit()
            print("✓ Created deployments table")
        except Exception as e:
            print(f"deployments table might already exist: {e}")

    print("\n✅ Migration completed successfully!")
    print("\nNext steps:")
    print("1. Add ENCRYPTION_KEY to your .env file")
    print("   Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    print("2. Restart the backend server")
    print("3. Configure your HuggingFace token in Settings")

if __name__ == "__main__":
    migrate()
