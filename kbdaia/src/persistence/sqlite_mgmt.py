# persistence/sqlite_mgmt.py
# version 0.2 - SQLite table qa_cache cache maintenance logic

import sqlite3
import os

def manage_cache_limit(db_path: str, limit: int):
    """
    Ensures the database stays at or below the 'limit' provided.
    Logic:
    1. Check if current count > limit.
    2. Apply Rule 1: Delete old (pre-today) NULL/Blank feedback.
    3. Apply Rule 2: Delete old (pre-today) 'N' feedback.
    4. Apply Rule 3: Delete old (pre-today) all records.
    5. Fallback: Delete absolute oldest until count == limit.
    """
    if not os.path.exists(db_path):
        print(f"[Maintenance] DB Path not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        # Get actual current count
        cur.execute("SELECT COUNT(*) FROM qa_cache")
        count_before = cur.fetchone()[0]

        if count_before <= limit:
            # print(f"[Maintenance] Current count ({count_before}) is within limit ({limit}). No deletion needed.")
            return

        print(f"[Maintenance] Count ({count_before}) exceeds limit ({limit}). Starting pruning logic...")

        # RULE 1: Delete records older than today with NULL or blank feedback
        cur.execute("""
            DELETE FROM qa_cache 
            WHERE (feedback_status IS NULL OR feedback_status = '') 
            AND date(created_at) < date('now')
        """)
        conn.commit()

        # Check count after Rule 1
        cur.execute("SELECT COUNT(*) FROM qa_cache")
        if cur.fetchone()[0] <= limit:
            print(f"[Maintenance] Rule 1 cleared enough records.")
            return

        # RULE 2: Delete records older than today with 'N' feedback
        cur.execute("""
            DELETE FROM qa_cache 
            WHERE feedback_status = 'N' 
            AND date(created_at) < date('now')
        """)
        conn.commit()

        # Check count after Rule 2
        cur.execute("SELECT COUNT(*) FROM qa_cache")
        if cur.fetchone()[0] <= limit:
            print(f"[Maintenance] Rule 2 cleared enough records.")
            return

        # RULE 3: Delete all records older than today
        cur.execute("""
            DELETE FROM qa_cache 
            WHERE date(created_at) < date('now')
        """)
        conn.commit()

        # FINAL FALLBACK (Strict Enforcement)
        # If we have 10 records and limit is 5, but all 10 were created TODAY,
        # Rule 1, 2, and 3 will do nothing. This step forces it to the limit.
        cur.execute("SELECT COUNT(*) FROM qa_cache")
        count_now = cur.fetchone()[0]
        
        if count_now > limit:
            to_delete = count_now - limit
            print(f"[Maintenance] Rules 1-3 (date-based) were not enough. Forcing deletion of {to_delete} oldest records.")
            
            # Delete the 'N' oldest records based on creation time
            cur.execute(f"""
                DELETE FROM qa_cache 
                WHERE id IN (
                    SELECT id FROM qa_cache 
                    ORDER BY created_at ASC 
                    LIMIT {to_delete}
                )
            """)
            conn.commit()

        cur.execute("SELECT COUNT(*) FROM qa_cache")
        print(f"[Maintenance] Cleanup finished. Final count: {cur.fetchone()[0]}")

    except Exception as e:
        print(f"[Maintenance] Error: {e}")
    finally:
        conn.close()