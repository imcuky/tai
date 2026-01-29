import sqlite3
import pandas as pd
import os

def filter_test_files(db_path="cs61a_metadata.db", output_csv="cs61a_test_files.csv"):
    
    # Path resolution for DB
    if not os.path.exists(db_path):
         # Try looking up
        if os.path.exists(os.path.join("../../..", db_path)):
            db_path = os.path.join("../../..", db_path)
    
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return

    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    try:
        # Filter records for slides or tutorials/discussions, excluding youtube
        query = """
            SELECT * FROM file 
            WHERE (
                lower(relative_path) LIKE '%slide%' 
                OR lower(relative_path) LIKE '%disc%'
                OR lower(relative_path) LIKE '%tutorial%'
            )
            AND lower(relative_path) NOT LIKE '%youtube%'
        """
        
        df = pd.read_sql_query(query, conn)
        print(f"Found {len(df)} records matching slides/disc/tutorial/lab (excluding youtube).")
        
        # If too few, maybe try lab?
        if len(df) < 5:
             print("Few results found. Trying to include 'lab'...")
             query_lab = """
                SELECT * FROM file 
                WHERE (
                    lower(relative_path) LIKE '%slide%' 
                    OR lower(relative_path) LIKE '%disc%'
                    OR lower(relative_path) LIKE '%tutorial%'
                    OR lower(relative_path) LIKE '%lab%'
                )
                AND lower(relative_path) NOT LIKE '%youtube%'
            """
             df = pd.read_sql_query(query_lab, conn)
             print(f"Found {len(df)} records with lab included.")

        if not df.empty:
            # Create output directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_script_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to output folder
            final_output_path = os.path.join(output_dir, os.path.basename(output_csv))
            df.to_csv(final_output_path, index=False)
            print(f"Saved filtered results to {final_output_path}")
        else:
            print("No records found.")
            
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    filter_test_files()
