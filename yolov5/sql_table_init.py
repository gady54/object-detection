import mysql.connector

# Connect to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="gady",
  password="Gad554007@",
  database="Drone_Detection_Data"
)

# Create a cursor object
mycursor = mydb.cursor()

# SQL statement to drop the table if it exists
drop_table_query = "DROP TABLE IF EXISTS data"

try:
    # Execute the drop table query
    mycursor.execute(drop_table_query)
    print("Table 'data' deleted successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")

# Close the cursor and connection
mycursor.close()
mydb.close()
