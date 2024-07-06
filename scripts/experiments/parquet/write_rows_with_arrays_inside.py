import pyarrow as pa
import pyarrow.parquet as pq

# Define the schema
schema = pa.schema([
    ('id', pa.int32()),
    ('values', pa.list_(pa.float64()))
])

# Create some data
data = [
    pa.array([1, 2, 3], type=pa.int32()),
    pa.array([[1.1, 2.2], [3.3, 4.4, 5.5], [6.6]], type=pa.list_(pa.float64()))
]

# Create a Table
table = pa.Table.from_arrays(data, schema=schema)

# Write the Table to a Parquet file
pq.write_table(table, 'example.parquet')

print("Parquet file written successfully.")

# Read the Table from the Parquet file
table_read = pq.read_table('example.parquet')

# print(table_read)

# Function to get the ith row without converting to pandas
def get_ith_row(table, i):
    if i < 0 or i >= table.num_rows:
        raise IndexError("Index out of range.")
    row_slice = table.slice(i, 1)
    row_data = {}
    for column in table.column_names:
        row_data[column] = row_slice[column][0].as_py()
    return row_data

# Get the 2nd row (index 1)
ith_row = get_ith_row(table_read, 1)
print("ith_row:")
print(ith_row)
# print(table.schema)