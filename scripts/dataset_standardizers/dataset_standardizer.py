from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import pyarrow as pa # we are using pyarrow because of https://stackoverflow.com/questions/51361356/a-comparison-between-fastparquet-and-pyarrow
import os

class DatasetStandardizer(ABC):
    def __init__(self, schema_fields: list[pa.Field]):
        self.schema = pa.schema(schema_fields)

    def prepare_parquet_file(self, raw_data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with pq.ParquetWriter(f"{output_dir}/dataset.parquet", self.schema) as writer:
            for data in self.data_generator(raw_data_dir):
                table = pa.Table.from_pydict(data, schema=self.schema)

                # Write the table chunk to the Parquet file
                writer.write_table(table)

    @abstractmethod
    def data_generator(self, raw_data_dir_path):
        pass