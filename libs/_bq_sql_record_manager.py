from typing import List, Optional, Sequence
from google.cloud import bigquery
import uuid
import time
from google.api_core.exceptions import BadRequest


class BQSQLRecordManager:
    def __init__(
        self,
        namespace: str,
        *,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> None:
        self.namespace = namespace
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=self.project_id)
        self.table_ref = self.client.get_table(f"{self.dataset_id}.{self.table_id}")
        self.schema = [
            bigquery.SchemaField("uuid", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("key", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("namespace", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("group_id", "STRING"),
            bigquery.SchemaField("updated_at", "FLOAT64", mode="REQUIRED"),
        ]

    def create_schema(self) -> None:
        table = bigquery.Table(self.table_ref, schema=self.schema)
        table = self.client.create_table(table, exists_ok=True)

    def get_time(self) -> float:
        query = "SELECT UNIX_SECONDS(CURRENT_TIMESTAMP()) AS current_time"
        query_job = self.client.query(query)
        result = query_job.result()
        return float(list(result)[0]["current_time"])

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

        # Get the current time from the server.
        update_time = self.get_time()

        if time_at_least and update_time < time_at_least:
            raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")

        rows_to_upsert = [
            {
                "uuid": str(uuid.uuid4()),
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids)
        ]

        # Create a temporary table to store the rows to upsert
        temp_table_name = f"{self.table_id}_temp_{uuid.uuid4().hex}"
        table_id = f"{self.project_id}.{self.dataset_id}.{temp_table_name}"
        temp_table = bigquery.Table(table_id, schema=self.schema)
        temp_table_ref = self.client.create_table(temp_table)

        # Load the rows into the temporary table
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        job_config.schema = self.schema

        job = self.client.load_table_from_json(
            rows_to_upsert, temp_table_ref, job_config=job_config
        )
        job.result()  # Wait for the load job to complete

        # Perform an upsert operation to merge the temporary table with the main table
        query = f"""
            MERGE `{self.project_id}.{self.dataset_id}.{self.table_id}` AS target
            USING `{self.project_id}.{self.dataset_id}.{temp_table_name}` AS source
            ON target.key = source.key AND target.namespace = source.namespace
            WHEN MATCHED THEN
                UPDATE SET target.updated_at = source.updated_at, target.group_id = source.group_id
            WHEN NOT MATCHED THEN
                INSERT (uuid, key, namespace, updated_at, group_id)
                VALUES (GENERATE_UUID(), source.key, source.namespace, source.updated_at, source.group_id)
        """

        query_job = self.client.query(query)
        query_job.result()  # Wait for the query to complete

        # Delete the temporary table
        self.client.delete_table(temp_table_ref)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        query = f"""
            SELECT key
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE key IN UNNEST(@keys)
                AND namespace = @namespace
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("keys", "STRING", keys),
                bigquery.ScalarQueryParameter("namespace", "STRING", self.namespace),
            ]
        )
        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        found_keys = {row["key"] for row in results}
        return [key in found_keys for key in keys]

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        query = f"""
            SELECT key
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE namespace = @namespace
        """
        query_parameters = [
            bigquery.ScalarQueryParameter("namespace", "STRING", self.namespace),
        ]

        if after:
            query += " AND updated_at > @after"
            query_parameters.append(
                bigquery.ScalarQueryParameter("after", "FLOAT64", after)
            )
        if before:
            query += " AND updated_at < @before"
            query_parameters.append(
                bigquery.ScalarQueryParameter("before", "FLOAT64", before)
            )
        if group_ids:
            query += " AND group_id IN UNNEST(@group_ids)"
            query_parameters.append(
                bigquery.ArrayQueryParameter("group_ids", "STRING", group_ids)
            )

        if limit:
            query += " LIMIT @limit"
            query_parameters.append(
                bigquery.ScalarQueryParameter("limit", "INT64", limit)
            )

        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()
        return [row["key"] for row in results]

    def delete_keys(self, keys: Sequence[str], retries=5, delay=60) -> None:
        """
        Attempts to delete keys from a BigQuery table, with retries and delays to handle
        errors related to the streaming buffer.

        Args:
            keys (Sequence[str]): The keys to delete.
            retries (int): Number of retry attempts.
            delay (int): Delay in seconds between retries.
        """
        query = f"""
            DELETE FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE key IN UNNEST(@keys)
                AND namespace = @namespace
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("keys", "STRING", keys),
                bigquery.ScalarQueryParameter("namespace", "STRING", self.namespace),
            ]
        )
        for attempt in range(retries):
            try:
                query_job = self.client.query(query, job_config=job_config)
                query_job.result()  # Wait for the job to complete
                return  # Exit if successful
            except BadRequest as e:
                if "would affect rows in the streaming buffer" in str(e):
                    if attempt < retries - 1:
                        time.sleep(delay)  # Wait before retrying
                    else:
                        raise RuntimeError(
                            f"Failed to delete keys after {retries} attempts."
                        )
                else:
                    raise e
