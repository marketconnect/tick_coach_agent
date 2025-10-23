import os
import pickle
import ydb
import ydb.iam
from typing import Any, Optional, Dict
import logging
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)

log = logging.getLogger(__name__)

class YDBCheckpointer(BaseCheckpointSaver):
    """
    A checkpointer that stores conversation state in Yandex Database (YDB).
    It serializes checkpoints using pickle.
    """

    def __init__(self):
        super().__init__()
        log.warning("Initializing YDBCheckpointer...")
        endpoint = os.environ.get("YDB_ENDPOINT")
        database = os.environ.get("YDB_DATABASE")
        log.warning(f"YDB_ENDPOINT: {endpoint}, YDB_DATABASE: {database}")

        if not endpoint or not database:
            log.error("YDB_ENDPOINT and YDB_DATABASE environment variables must be set.")
            raise ValueError("YDB_ENDPOINT and YDB_DATABASE environment variables must be set.")

        self.table_path = os.path.join(database, "checkpoints")
        
        # Assumes the cloud function has a service account attached.
        credentials = ydb.iam.MetadataUrlCredentials()
        
        driver_config = ydb.DriverConfig(
            endpoint=endpoint,
            database=database,
            credentials=credentials,
        )
        self.driver = ydb.Driver(driver_config)
        try:
            log.warning("Connecting to YDB driver...")
            self.driver.wait(timeout=5, fail_fast=True)
            log.warning("YDB driver connection successful.")
        except Exception as e:
            log.exception("Failed to connect to YDB driver.")
            raise RuntimeError(f"Failed to connect to YDB: {e}")
            
        self.pool = ydb.SessionPool(self.driver)

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.get_tuple(config).checkpoint if self.get_tuple(config) else None

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        client_id = config["configurable"]["thread_id"]
        log.warning(f"Getting checkpoint tuple for client_id: {client_id}")

        def callee(session: ydb.Session):
            query = f"""
                DECLARE $client_id AS Utf8;
                SELECT checkpoint FROM `{self.table_path}` WHERE client_id = $client_id;
            """
            
            prepared_query = session.prepare(query)
            result_sets = session.transaction(ydb.SerializableReadWrite()).execute(
                prepared_query,
                {"$client_id": client_id},
                commit_tx=True,
            )
            
            if not result_sets or not result_sets[0].rows or result_sets[0].rows[0].checkpoint is None:
                log.warning(f"No checkpoint tuple found for client_id: {client_id}")
                return None
            
            log.warning(f"Checkpoint tuple found for client_id: {client_id}")
            pickled_checkpoint = result_sets[0].rows[0].checkpoint
            try:
                saved_data = pickle.loads(pickled_checkpoint)
                checkpoint = saved_data["checkpoint"]
                metadata = saved_data["metadata"]

                if not isinstance(checkpoint, dict) or "v" not in checkpoint:
                    log.warning(f"Loaded checkpoint for client_id {client_id} is invalid or outdated. Discarding.")
                    return None

                return CheckpointTuple(
                    config=config,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=None,
                )
            except Exception as e:
                log.warning(f"Failed to load/decode checkpoint for client_id: {client_id}. Treating as new conversation. Error: {e}")
                return None

        return self.pool.retry_operation_sync(callee)

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ):
        """
        Store a checkpoint with its configuration and metadata.

        NOTE: Our schema stores only the latest checkpoint per client_id.
        We return an updated config with checkpoint_id so LangGraph can proceed.
        """
        # Reuse existing tuple saver to persist checkpoint+metadata
        self.put_tuple(
            config,
            CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
            ),
        )

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Return updated config as required by BaseCheckpointSaver.put(...)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_tuple(self, config: Dict[str, Any], checkpoint_tuple: CheckpointTuple) -> None:
        client_id = config["configurable"]["thread_id"]
        log.warning(f"Putting checkpoint tuple for client_id: {client_id}")
        data_to_save = {
            "checkpoint": checkpoint_tuple.checkpoint,
            "metadata": checkpoint_tuple.metadata,
        }
        pickled_checkpoint = pickle.dumps(data_to_save)

        def callee(session: ydb.Session):
            query = f"""
                DECLARE $client_id AS Utf8;
                DECLARE $checkpoint AS String;

                UPSERT INTO `{self.table_path}` (client_id, checkpoint, created_at)
                VALUES ($client_id, $checkpoint, CurrentUtcTimestamp());
            """
            prepared_query = session.prepare(query)
            session.transaction(ydb.SerializableReadWrite()).execute(
                prepared_query,
                {
                    "$client_id": client_id,
                    "$checkpoint": pickled_checkpoint,
                },
                commit_tx=True,
            )
 
        self.pool.retry_operation_sync(callee)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes,
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint.

        Minimal no-op implementation to satisfy LangGraph's interface and
        avoid NotImplementedError during execution. Extend to persist writes
        in YDB if/when required.
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "")
            count = len(writes) if writes else 0
            log.warning(
                f"put_writes called for thread_id: {thread_id}, task_id: {task_id}, writes: {count} (no-op)."
            )
        except Exception:
            # Be conservative: never propagate errors from no-op path
            pass
        return None

    def delete_thread(self, thread_id: str) -> None:
        """
        Delete all checkpoints associated with a specific thread ID.
        """
        def callee(session: ydb.Session):
            query = f"""
                DECLARE $client_id AS Utf8;
                DELETE FROM `{self.table_path}` WHERE client_id = $client_id;
            """
            prepared_query = session.prepare(query)
            session.transaction(ydb.SerializableReadWrite()).execute(
                prepared_query,
                {"$client_id": thread_id},
                commit_tx=True,
            )

        self.pool.retry_operation_sync(callee)