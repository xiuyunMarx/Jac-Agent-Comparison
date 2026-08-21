"""Dump and restore the benchmark dataset for reproducible runs.

Scanning a YouTube channel is non-deterministic (rate limits, channels change,
transcripts get edited), so benchmark runs must not depend on live scraping.
This command snapshots everything the agent queries at answer time — channels,
videos, transcript chunks, and the PGVector embedding tables — into a single
portable file, and restores it byte-identically into any database. All
implementations under comparison then run against the exact same data.

Usage:
    python manage.py benchmark_snapshot dump benchmark/data/snapshot.json.gz
    python manage.py benchmark_snapshot load benchmark/data/snapshot.json.gz --replace
"""

import gzip
import io
import json

import psycopg2
from django.conf import settings
from django.core.management.base import (
    BaseCommand,
    CommandError,
)

# Insert order respects foreign keys; deletes run in reverse.
TABLES = [
    "app_channel",
    "app_video",
    "app_videochunk",
    "langchain_pg_collection",
    "langchain_pg_embedding",
]

SNAPSHOT_FORMAT = 1


class Command(BaseCommand):
    """Snapshot or restore the channel/video/chunk/embedding tables."""

    help = (
        "Dump or restore the benchmark dataset (channels, videos, transcript chunks, "
        "and PGVector embeddings) so every benchmark run queries identical data. "
        "Paths ending in .gz are compressed."
    )

    def add_arguments(self, parser):
        """Register CLI arguments."""
        parser.add_argument("action", choices=["dump", "load"], help="dump: database -> file; load: file -> database")
        parser.add_argument("path", help="Snapshot file path (use a .gz suffix for compression)")
        parser.add_argument(
            "--replace",
            action="store_true",
            help="load only: delete existing rows in the target tables first "
            "(user accounts are detached from channels, not deleted)",
        )

    def handle(self, *args, **options):
        """Entry point: open one connection and dispatch to dump or load."""
        conn = psycopg2.connect(settings.PSYCOPG2_DATABASE_URL)
        try:
            if options["action"] == "dump":
                self._dump(conn, options["path"])
            else:
                self._load(conn, options["path"], replace=options["replace"])
        finally:
            conn.close()

    @staticmethod
    def _open(path, mode):
        """Open a snapshot file, transparently gzipped for .gz paths."""
        if path.endswith(".gz"):
            return gzip.open(path, mode + "t", encoding="utf-8")
        return open(path, mode, encoding="utf-8")

    @staticmethod
    def _table_exists(cursor, table):
        """Check whether a table exists in the connected database."""
        cursor.execute("SELECT to_regclass(%s)", (table,))
        return cursor.fetchone()[0] is not None

    def _dump(self, conn, path):
        """Copy every dataset table out of the database into a snapshot file."""
        snapshot = {"format": SNAPSHOT_FORMAT, "tables": {}}
        with conn.cursor() as cursor:
            for table in TABLES:
                if not self._table_exists(cursor, table):
                    self.stderr.write(
                        f"Warning: table '{table}' does not exist, skipping "
                        "(embedding tables are created on the first channel scan)"
                    )
                    continue
                buffer = io.StringIO()
                cursor.copy_expert(f"COPY {table} TO STDOUT WITH (FORMAT csv, HEADER true)", buffer)
                data = buffer.getvalue()
                snapshot["tables"][table] = data
                rows = max(data.count("\n") - 1, 0)
                self.stdout.write(f"  {table}: {rows} rows")

        if not snapshot["tables"]:
            raise CommandError("Nothing to dump: none of the dataset tables exist yet")

        with self._open(path, "w") as f:
            json.dump(snapshot, f)
        self.stdout.write(self.style.SUCCESS(f"Snapshot written to {path}"))

    def _load(self, conn, path, *, replace):
        """Restore a snapshot file into the database in one transaction."""
        try:
            with self._open(path, "r") as f:
                snapshot = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CommandError(f"Cannot read snapshot {path}: {e}")

        if snapshot.get("format") != SNAPSHOT_FORMAT:
            raise CommandError(f"Unsupported snapshot format: {snapshot.get('format')!r}")
        tables = snapshot.get("tables", {})

        with conn.cursor() as cursor:
            missing = [t for t in tables if not self._table_exists(cursor, t)]
            if missing:
                raise CommandError(
                    f"Target tables missing: {missing}. Run 'python manage.py migrate' first; "
                    "the langchain_pg_* tables are created by PGVector on the first channel scan "
                    "(or the first agent invocation against a channel)."
                )

            if replace:
                # Users reference channels; detach them instead of cascading deletes.
                if self._table_exists(cursor, "app_user"):
                    cursor.execute("UPDATE app_user SET channel_id = NULL")
                for table in reversed(TABLES):
                    if table in tables:
                        cursor.execute(f"DELETE FROM {table}")

            for table in TABLES:
                data = tables.get(table)
                if data is None:
                    continue
                cursor.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)", io.StringIO(data))
                rows = max(data.count("\n") - 1, 0)
                self.stdout.write(f"  {table}: {rows} rows")

            # COPY bypasses sequences; realign the only serial PK among the tables.
            if "app_videochunk" in tables:
                cursor.execute(
                    "SELECT setval(pg_get_serial_sequence('app_videochunk', 'id'), "
                    "COALESCE((SELECT MAX(id) FROM app_videochunk), 1))"
                )

        conn.commit()
        self.stdout.write(self.style.SUCCESS(f"Snapshot {path} loaded"))
