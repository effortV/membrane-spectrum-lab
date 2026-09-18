"""Verified, non-destructive migration: originals remain recoverable; active paths switch."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3

from .config import save_runtime_settings
from .utils import json_dumps, sha256_file, utc_now


def _copy_verified(source: Path, destination: Path, records: list[dict]) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Not a regular input file: {source}")
    digest = sha256_file(source)
    if destination.exists() and sha256_file(destination) != digest:
        raise ValueError(f"Destination conflict; original not overwritten: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise ValueError(f"Copy hash mismatch: {destination}")
    records.append(
        {
            "source": str(source),
            "destination": str(destination),
            "sha256": digest,
            "bytes": source.stat().st_size,
        }
    )


def migrate_storage(
    project: Path,
    source_data: Path,
    destination: Path,
    *,
    apply: bool = False,
    server_stopped: bool = False,
) -> dict:
    project, source_data, destination = (
        project.resolve(),
        source_data.resolve(),
        destination.resolve(),
    )
    if (
        destination == destination.anchor
        or destination.parent == destination
        or destination == source_data
        or destination == project
        or destination.name.lower() != "xps-agent"
    ):
        raise ValueError(
            "Destination must be a dedicated XPS-agent directory, never a drive/data/project root"
        )
    if (
        destination.is_relative_to(project)
        or project.is_relative_to(destination)
        or source_data.is_relative_to(destination)
        or destination.is_relative_to(source_data)
    ):
        raise ValueError("Input/project and centralized output directories must not overlap")
    manifest = destination / "catalog" / "storage_migration.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    old_workspace = project / "workspace"
    new_workspace = destination / "workspace"
    mappings = [
        (source_data / "reference", destination / "raw" / "literature"),
        (project / "final", destination / "raw" / "spectra"),
        (old_workspace, new_workspace),
    ]
    inputs = []
    for name in ("NF.xlsx", "RO.xlsx"):
        source = source_data / name
        if not source.is_file():
            raise ValueError(f"Missing original table: {source}")
        inputs.append((source, destination / "raw" / "tables" / name))
    for source_root, new_root in mappings[:2]:
        if source_root.name == "final":
            directories = [source_root / name for name in ("NF-N", "NF-O", "RO-N", "RO-O")]
        else:
            directories = [source_root]
        for directory in directories:
            if not directory.is_dir():
                raise ValueError(f"Missing scientific input: {directory}")
            for source in directory.rglob("*"):
                if source.is_file():
                    if source.is_symlink() or not source.resolve().is_relative_to(
                        source_root.resolve()
                    ):
                        raise ValueError("Input path escapes the selected scientific directory")
                    inputs.append((source, new_root / source.relative_to(source_root)))
    result = {
        "created_at": utc_now(),
        "project": str(project),
        "storage_root": str(destination),
        "raw_files": len(inputs),
        "policy": "Verified copies; legacy originals and old workspace retained as pre-migration recovery material. All future active writes use centralized storage.",
        "mappings": [{"from": str(a), "to": str(b)} for a, b in mappings],
    }
    if not apply:
        return {**result, "dry_run": True}
    if not server_stopped:
        raise ValueError(
            "Stop only the recognized project UI after its active task completes before migrating live state"
        )
    if (new_workspace / "state" / "xps_agent.sqlite3").exists():
        raise ValueError(
            "Destination already has state without a completed migration; review before continuing"
        )
    records = []
    for source, target in inputs:
        _copy_verified(source, target, records)
    # Avoid copying WAL/SHM as a SQLite snapshot. No delete or move of original data.
    ignored = {
        "xps_agent.sqlite3",
        "xps_agent.sqlite3-wal",
        "xps_agent.sqlite3-shm",
        "migration_hold.json",
        "migration_ready.json",
    }
    if old_workspace.exists():
        for source in old_workspace.rglob("*"):
            if (
                source.is_file()
                and not (source.parent == old_workspace / "state" and source.name in ignored)
                and "__pycache__" not in source.parts
            ):
                if source.is_symlink() or not source.resolve().is_relative_to(
                    old_workspace.resolve()
                ):
                    raise ValueError("Workspace path escapes project workspace")
                _copy_verified(source, new_workspace / source.relative_to(old_workspace), records)
    database = new_workspace / "state" / "xps_agent.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    if (old_workspace / "state" / "xps_agent.sqlite3").exists():
        with (
            sqlite3.connect(old_workspace / "state" / "xps_agent.sqlite3") as source,
            sqlite3.connect(database) as target,
        ):
            source.backup(target)
    replacements = [(str(a), str(b)) for a, b in mappings]
    replacements.append((str(source_data), str(destination / "raw" / "tables")))

    def remap(value):
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            return value
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    from .db import StateDB

    db = StateDB(database)
    db.initialize()
    with db.connect() as con:
        for table, key, fields in (
            ("documents", "doc_key", ["local_path", "metadata_json"]),
            ("document_references", "rowid", ["source_path"]),
            ("experiments", "experiment_id", ["run_dir", "metrics_json"]),
            ("artifacts", "artifact_id", ["path", "metadata_json"]),
        ):
            for row in con.execute(
                f"SELECT {key} AS item_key,{','.join(fields)} FROM {table}"
            ).fetchall():
                values = []
                for field in fields:
                    value = row[field]
                    values.append(
                        json_dumps(remap(json.loads(value)))
                        if field.endswith("_json") and value
                        else remap(value)
                    )
                con.execute(
                    f"UPDATE {table} SET {','.join(name + '=?' for name in fields)} WHERE {key}=?",
                    [*values, row["item_key"]],
                )
        check = con.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise ValueError("Migrated database integrity check failed")
    result.update(files=records, database_integrity="ok", original_data_preserved=True)
    destination.joinpath("catalog").mkdir(parents=True, exist_ok=True)
    # Configuration is committed last and preserves secrets through the existing secure writer.
    save_runtime_settings(
        project,
        {
            "XPS_STORAGE_ROOT": str(destination),
            "XPS_WORKSPACE_ROOT": str(new_workspace),
            "XPS_DATA_ROOT": str(destination / "raw" / "tables"),
            "XPS_REFERENCE_ROOT": str(destination / "raw" / "literature"),
            "XPS_LEGACY_ROOT": str(destination / "raw" / "spectra"),
        },
    )
    manifest.write_text(json_dumps(result), encoding="utf-8")
    return {key: value for key, value in result.items() if key != "files"}
