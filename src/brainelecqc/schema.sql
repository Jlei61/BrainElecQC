BEGIN;
CREATE TABLE IF NOT EXISTS ephys_recordings (
    recording_id TEXT PRIMARY KEY,
    participant_id TEXT,
    label TEXT NOT NULL,
    modality TEXT NOT NULL CHECK(modality IN ('eeg', 'seeg', 'ecog')),
    session_id TEXT,
    input_artifact_id TEXT NOT NULL UNIQUE,
    import_key TEXT UNIQUE,
    request_sha256 TEXT NOT NULL,
    descriptor_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ephys_artifacts (
    artifact_id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES ephys_recordings(recording_id),
    kind TEXT NOT NULL CHECK(kind IN ('raw_input', 'derived_voltage')),
    storage_path TEXT NOT NULL UNIQUE,
    content_sha256 TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ephys_artifacts_raw_input
    ON ephys_artifacts(recording_id) WHERE kind='raw_input';
CREATE INDEX IF NOT EXISTS idx_ephys_recordings_participant ON ephys_recordings(participant_id);
CREATE TABLE IF NOT EXISTS seeg_qc_runs (
    run_id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES ephys_recordings(recording_id),
    artifact_id TEXT NOT NULL REFERENCES ephys_artifacts(artifact_id),
    status TEXT NOT NULL CHECK(status IN ('running', 'succeeded', 'failed', 'cancelled')),
    provenance_json TEXT NOT NULL,
    request_json TEXT NOT NULL,
    report_json TEXT,
    report_sha256 TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    review_json TEXT NOT NULL,
    frozen_json TEXT,
    export_sha256 TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seeg_qc_runs_recording ON seeg_qc_runs(recording_id, created_at);
CREATE TABLE IF NOT EXISTS seeg_qc_audit (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES seeg_qc_runs(run_id),
    revision INTEGER NOT NULL,
    action TEXT NOT NULL,
    reviewer TEXT,
    snapshot_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seeg_qc_audit_run ON seeg_qc_audit(run_id);
PRAGMA user_version=2;
COMMIT;
