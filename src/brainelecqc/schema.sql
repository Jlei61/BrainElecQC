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
PRAGMA user_version=1;
COMMIT;
