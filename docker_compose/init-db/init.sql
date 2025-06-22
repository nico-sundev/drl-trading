CREATE TABLE IF NOT EXISTS feature_config_versions (
    semver TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feature_definitions JSONB NOT NULL,
    description TEXT
);
