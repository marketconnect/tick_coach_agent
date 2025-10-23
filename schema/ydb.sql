CREATE TABLE checkpoints (
    client_id Utf8,
    checkpoint String,
    created_at Timestamp,
    PRIMARY KEY (client_id)
)
WITH (
    TTL = Interval("P3D") ON created_at
);