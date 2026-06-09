POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traffic_light_config (
    intersection_id INTEGER NOT NULL,
    incoming_from INTEGER NOT NULL,
    green DOUBLE PRECISION NOT NULL,
    yellow DOUBLE PRECISION NOT NULL,
    red DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (intersection_id, incoming_from)
);

CREATE TABLE IF NOT EXISTS metrics (
    timestamp DOUBLE PRECISION NOT NULL,
    intersection_id INTEGER NOT NULL,
    incoming_from INTEGER NOT NULL,
    mc_avg_speed DOUBLE PRECISION NOT NULL,
    mc_avg_density DOUBLE PRECISION NOT NULL,
    mc_queue_length INTEGER NOT NULL,
    car_avg_speed DOUBLE PRECISION NOT NULL,
    car_avg_density DOUBLE PRECISION NOT NULL,
    car_queue_length INTEGER NOT NULL,
    local_imbalance DOUBLE PRECISION NOT NULL,
    global_imbalance DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_intersection ON metrics(intersection_id, incoming_from, timestamp);
"""
