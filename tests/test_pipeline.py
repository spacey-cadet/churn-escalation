"""
Sanity tests run in CI on every push (see .github/workflows/ci.yml). These aren't
exhaustive model-quality tests -- they check that the pipeline's mechanics (gates,
registry, feature store, cascade ordering) behave correctly, since that's what
silently breaking would be most dangerous.

Now that the feature store and registry are AWS-backed (DynamoDB + S3), the
isolation fixture below uses moto's `mock_aws()` to stand up throwaway tables and
a throwaway bucket per test -- no real AWS calls, no shared state between tests,
no local SQLite/disk dependency, and no AWS bill from running the suite.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import boto3
import moto
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config

AWS_REGION = "us-east-1"


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """Every test gets its own mocked DynamoDB tables and S3 bucket. moto
    intercepts the botocore HTTP layer, so it doesn't matter that
    feature_store.py / registry.py created their boto3 client/resource objects
    at import time -- as long as mock_aws() is active when a call is actually
    made, it's routed to moto's virtual AWS instead of a real account."""
    # moto still requires botocore to find *some* credentials to sign requests
    # with, even though mock_aws() intercepts them before anything real
    # happens -- dummy values are the documented moto pattern, safe in CI.
    monkeypatch.setenv("AWS_DEFAULT_REGION", AWS_REGION)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setattr(config, "PRODUCTION_POINTER", "production")
    monkeypatch.setattr(config, "STAGING_POINTER", "staging")

    import feature_store
    import registry

    with moto.mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        dynamodb.create_table(
            TableName=feature_store.ONLINE_TABLE,
            KeySchema=[{"AttributeName": "entity_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "entity_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName=feature_store.OFFLINE_TABLE,
            KeySchema=[
                {"AttributeName": "entity_id", "KeyType": "HASH"},
                {"AttributeName": "ts", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "entity_id", "AttributeType": "S"},
                {"AttributeName": "ts", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.create_table(
            TableName=feature_store.LOG_TABLE,
            KeySchema=[
                {"AttributeName": "log_shard", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "log_shard", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.create_bucket(Bucket=registry.BUCKET)

        yield


def test_ingestion_gate_quarantines_within_tolerance_and_proceeds():
    import quality.ingestion_gate as ingestion_gate
    df = pd.DataFrame({
        "entity_id": ["a", None, "c"],
        "support_tickets_30d": [1, 2, 3],
    })
    result = ingestion_gate.run_ingestion_gate(df)
    # 1/3 nulls is above MAX_QUARANTINE_RATE, so this batch should be BLOCKED even
    # though quarantine still isolates the bad row.
    assert result["checks"]["not_null_entity_id"]["passed"] is False
    assert len(result["quarantined_df"]) == 1
    assert len(result["clean_df"]) == 2


def test_ingestion_gate_passes_when_quarantine_rate_is_low():
    import quality.ingestion_gate as ingestion_gate
    # 1 null out of 100 rows is within MAX_QUARANTINE_RATE (5%) -- proceeds.
    df = pd.DataFrame({
        "entity_id": ["a"] * 99 + [None],
        "support_tickets_30d": list(range(100)),
    })
    result = ingestion_gate.run_ingestion_gate(df)
    assert result["checks"]["not_null_entity_id"]["passed"] is True
    assert len(result["quarantined_df"]) == 1
    assert len(result["clean_df"]) == 99


def test_ingestion_gate_passes_clean_data():
    import quality.ingestion_gate as ingestion_gate
    df = pd.DataFrame({"entity_id": ["a", "b", "c"], "support_tickets_30d": [1, 2, 3]})
    result = ingestion_gate.run_ingestion_gate(df)
    assert result["checks"]["not_null_entity_id"]["passed"] is True
    assert len(result["quarantined_df"]) == 0


def test_transformation_gate_catches_orphaned_keys():
    import quality.transformation_gate as transformation_gate
    features = pd.DataFrame({"entity_id": ["a", "b", "c"]})
    labels = pd.DataFrame({"entity_id": ["a", "b"]})  # "c" is orphaned
    result = transformation_gate.run_transformation_gate(features, labels)
    assert result["passed"] is False
    ref_check = [r for r in result["checks"] if "relationships" in r["test"]][0]
    assert ref_check["orphaned_count"] == 1


def test_transformation_gate_passes_clean_data():
    import quality.transformation_gate as transformation_gate
    features = pd.DataFrame({"entity_id": ["a", "b", "c"]})
    labels = pd.DataFrame({"entity_id": ["a", "b", "c"]})
    result = transformation_gate.run_transformation_gate(features, labels)
    assert result["passed"] is True


def test_feature_store_point_in_time_join_ignores_future_writes():
    import feature_store
    t0 = datetime(2026, 1, 1)
    t1 = datetime(2026, 1, 10)
    t2 = datetime(2026, 1, 20)  # future, relative to the label we'll join against

    feature_store.write_features("cust_1", {"support_tickets_30d": 1.0}, ts=t0)
    feature_store.write_features("cust_1", {"support_tickets_30d": 2.0}, ts=t1)
    feature_store.write_features("cust_1", {"support_tickets_30d": 99.0}, ts=t2)  # future leakage bait

    as_of = datetime(2026, 1, 15)
    result = feature_store.point_in_time_join("cust_1", as_of)
    assert result["features"]["support_tickets_30d"] == 2.0  # NOT 99.0 -- that would be leakage


def test_feature_store_online_read_returns_latest():
    import feature_store
    feature_store.write_features("cust_2", {"support_tickets_30d": 1.0}, ts=datetime(2026, 1, 1))
    feature_store.write_features("cust_2", {"support_tickets_30d": 5.0}, ts=datetime(2026, 1, 2))
    online = feature_store.read_online("cust_2")
    assert online["features"]["support_tickets_30d"] == 5.0  # overwrite semantics, like Redis


def test_reconciliation_catches_injected_sync_bug():
    import feature_store
    ts = datetime(2026, 1, 1)
    feature_store.write_features("cust_3", {"support_tickets_30d": 1.0}, ts=ts)

    # Simulate a sync bug: online store silently has a stale/wrong value.
    # (Was a raw sqlite3 UPDATE against the local file; now an UpdateItem
    # against the mocked online table -- same intent, new backend.)
    table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(feature_store.ONLINE_TABLE)
    table.update_item(
        Key={"entity_id": "cust_3"},
        UpdateExpression="SET features_json = :v",
        ExpressionAttributeValues={":v": json.dumps({"support_tickets_30d": 999.0})},
    )

    mismatches = feature_store.reconcile(["cust_3"])
    assert len(mismatches) == 1
    assert mismatches[0]["entity_id"] == "cust_3"


def test_registry_register_and_load_roundtrip():
    import registry
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    calibrator = LogisticRegression().fit([[0], [1]], [0, 1])
    version = registry.register(
        model=model, calibrator=calibrator,
        thresholds={"t_low": 0.2, "t_high": 0.6},
        metrics={"pr_auc": 0.9}, data_version="test@1",
        hyperparameters={"n_estimators": 10},
    )
    bundle = registry.load(version)
    assert bundle["model_card"]["thresholds"]["t_low"] == 0.2
    assert bundle["model_card"]["stage"] == "none"


def test_registry_first_model_always_passes_gate():
    import registry
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    calibrator = LogisticRegression().fit([[0.1], [0.9]], [0, 1])
    version = registry.register(
        model=model, calibrator=calibrator, thresholds={"t_low": 0.3, "t_high": 0.7},
        metrics={"pr_auc": 0.5}, data_version="test@1", hyperparameters={},
    )
    X = pd.DataFrame({"f": [0, 1, 2, 3]})
    # Registry loads model with feature name-agnostic array input in this test's model.
    y = pd.Series([0, 0, 1, 1])
    gate = registry.champion_challenger_gate(version, {"held_out": (X.values.reshape(-1, 1), y)})
    assert gate["passed"] is True
    assert gate["champion_version"] is None


def test_cascade_threshold_ordering_is_enforced():
    from src.modeling.train import derive_cascade_thresholds
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.random.uniform(0, 0.4, 90), np.random.uniform(0.6, 1.0, 10)])
    thresholds = derive_cascade_thresholds(y, scores)
    assert thresholds["t_low"] <= thresholds["t_high"]
