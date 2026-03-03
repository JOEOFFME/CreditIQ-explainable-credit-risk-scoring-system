from credit_risk_scoring.app import health


def test_health() -> None:
    assert health() == {"status": "ok"}
