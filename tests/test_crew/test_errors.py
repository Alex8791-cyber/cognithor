import pickle

from cognithor.crew.errors import GuardrailFailure


def test_guardrail_failure_pickle_roundtrip():
    """GuardrailFailure must survive pickle.dumps -> pickle.loads intact.

    Regression: a plain @dataclass Exception subclass breaks multiprocessing /
    ProcessPoolExecutor / Celery because Exception's unpickle path passes a
    single arg to __init__, which the dataclass __init__ rejects. The custom
    __reduce__ fixes this by telling pickle to pass all 4 fields.
    """
    original = GuardrailFailure(
        task_id="t42",
        guardrail_name="no_pii",
        attempts=3,
        reason="email detected",
    )
    roundtripped = pickle.loads(pickle.dumps(original))

    assert isinstance(roundtripped, GuardrailFailure)
    assert roundtripped.task_id == "t42"
    assert roundtripped.guardrail_name == "no_pii"
    assert roundtripped.attempts == 3
    assert roundtripped.reason == "email detected"
    assert str(roundtripped) == str(original)
