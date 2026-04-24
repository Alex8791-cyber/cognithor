def test_public_api_exports():
    from cognithor import crew

    assert hasattr(crew, "CrewAgent")
    assert hasattr(crew, "CrewTask")
    assert hasattr(crew, "Crew")
    assert hasattr(crew, "CrewProcess")
    assert hasattr(crew, "CrewOutput")
    assert hasattr(crew, "TaskOutput")
    assert hasattr(crew, "GuardrailFailure")
    assert hasattr(crew, "ToolNotFoundError")
