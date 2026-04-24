from __future__ import annotations

from dataclasses import dataclass


class CrewError(Exception):
    """Base class for every Crew-Layer error."""


class CrewCompilationError(CrewError):
    """Raised when the Compiler cannot translate a Crew into PGE inputs."""


class ToolNotFoundError(CrewError):
    """Raised when an agent references a tool the registry does not expose."""


@dataclass
class GuardrailFailure(CrewError):
    """Raised when a guardrail rejects output after exhausting retries.

    The message says "after N attempt(s)" where N is the actual number of
    attempts made (initial try + retries). Avoids the "max_retries" off-by-one
    surprise where max_retries=2 meant 3 attempts.

    Includes a custom ``__reduce__`` so the exception can be pickled and
    unpickled correctly — a plain ``@dataclass`` subclass of Exception fails
    under ``ProcessPoolExecutor`` / ``multiprocessing.Queue`` / Celery because
    the dataclass-generated ``__init__`` signature does not match the
    single-arg unpickle path ``Exception.__init__`` uses by default.
    """

    task_id: str
    guardrail_name: str
    attempts: int
    reason: str

    def __str__(self) -> str:
        return (
            f"Guardrail '{self.guardrail_name}' rejected output from task "
            f"'{self.task_id}' after {self.attempts} attempt(s): {self.reason}"
        )

    def __post_init__(self) -> None:
        # Keep Exception.args in sync so stack traces show a useful repr.
        super().__init__(str(self))

    def __reduce__(self) -> tuple:
        """Support pickling across process boundaries.

        Without this, ``pickle.dumps(GuardrailFailure(...))`` succeeds but
        ``pickle.loads(...)`` raises a misleading
        ``TypeError: __init__() missing 3 required positional arguments``
        because Exception's default unpickle path calls ``__init__`` with a
        single positional arg (``self.args[0]``) which doesn't match the
        dataclass signature.
        """
        return (
            self.__class__,
            (self.task_id, self.guardrail_name, self.attempts, self.reason),
        )
