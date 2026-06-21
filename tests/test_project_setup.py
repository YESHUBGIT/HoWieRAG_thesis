from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import IntentResult
from howie_rag.core.utils import stable_id


def test_stable_id_is_deterministic() -> None:
    text = "hello world"
    assert stable_id(text) == stable_id(text)


def test_intent_result_can_be_created() -> None:
    result = IntentResult(intent="greeting", confidence=0.95)
    assert result.intent == "greeting"
    assert result.confidence == 0.95
    assert result.reasoning is None
