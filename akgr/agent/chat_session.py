from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnRecord:
    user_text: str
    parsed_control: Optional[Dict[str, Any]] = None
    model_input: Optional[Dict[str, Any]] = None
    model_output: Optional[Dict[str, Any]] = None
    final_response: Optional[str] = None


@dataclass
class ChatSession:
    session_id: str
    history: List[TurnRecord] = field(default_factory=list)
    user_inputs: List[str] = field(default_factory=list)
    last_observation_entities: List[str] = field(default_factory=list)
    last_condition_type: Optional[str] = None
    last_condition_value: Optional[str] = None
    last_conditions: List[Dict[str, Any]] = field(default_factory=list)
    last_parsed_control: Optional[Dict[str, Any]] = None
    last_hypothesis: Optional[str] = None

    def add_turn(self, turn: TurnRecord) -> None:
        self.history.append(turn)
        self.user_inputs.append(turn.user_text)

    def reset_context(self) -> None:
        """Clear previous parsed JSON and conversational memory for a new question."""
        self.history.clear()
        self.user_inputs.clear()
        self.last_observation_entities = []
        self.last_condition_type = None
        self.last_condition_value = None
        self.last_conditions = []
        self.last_parsed_control = None
        self.last_hypothesis = None

    def update_memory(self, parsed_control: Dict[str, Any], model_output: Dict[str, Any]) -> None:
        self.last_parsed_control = parsed_control
        obs = parsed_control.get("observation_entities", [])
        if obs:
            self.last_observation_entities = obs

        conditions = parsed_control.get("conditions", [])
        if isinstance(conditions, list):
            normalized_conditions = [item for item in conditions if isinstance(item, dict)]
            self.last_conditions = normalized_conditions
            if normalized_conditions:
                self.last_condition_type = normalized_conditions[0].get("type")
                self.last_condition_value = normalized_conditions[0].get("value")
            else:
                self.last_condition_type = parsed_control.get("condition_type")
                self.last_condition_value = parsed_control.get("condition_value")
        else:
            self.last_condition_type = parsed_control.get("condition_type")
            self.last_condition_value = parsed_control.get("condition_value")
            self.last_conditions = []
        self.last_hypothesis = model_output.get("hypothesis_text")