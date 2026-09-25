# tests/test_agent_factory.py
import pytest
from agent_jerry_gao.factory import agent_factory

def test_agent_factory_create_invalid():
    with pytest.raises(Exception):
        agent_factory.create_agent("invalid_type")
