"""Tests for native tool-format conversion in 1_dataset_generator.py (Exp 1)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dataset_generator",
    Path(__file__).parent.parent / "1_dataset_generator.py",
)
dg = importlib.util.module_from_spec(_spec)
sys.modules["dataset_generator"] = dg
_spec.loader.exec_module(dg)


class TestParseXmlTool:
    def test_python_execute_code_with_inner_quotes(self):
        name, args = dg._parse_xml_tool("python_execute(code='result = 42\\nprint(f\"x={result}\")')")
        assert name == "python_execute"
        assert args["code"].startswith("result = 42")
        assert "print(" in args["code"]

    def test_exchange_kwargs(self):
        name, args = dg._parse_xml_tool("get_exchange_rate(from='USD', to='EUR')")
        assert name == "get_exchange_rate"
        assert args == {"from": "USD", "to": "EUR"}

    def test_unquoted_arg(self):
        name, args = dg._parse_xml_tool("update_profile(name='Sam', age=30, occupation='nurse')")
        assert name == "update_profile"
        assert args["name"] == "Sam" and args["age"] == "30" and args["occupation"] == "nurse"

    def test_no_args(self):
        name, args = dg._parse_xml_tool("get_user_profile()")
        assert name == "get_user_profile" and args == {}


class TestRemapTool:
    def test_exchange_to_web_search(self):
        n, a = dg._remap_tool("get_exchange_rate", {"from": "USD", "to": "EUR"})
        assert n == "web_search"
        assert "USD" in a["query"] and "EUR" in a["query"]

    def test_get_profile_to_memory_read(self):
        n, a = dg._remap_tool("get_user_profile", {})
        assert n == "user_memory_read" and "prompt" in a

    def test_update_profile_to_memory_update(self):
        n, a = dg._remap_tool("update_profile", {"name": "Sam", "age": "30"})
        assert n == "user_memory_update" and a["section"] == "profile"
        assert "Sam" in a["content"]

    def test_sentiment_to_python(self):
        n, a = dg._remap_tool("analyze_sentiment", {"text": "I feel great"})
        assert n == "python_execute" and "code" in a

    def test_real_tool_passthrough(self):
        n, a = dg._remap_tool("python_execute", {"code": "x=1"})
        assert n == "python_execute" and a == {"code": "x=1"}


class TestToNative:
    def _example(self):
        return {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "convert 100 USD to EUR"},
                {"role": "assistant",
                 "content": "<think>need a rate</think>\n<tool>get_exchange_rate(from='USD', to='EUR')</tool>"},
                {"role": "tool", "content": '{"rate": 0.92}'},
                {"role": "assistant", "content": "<answer>about 92 EUR</answer>"},
            ],
            "metadata": {"model_variant": "interleaved"},
        }

    def test_assistant_tool_converted_and_remapped(self):
        g = dg.DatasetGenerator({}, tool_format="native")
        out = g._to_native(self._example())
        asst = out["messages"][2]["content"]
        assert "<tool>" not in asst and "<tool_call>" in asst
        m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", asst, re.DOTALL)
        call = json.loads(m.group(1))
        assert call["name"] == "web_search"  # get_exchange_rate remapped
        assert "query" in call["arguments"]

    def test_tool_result_wrapped(self):
        g = dg.DatasetGenerator({}, tool_format="native")
        out = g._to_native(self._example())
        assert out["messages"][3]["content"].startswith("[TOOL_RESULT: web_search]")

    def test_metadata_stamped(self):
        g = dg.DatasetGenerator({}, tool_format="native")
        out = g._to_native(self._example())
        assert out["metadata"]["tool_format"] == "native"
        assert out["metadata"]["tool_profile"] == "all_tools"
        assert len(out["metadata"]["native_tools"]) > 0
        # served registry schema shape
        assert out["metadata"]["native_tools"][0]["type"] == "function"

    def test_no_fictional_tool_names_survive(self):
        g = dg.DatasetGenerator({}, tool_format="native")
        out = g._to_native(self._example())
        served = {"python_execute", "web_search", "read_url", "get_datetime",
                  "scratchpad_sections", "scratchpad_read", "scratchpad_update",
                  "user_memory_sections", "user_memory_read", "user_memory_update"}
        for msg in out["messages"]:
            for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", msg["content"], re.DOTALL):
                assert json.loads(m.group(1))["name"] in served
