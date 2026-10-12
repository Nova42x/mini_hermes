import io
import unittest
from types import SimpleNamespace

from agent import Agent
from cli_display import TurnDisplay


def chunk(text="", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=text, tool_calls=calls))])


class ToolEventsTest(unittest.TestCase):
    def make_agent(self):
        agent = Agent(None, "gpt-test", "system", [], stream=True)
        agent.configure_learning(memory_nudge=0, skill_nudge=0)
        return agent

    def test_start_is_emitted_before_handler_and_end_has_result(self):
        agent = self.make_agent()
        events = []
        agent.set_tool_callback(events.append)

        def handler(command):
            self.assertEqual(events[0]["type"], "start")
            self.assertEqual(events[0]["arguments"], {"command": command})
            self.assertEqual(events[0]["call_id"], "call_1")
            return "file.txt"

        agent.set_handlers({"terminal": handler})
        self.assertEqual(agent._execute_tool("terminal", {"command": "dir"},
                                             "call_1"), "file.txt")
        self.assertEqual(len(events), 2)
        self.assertTrue(events[1]["success"])
        self.assertEqual(events[1]["result"], "file.txt")
        self.assertGreaterEqual(events[1]["duration"], 0)

    def test_failures_emit_end_events(self):
        agent = self.make_agent()

        def broken():
            raise ValueError("bad path")

        agent.set_handlers({"broken": broken,
                            "read_file": lambda: "Error: file not found"})
        for name in ("missing", "broken", "read_file"):
            with self.subTest(name=name):
                events = []
                agent.set_tool_callback(events.append)
                result = agent._execute_tool(name, {})
                self.assertEqual([e["type"] for e in events], ["start", "end"])
                self.assertFalse(events[-1]["success"])
                self.assertEqual(events[-1]["result"], result)

    def test_display_failure_does_not_block_execution(self):
        agent = self.make_agent()

        def broken_display(event):
            raise RuntimeError("display failed")

        agent.set_tool_callback(broken_display)
        agent.set_handlers({"ok": lambda: "ok"})
        with self.assertLogs("agent", level="ERROR"):
            self.assertEqual(agent._execute_tool("ok", {}), "ok")

    def test_streamed_tool_round_trip_in_all_display_modes(self):
        for mode in TurnDisplay.MODES:
            with self.subTest(mode=mode):
                tool_delta = SimpleNamespace(index=0, id="call_1",
                    function=SimpleNamespace(name="terminal", arguments='{"command":'))
                arg_delta = SimpleNamespace(index=0, id=None,
                    function=SimpleNamespace(name=None, arguments='"dir"}'))
                streams = iter([
                    iter([chunk("Checking files."), chunk(calls=[tool_delta]),
                          chunk(calls=[arg_delta])]),
                    iter([chunk("Found file.txt.")]),
                ])
                agent = self.make_agent()
                agent.client = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kw: next(streams))))
                agent.set_handlers({"terminal": lambda command: "file.txt"})
                output = io.StringIO()
                display = TurnDisplay(output, mode)
                agent.set_stream_callback(display.on_text)
                agent.set_tool_callback(display.on_tool_event)
                display.begin_turn()
                response = agent.run("list files")
                display.finish_turn(response)
                rendered = output.getvalue()
                self.assertEqual(response, "Found file.txt.")
                self.assertEqual(rendered.count("Found file.txt."), 1)
                self.assertIn("Checking files.\n", rendered)
                self.assertEqual(agent.messages[-2]["role"], "tool")
                self.assertEqual(agent.messages[-2]["content"], "file.txt")
                if mode == "off":
                    self.assertNotIn("[running]", rendered)
                else:
                    self.assertIn("terminal [running]", rendered)
                    self.assertIn("[done]", rendered)
                    self.assertLess(rendered.index("[running]"), rendered.index("[done]"))
                    self.assertLess(rendered.index("[done]"), rendered.index(response))


class ToolDisplayTest(unittest.TestCase):
    def event(self, result, success=True):
        return {"type": "end", "name": "terminal", "arguments": {},
                "result": result, "duration": 0.25, "success": success}

    def test_compact_preview_is_bounded_and_full_preserves_lines(self):
        result = "\n".join(f"line {i}" for i in range(12))
        output = io.StringIO()
        display = TurnDisplay(output)
        display.on_tool_event(self.event(result))
        self.assertIn("6 more lines", output.getvalue())
        self.assertNotIn("line 11", output.getvalue())
        output = io.StringIO()
        display = TurnDisplay(output, "full")
        display.on_tool_event(self.event(result))
        self.assertIn("line 11", output.getvalue())
        self.assertNotIn("more lines", output.getvalue())

    def test_terminal_controls_are_removed_from_preview(self):
        output = io.StringIO()
        display = TurnDisplay(output)
        display.on_tool_event(self.event("\033[31mred\033[0m\r\x07"))
        self.assertIn("red", output.getvalue())
        self.assertNotIn("\033", output.getvalue())
        self.assertNotIn("\r", output.getvalue())

    def test_partial_reply_does_not_hide_error(self):
        output = io.StringIO()
        display = TurnDisplay(output)
        display.begin_turn()
        display.on_text("Partial reply")
        display.finish_turn("[Error: LLM call failed]")
        self.assertIn("Partial reply\nhermes > [Error: LLM call failed]", output.getvalue())

    def test_non_streaming_final_reply_and_failed_tool(self):
        output = io.StringIO()
        display = TurnDisplay(output)
        display.begin_turn()
        display.on_tool_event({"type": "start", "name": "read_file",
                               "arguments": {"path": "missing.txt"}})
        display.on_tool_event(self.event("Error: file not found", False))
        display.finish_turn("Could not read file.")
        self.assertIn("[failed] 0.25s", output.getvalue())
        self.assertIn("hermes > Could not read file.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
