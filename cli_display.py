"""Render streamed replies and tool activity without interleaving lines."""

import json
import re
import sys


class TurnDisplay:
    MODES = ("compact", "full", "off")

    def __init__(self, output=None, tool_output="compact"):
        self.output = output if output is not None else sys.stdout
        self.tool_output = tool_output
        self.color = self.output.isatty()
        self._thinking = False
        self._text = ""

    def _write(self, text):
        self.output.write(text)
        self.output.flush()

    def _style(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    @staticmethod
    def _clean(value):
        # Tool output must not move the cursor or inject terminal commands.
        text = str(value).replace("\r\n", "\n")
        text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        return "".join(c for c in text if c in "\n\t" or
                       (ord(c) >= 32 and not 127 <= ord(c) <= 159))

    def _clear_thinking(self):
        if self._thinking:
            self._write("\r" + " " * 40 + "\r")
            self._thinking = False

    def _wait(self):
        if self.color:
            self._write(self._style("  thinking...", "2"))
            self._thinking = True

    def begin_turn(self):
        self._text = ""
        self._wait()

    def on_text(self, text):
        if not text:
            return
        self._clear_thinking()
        if not self._text:
            self._write(self._style("hermes > ", "1;36"))
        self._write(text)
        self._text += text

    def _preview(self, value, prefix, limit):
        lines = self._clean(value).splitlines() or ["(no output)"]
        for line in lines[:limit] if limit is not None else lines:
            if limit is not None and len(line) > 160:
                line = line[:157] + "..."
            self._write(f"{prefix}{line}\n")
        if limit is not None and len(lines) > limit:
            self._write(self._style(
                f"{prefix}... {len(lines) - limit} more lines (/tools full)\n", "2"))

    def on_tool_event(self, event):
        self._clear_thinking()
        if event["type"] == "start":
            if self._text:
                self._write("\n")
                self._text = ""
            if self.tool_output == "off":
                return
            name = self._clean(event["name"]).replace("\n", " ")
            self._write(self._style(f"  > {name} [running]\n", "1;33"))
            for key, value in event["arguments"].items():
                label = self._clean(key).replace("\n", " ")
                value = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False, default=str)
                self._write(self._style(f"    {label}:\n", "2"))
                self._preview(value, "      ", 3 if self.tool_output == "compact" else None)
        elif event["type"] == "end":
            if self.tool_output == "off":
                self._wait()
                return
            self._preview(event["result"], "    | ",
                          6 if self.tool_output == "compact" else None)
            status = "done" if event["success"] else "failed"
            code = "32" if event["success"] else "31"
            self._write(self._style(
                f"    [{status}] {event['duration']:.2f}s\n\n", code))
            self._wait()

    def finish_turn(self, response):
        self._clear_thinking()
        if self._text:
            self._write("\n")
        # Errors and iteration-limit messages may follow partial streamed text.
        if response and response != self._text:
            self._write(self._style("hermes > ", "1;36") + response + "\n")
        self._write("\n")
        self._text = ""
