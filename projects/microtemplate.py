import argparse
import re
import sys
from pathlib import Path
from typing import Mapping, TextIO, Tuple


class TemplateEngine:
    _statement_re = re.compile(r"\{\% (\w+)(.*) \%\}")

    def __init__(self, variables: Mapping[str, str]) -> None:
        self._variables = variables
        self._global_line_counter = 0

    def process(self, source: TextIO, sink: TextIO) -> None:
        while True:
            line = self._read_line(source)
            if not line:
                break

            match = self._statement_re.match(line)
            if match is not None:
                statement = match.group(1)
                condition = match.group(2)

                if statement == "if":
                    self._process_if_statement(condition, source, sink)
                else:
                    self._raise_parsing_error(f"'{statement}' unsupported statement")
            else:
                sink.write(line)

    def _process_if_statement(self, condition: str, source: TextIO, sink: TextIO) -> None:
        condition_result = self._evaluate_condition(condition)

        while True:
            line = self._read_line(source)
            if not line:
                self._raise_parsing_error("expected closing statement for 'if' clause")

            match = self._statement_re.match(line)
            if match is not None:
                statement = match.group(1)
                if statement == "endif":
                    break
                else:
                    self._raise_parsing_error("nested statements not supported yet")

            if condition_result:
                sink.write(line)

    def _evaluate_condition(self, condition: str) -> bool:
        return bool(eval(condition, {}, self._variables))

    def _raise_parsing_error(self, message: str) -> None:
        raise ValueError(f"Line {self._global_line_counter}: {message}")

    def _read_line(self, source: TextIO) -> str:
        line = source.readline()
        if line:
            self._global_line_counter += 1
        return line


def process(config) -> None:
    def parse_param(name_value: str) -> Tuple[str, str]:
        name, value = [x.strip() for x in name_value.split("=")]
        return (name, value)

    params = config.param or []
    variables = dict(parse_param(param) for param in params)

    engine = TemplateEngine(variables)
    with open(config.filename, "r") as file:
        engine.process(file, sys.stdout)


def main():
    parser = argparse.ArgumentParser("Minimalistic template engine with Jinja2-like syntax")
    parser.add_argument("filename", type=Path, help="Path to input file")
    parser.add_argument("-p", "--param", type=str, action="append", help="Variables and their values")
    process(parser.parse_args())


if __name__ == "__main__":
    main()
