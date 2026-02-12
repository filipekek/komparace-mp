
import sys
from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter

# Capture pylint output
pylint_output = StringIO()
reporter = TextReporter(pylint_output)

# Run pylint on a file
print("maker:")
for i in ["chatgpt-tetris-maker.py", "gemini-tetris-maker.py", "opus-tetris-maker.py", "sonnet-tetris-maker.py"]:
    file_name = i
    results = Run([file_name], reporter=reporter, exit=False)
    score = results.linter.stats.global_note
    print(i, str(score))

print("solver:")
for i in ["chatgpt-tetris-solver.py", "gemini-tetris-solver.py", "opus-tetris-solver.py", "sonnet-tetris-solver.py"]:
    file_name = i
    results = Run([file_name], reporter=reporter, exit=False)
    score = results.linter.stats.global_note
    print(i, str(score))