
import sys
from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter

# Capture pylint output
pylint_output = StringIO()
reporter = TextReporter(pylint_output)

# Run pylint on a file
print("maker:")
for i in ["chatgpt-maze-maker.py", "gemini-maze-maker.py", "opus-maze-maker.py", "sonnet-maze-maker.py"]:
    file_name = i
    results = Run([file_name], reporter=reporter, exit=False)
    score = results.linter.stats.global_note
    print(i, str(score))

print("solver:")
for i in ["chatgpt-maze-solver.py", "gemini-maze-solver.py", "opus-maze-solver.py", "sonnet-maze-solver.py"]:
    file_name = i
    results = Run([file_name], reporter=reporter, exit=False)
    score = results.linter.stats.global_note
    print(i, str(score))