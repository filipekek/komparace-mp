
import sys
from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter

# Capture pylint output
pylint_output = StringIO()
reporter = TextReporter(pylint_output)

# Run pylint on a file
for i in ["chatgpt-map-gen.py", "gemini-map-gen.py", "opus-map-gen.py", "sonnet-map-gen.py"]:
    file_name = i
    results = Run([file_name], reporter=reporter, exit=False)
    score = results.linter.stats.global_note
    print(i, str(score))
