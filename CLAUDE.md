# CLAUDE.md - Environment & Style Guide

## Environment
- Project uses conda environment: `conda create -n transformers-project python=3.10`
- Activate with: `conda activate transformers-project`
- Install dependencies: `pip install -r src/tasks/requirements.txt`
- Run examples: `python src/tasks/<task_type>/<task_folder>/example1.py`

## Code Style Guidelines
- **Imports**: Standard lib first, then third-party, grouped logically
- **Formatting**: 4-space indentation, use f-strings for formatting
- **Functions**: Use snake_case, descriptive names
- **Variables**: Descriptive names, consistent conventions
- **Type Hints**: Not currently used but encouraged for new code
- **Error Handling**: Use try/except for expected exceptions
- **Comments**: Add descriptive comments explaining functionality
- **Pattern**: Wrap main code in `main()` function with `if __name__ == "__main__"` guard
- **Model Usage**: Follow transformers pipeline pattern where possible
- **File Structure**: Group tasks by domain (nlp/vision) and subtask