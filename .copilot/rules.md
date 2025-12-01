# Copilot Code Generation Rules

These rules guide how GitHub Copilot should generate and modify code in this repository.  
Copilot should treat these as **binding defaults** unless the existing file’s style clearly differs.

---

## Logging Rules
- All logger messages must use **`%s` formatting**, _never_ f-strings or `.format()`.
  - **Good:** `logger.info("Loaded item %s", item_id)`
  - **Bad:** `logger.info(f"Loaded item {item_id}")`
- For logs involving an **exception**, use an exception object `e`:
  - Always log the exception using **`repr(e)`**
  - Always set **`exc_info=True`**
  - **Example:**
    ```python
    except Exception as e:
        logger.error("Failed to load resource: %s", repr(e), exc_info=True)
    ```

---

## Comments and Docstrings
- **Comments must be lowercase**, unless referencing a function, class, constant, or variable that is intentionally uppercase or CamelCase.
- Docstrings (module, class, function, method):
  - Must be **succinct**
  - Should describe only **purpose** and **key arguments / return values**
  - Avoid long paragraphs unless essential
- Inline comments should be short and avoid restating obvious behavior.

---

## Formatting and Style (Ruff)
- All Python code must comply with **Ruff** formatting and linting rules.
- Use **PEP 8** conventions unless Ruff enforces a stricter rule.
- Ensure imports are grouped and sorted according to Ruff’s expectations.

---

## Python Code Structure
- Prefer **pure functions** unless side effects are explicitly needed.
- Use **type hints for all functions**, including return type hints.
- Use **Pydantic models** for structured validation when appropriate.
- Avoid deep nesting; prefer early returns.
- Avoid mutable default arguments.
- Use explicit exception types whenever possible.
- Avoid broad `except:` blocks; use `except Exception:` instead.
- Keep functions focused; avoid multi-purpose large functions.

---

## Strings and Data
- Use triple-quoted strings only for docstrings; otherwise prefer double quotes consistently.
- Never include secrets, API keys, or credentials in code or comments.
- Use raw strings (`r"..."`) for regular expressions.

---

## Code Generation Priorities
When generating or modifying code, Copilot should prioritize:

1. Correctness
2. Clarity
3. Modularity and reusability
4. Maintaining repository-wide patterns defined here
5. Minimal changes when editing existing code (preserve author style)

---

## Tests
- Write tests using `pytest`.
- Prefer clear test names and explicit assertions.
- Mock external services and network calls.
- Keep tests independent and deterministic.

---

## Performance and Safety
- Avoid unnecessary CPU- or memory-heavy operations.
- Use caching only when appropriate and with clear invalidation.
- Validate external input aggressively.
- Log failures, not successes, unless success logs are important for observability.

---

## Text Generation
- Never include followup text in comments, docstrings, READMEs or code.
  - This includes any text that isn't strictly necessary, and any text that suggests further action or context.

By following these rules, Copilot should produce code that fits naturally within this project’s architecture, style, and quality expectations.