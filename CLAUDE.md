# CLAUDE

## Coding Standards
- Avoid reduncancy where possible, prefer modularity
- Keep code clean, use comments only where required
- Use pydantic classes / schemas where appropriate
- All imports must be top-level unless a circular dependency requires otherwise
- Always remove dead code
- Always keep requirements.txt updated

## Token Efficiency
- Only read what you absolutely have to - avoid reading everything in the first pass
- Keep summaries of your work short and succinct - avoid long detailed explanations of your changes

## Documentation
- Always keep all markdown files up to date - but avoid making edits to this one
- Keep documentation succinct while retaining all importatnt information - avoid re-stating information

## Prompts
- All LLM prompts must live in `app/prompts/` as Jinja2 templates, loaded via `PromptRegistry`
- Never inline prompt strings in Python code

## Git
- Never commit or push code

## User Changes
- Never remove or overwrite changes the user has manually made to any file without explicitly asking first
