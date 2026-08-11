---
name: commit-message
description: Write clear Conventional Commit messages from a description of code changes, especially for changes made in the course repository.
---

# Commit Message

When asked to write a commit message, follow this procedure:

1. Identify the main type of change:
   - feat: new functionality
   - fix: bug fix
   - refactor: code restructuring without changing behavior
   - docs: documentation only
   - test: tests
   - chore: maintenance or configuration

2. Write exactly one Conventional Commit message using:
   `type(scope): short description`

3. Keep the subject:
   - under 72 characters;
   - imperative and concise;
   - without a period at the end.

4. If the user provides multiple unrelated changes, ask them to split the changes into separate commits.

5. Do not invent files, behavior, tests, or fixes that were not mentioned.

6. Return only the recommended commit message unless the user explicitly asks for an explanation.

Examples:

- `feat(agent): add deep agent builder`
- `fix(agent): handle fake agent response`
- `docs(day3): add agent skills guide`
- `chore(day3): update project dependencies`

Forbidden:

- vague messages such as `update code` or `changes`
- exaggerated language such as `massive improvement`
- invented details




