# Contribution Guide

Authors fork and submit pull requests. The project owner will review and merge them.

Changes does not need to be rebased, you can keep every history. But try to follow below constraints for commit messages:

1. Prefix it with at least one of these words:
    - `[Tiny]` for trivial modification.
    - `[Deps]` when commits contain dependency variation.
    - `[Bugs] fixed ...` when you fixed one or more bugs.
    - `[Bugs] tbfix ...` when you discovered bugs but not jet fixed. Add `TODO` tags and descriptions in the code comments.
    - `[Feat]` when a new feature is basically/fully implemented, or behavior changed.
    - `[Perf]` when a performance related issue is handled.
    - `[Docs]` when only documentation is modified. Note that
        - You should modify `tutorials` and `devnotes` in `docs` folder directly, but modify `references` in source code docstring following Google docstring style, as it is automatically generated using `lazydocs`.
        - When modified docstring, commit python files and markdown files seperately. You can denote markdown commit with "[Docs] regeneration."
    - `[Misc]` for other cases.
2. After the prefix, describe the details with concise but meaningful words.
3. Commits can be both small and large, but try to always accomplish one concise and meaningful thing.
