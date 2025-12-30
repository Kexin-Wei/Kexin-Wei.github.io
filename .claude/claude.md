# Claude Instructions

## Git Commits

Group by purpose. Use conventional commits: `feat:`, `fix:`, `docs:`, `style:`, `chore:`

```bash
# Example
git add src/layouts/*.astro && git commit -m "style: layout changes"
git add src/content/**/*.md && git commit -m "docs: content updates"
git add src/.config/*.ts && git commit -m "chore: config updates"
```
