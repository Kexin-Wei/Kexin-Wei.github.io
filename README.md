# kexinwei.org

Personal portfolio and blog of **Kexin Wei** — R&D Engineer, Robotics & AI, Singapore.

Live at [kexinwei.org](https://kexinwei.org/).

## Stack

- [Astro 5](https://astro.build/) static site, TypeScript
- [UnoCSS](https://unocss.dev/) for styling (design tokens in `src/theme/tokens.ts`)
- Markdown content collections (`src/content/posts/`), KaTeX math, RSS feed at `/atom.xml`
- Deployed to GitHub Pages via `.github/workflows/deploy.yaml`

## Development

```bash
pnpm install
pnpm dev        # astro check + dev server
pnpm build      # astro check + production build
pnpm preview    # preview the build
pnpm post:create  # scaffold a new blog post
```

## Structure

- `src/pages/index.astro` — portfolio landing page (hero, about, skills, contact)
- `src/pages/posts/` — blog (paginated list + posts)
- `src/content/posts/*.md` — blog posts
- `src/data/profile.ts` — structured profile data for the landing page
- `src/.config/` — site config (title, nav, SEO, RSS)
- `public/` — static assets (images, `resume.pdf`, `CNAME`)

## Credits

Originally based on [astro-theme-typography](https://github.com/moeyua/astro-theme-typography) by Moeyua (MIT), since heavily refactored into a portfolio/CV design.
