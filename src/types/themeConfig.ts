import type { Link, Meta } from 'astro-seo'
import type { LANGUAGES } from '../i18n.ts'

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
}

export interface ThemeConfig {
  site: ConfigSite
  appearance: ConfigAppearance
  seo: ConfigSEO
  rss: ConfigRSS
  latex: ConfigLaTeX
}

export type UserConfig = DeepPartial<ThemeConfig>

export interface ConfigSite {
  title: string
  subtitle: string
  author: string
  description: string
  website: string
  pageSize: number
  socialLinks: { name: string, href: string }[]
  navLinks: { name: string, href: string, highlight?: boolean }[]
  categoryMap: { name: string, path: string }[]
  footer: string[]
}

export interface ConfigAppearance {
  theme: 'light' | 'dark' | 'system'
  locale: keyof typeof LANGUAGES
}

export interface ConfigSEO {
  twitter: string
  meta: Partial<Meta>[]
  link: Partial<Link>[]
}

export interface ConfigRSS {
  fullText?: boolean
  /** https://github.com/RSSNext/follow */
  follow?: { feedId: string, userId: string }
}

export interface ConfigLaTeX {
  katex: boolean
}
