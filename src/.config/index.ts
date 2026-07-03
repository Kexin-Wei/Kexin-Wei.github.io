import type { ThemeConfig } from '~/types'

export const themeConfig: ThemeConfig = {
  site: {
    title: 'Kexin Wei',
    subtitle: 'Robotics & AI',
    author: 'Kexin Wei',
    description: 'Kexin Wei — R&D Engineer in Robotics & AI. Portfolio and notes on RL, ROS, LLM and more.',
    website: 'https://kexinwei.org/',
    pageSize: 5,
    socialLinks: [
      {
        name: 'github',
        href: 'https://github.com/Kexin-Wei',
      },
      {
        name: 'rss',
        href: '/atom.xml',
      },
    ],
    navLinks: [
      {
        name: 'Home',
        href: '/',
      },
      {
        name: 'Posts',
        href: '/posts/',
      },
      {
        name: 'Resume',
        href: '/resume.pdf',
        highlight: true,
      },
    ],
    categoryMap: [],
    footer: [
      '© %year <a target="_blank" href="%website">%author</a>',
    ],
  },
  appearance: {
    theme: 'system',
    locale: 'en-us',
  },
  seo: {
    twitter: '',
    meta: [],
    link: [],
  },
  rss: {
    fullText: true,
  },
  latex: {
    katex: true,
  },
}
