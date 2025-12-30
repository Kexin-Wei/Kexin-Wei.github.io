import type { UserConfig } from '~/types'

export const userConfig: Partial<UserConfig> = {
  site: {
    title: 'Kexin\'s Blog',
    subtitle: 'Tech Notes',
    author: 'Kexin Wei',
    description: 'Personal blog about RL, ROS, LLM and more',
    website: 'https://kexinwei.org/',
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
        name: 'Posts',
        href: '/',
      },
      {
        name: 'Archive',
        href: '/archive/',
      },
      {
        name: 'Tags',
        href: '/categories',
      },
      { name: 'Logseq Notes', href: '/logseq_notes/' },
      {
        name: 'About',
        href: '/about',
      },
    ],
  },
  appearance: {
    locale: 'en-us',
  },
  latex: {
    katex: true,
  },
}
