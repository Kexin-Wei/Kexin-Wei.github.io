import type { UserConfig } from '~/types'

export const userConfig: Partial<UserConfig> = {
  site: {
    title: "Kristin's Blog",
    subtitle: 'Tech Notes',
    author: 'Kristin Wei',
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
  },
  appearance: {
    locale: 'en-us',
  },
  latex: {
    katex: true,
  },
}
