export type PixelIcon = 'robot' | 'gear' | 'chip' | 'wave' | 'node' | 'code' | 'lock'

export interface FocusArea {
  title: string
  description: string
  accent: 'accent' | 'accent-alt' | 'surface'
  icons: PixelIcon[]
}

export interface Profile {
  name: string
  role: string
  location: string
  tagline: string
  avatar: string
  resume: string
  email: string
  skills: string[]
  focusAreas: FocusArea[]
  socials: { name: string, label: string, href: string }[]
}

export const profile: Profile = {
  name: 'Kexin Wei',
  role: 'R&D Engineer — Robotics, AI & Cybersecurity',
  location: 'Singapore',
  tagline: 'Building intelligent systems that move, reason and stay secure.',
  avatar: '/profile.jpg',
  resume: '/resume.pdf',
  email: 'weikexin611@gmail.com',
  skills: [
    'Robotics',
    'AI',
    'LLM',
    'Cybersecurity',
    'C++',
    'Computer Vision',
    'ROS',
    'Software Design',
    'Simulation',
    'Web Dev',
  ],
  focusAreas: [
    {
      title: 'Robotics & Simulation',
      description:
        'Developing robotic systems with ROS 2, motion planning and navigation. Building simulation environments that close the gap between virtual prototyping and physical deployment.',
      accent: 'accent',
      icons: ['robot', 'gear', 'wave'],
    },
    {
      title: 'AI & LLM Agents',
      description:
        'Applying reinforcement learning, computer vision and LLM agents to real-world tasks — from perception pipelines and decision-making policies to agentic workflows.',
      accent: 'accent-alt',
      icons: ['node', 'chip', 'wave'],
    },
    {
      title: 'Cybersecurity & Systems',
      description:
        'Bringing security thinking to intelligent systems — secure software design, threat awareness and the engineering discipline that keeps robots and AI agents trustworthy.',
      accent: 'surface',
      icons: ['lock', 'code', 'chip'],
    },
  ],
  socials: [
    { name: 'github', label: 'GitHub', href: 'https://github.com/Kexin-Wei' },
    { name: 'linkedin', label: 'LinkedIn', href: 'https://www.linkedin.com/in/kexin-wei-611s96' },
    { name: 'email', label: 'Email', href: 'mailto:weikexin611@gmail.com' },
  ],
}
