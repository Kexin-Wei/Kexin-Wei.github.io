export interface FocusArea {
  title: string
  description: string
  accent: 'accent' | 'accent-alt' | 'surface'
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
  role: 'R&D Engineer, Robotics & AI',
  location: 'Singapore',
  tagline: 'Building intelligent robotic systems — from perception to motion.',
  avatar: '/me.jpeg',
  resume: '/resume.pdf',
  email: 'weikexin611@gmail.com',
  skills: [
    'Robotics',
    'AI',
    'LLM',
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
    },
    {
      title: 'AI & Machine Learning',
      description:
        'Applying reinforcement learning, computer vision and LLM agents to real-world robotic tasks — from perception pipelines to decision-making policies.',
      accent: 'accent-alt',
    },
    {
      title: 'Software & Systems',
      description:
        'Designing maintainable C++ and Python systems. Strong focus on software architecture, tooling and the engineering discipline that turns research into products.',
      accent: 'surface',
    },
  ],
  socials: [
    { name: 'github', label: 'GitHub', href: 'https://github.com/Kexin-Wei' },
    { name: 'linkedin', label: 'LinkedIn', href: 'https://www.linkedin.com/in/kexin-wei-611s96' },
    { name: 'email', label: 'Email', href: 'mailto:weikexin611@gmail.com' },
  ],
}
