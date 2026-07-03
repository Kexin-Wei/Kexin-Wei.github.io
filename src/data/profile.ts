export interface Profile {
  name: string
  role: string
  location: string
  tagline: string
  avatar: string
  resume: string
  email: string
  skills: string[]
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
  socials: [
    { name: 'github', label: 'GitHub', href: 'https://github.com/Kexin-Wei' },
    { name: 'linkedin', label: 'LinkedIn', href: 'https://www.linkedin.com/in/kexin-wei-611s96' },
    { name: 'email', label: 'Email', href: 'mailto:weikexin611@gmail.com' },
  ],
}
