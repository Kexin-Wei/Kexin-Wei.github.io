export type PixelIcon = 'robot' | 'gear' | 'chip' | 'wave' | 'node' | 'code' | 'lock'

export interface FocusArea {
  title: string
  description: string
  accent: 'accent' | 'accent-alt' | 'accent-blue' | 'surface'
  icons: PixelIcon[]
}

export interface Project {
  title: string
  description: string
  tags: string[]
  href: string
  area: 'robotics' | 'ai' | 'systems'
  stars?: number
  image?: string
}

export interface Involvement {
  title: string
  description: string
  href?: string
  icons?: PixelIcon[]
  logo?: string
  logoBg?: 'dark' // white/transparent logos need a dark chip to be visible
}

export interface Profile {
  name: string
  role: string
  location: string
  tagline: string
  avatar: string
  resume: string
  email: string
  emailAcademic: string
  skills: string[]
  focusAreas: FocusArea[]
  projects: Project[]
  openSource: Involvement[]
  community: Involvement[]
  socials: { name: string, label: string, href: string }[]
}

export const profile: Profile = {
  name: 'Kexin Wei',
  role: 'R&D Engineer — Robotics, AI & Cybersecurity',
  location: 'Singapore',
  tagline: 'Building intelligent systems that move, reason, and stay secure.',
  avatar: '/portrait.jpg',
  resume: '/Kexin%20Wei%20Resume.pdf',
  email: 'weikexin611@gmail.com',
  emailAcademic: 'k.wei@imperial.ac.uk',
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
      accent: 'accent-blue',
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
  projects: [
    {
      title: 'Soft-Tissue Deformation for Laparoscopic Surgery (DRL)',
      description:
        'M.Eng thesis: a deep-RL agent (SpinningUp) drives surgical instruments through soft-tissue tasks in a CoppeliaSim training simulation.',
      tags: ['Deep RL', 'CoppeliaSim', 'Python'],
      href: 'https://github.com/Kexin-Wei/Soft-tissue-deformation-in-laparoscopic-surgery-planning-with-DRL',
      area: 'robotics',
      stars: 6,
      image: '/projects/soft-tissue-drl.jpg',
    },
    {
      title: 'RAG & LLM Agent',
      description:
        'Retrieval-augmented generation and an agentic LLM workflow with tool use and multi-step planning.',
      tags: ['RAG', 'LLM Agent', 'Python'],
      href: 'https://github.com/Kexin-Wei/RAG-LLM-and-LLM-Agent',
      area: 'ai',
      image: '/projects/rag-agent.jpg',
    },
    {
      title: 'Medical Organ Surface Editing (VTK)',
      description:
        'Label a volumetric organ (e.g. kidney), edit its surface with a NURBS UI, and reconstruct the mesh in VTK for visualization.',
      tags: ['VTK', 'NURBS', 'C++'],
      href: 'https://github.com/Kexin-Wei/VTKMedicalOrganSurfaceManipulate_Demo',
      area: 'systems',
      image: '/projects/medical-vtk.jpg',
    },
    {
      title: 'UFactory Lite 6 Control',
      description:
        'C++ experiments controlling the UFactory Lite 6 collaborative robot arm.',
      tags: ['C++', 'Robotics'],
      href: 'https://github.com/Kexin-Wei/lite6Robot_demo',
      area: 'robotics',
      image: '/projects/lite6.jpg',
    },
    {
      title: 'Qt C++ ↔ Python Imaging',
      description:
        'Bridge a Qt C++ application with Python image-processing scripts and display the results.',
      tags: ['Qt', 'C++', 'Python'],
      href: 'https://github.com/Kexin-Wei/QTCpp_call_PythonScript_DisplayImage_Demo',
      area: 'systems',
      stars: 2,
      image: '/projects/qt-python.jpg',
    },
    {
      title: 'AI SDK Comparison',
      description:
        'Side-by-side comparison of LLM SDKs across providers — Claude, OpenAI, and Gemini.',
      tags: ['Claude', 'OpenAI', 'Gemini'],
      href: 'https://github.com/Kexin-Wei/ai-sdk-comparison',
      area: 'ai',
      image: '/projects/ai-sdk.jpg',
    },
    {
      title: 'ROS Workspace & Demos',
      description:
        'A ROS workspace of hands-on demos built while working through ROS robotics coursework.',
      tags: ['ROS', 'Docker', 'Robotics'],
      href: 'https://github.com/Kexin-Wei/ROS-demo',
      area: 'robotics',
      image: '/projects/ros.jpg',
    },
    {
      title: 'Robot Kinematics & Control',
      description:
        'Worked demos and notes on robot kinematics, dynamics, visualization, and control.',
      tags: ['Kinematics', 'Dynamics', 'Python'],
      href: 'https://github.com/Kexin-Wei/Robotics-Demo',
      area: 'robotics',
      stars: 1,
      image: '/projects/robotics-kinematics.jpg',
    },
  ],
  openSource: [
    {
      title: 'Women Devs SG — community website',
      description: 'Contributed a merged feature to the Women Devs SG site.',
      href: 'https://github.com/Women-Devs-SG/womendevssg/pull/77',
      icons: ['code', 'node'],
      logo: '/community/women-devs-sg.png',
    },
    {
      title: 'bibsnbub — childcare-finder app',
      description: 'Contributed to Women Devs SG’s bibsnbub app (facility-name formatting).',
      href: 'https://github.com/Women-Devs-SG/bibsnbub/pull/40',
      icons: ['code', 'chip'],
      logo: '/community/women-devs-sg.png',
    },
  ],
  community: [
    {
      title: 'Women Devs SG',
      description: 'Member and open-source contributor to the Women Devs SG community.',
      href: 'https://github.com/Women-Devs-SG',
      icons: ['node', 'wave'],
      logo: '/community/women-devs-sg.png',
    },
    {
      title: 'Junior Developers SG',
      description: 'Coding dojos — hands-on TDD, refactoring, and test-automation practice.',
      href: 'https://juniordev.sg',
      icons: ['code', 'gear'],
      logo: '/community/juniordev.png',
    },
    {
      title: 'Division Zero (div0)',
      description: 'Cybersecurity workshops — Building Your Own Home Lab, SHELLgym, and Hack Your First Flag.',
      href: 'https://www.div0.sg',
      icons: ['lock', 'chip'],
      logo: '/community/div0.png',
      logoBg: 'dark',
    },
  ],
  socials: [
    { name: 'github', label: 'GitHub', href: 'https://github.com/Kexin-Wei' },
    { name: 'linkedin', label: 'LinkedIn', href: 'https://www.linkedin.com/in/kexin-wei-research/' },
    { name: 'email', label: 'Email', href: 'mailto:weikexin611@gmail.com' },
  ],
}
