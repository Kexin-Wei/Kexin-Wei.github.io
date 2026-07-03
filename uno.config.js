import presetAttributify from '@unocss/preset-attributify'
import transformerDirectives from '@unocss/transformer-directives'
import {
  defineConfig,
  presetIcons,
  presetTypography,
  presetWind3,
  transformerVariantGroup,
} from 'unocss'
import presetTheme from 'unocss-preset-theme'
import { themeConfig } from './src/.config'
import { colorsDark, colorsLight, fonts } from './src/theme/tokens'

const cssExtend = {
  ':root': {
    '--prose-borders': '#eee',
  },

  'code::before,code::after': {
    content: 'none',
  },

  ':where(:not(pre):not(a) > code)': {
    'white-space': 'normal',
    'word-wrap': 'break-word',
    'padding': '2px 4px',
    'color': '#c7254e',
    'font-size': '90%',
    'background-color': '#f9f2f4',
    'border-radius': '4px',
  },

  'li': {
    'white-space': 'normal',
    'word-wrap': 'break-word',
  },
}

export default defineConfig({
  rules: [
    [
      /^row-(\d+)-(\d)$/,
      ([, start, end]) => ({ 'grid-row': `${start}/${end}` }),
    ],
    [
      /^col-(\d+)-(\d)$/,
      ([, start, end]) => ({ 'grid-column': `${start}/${end}` }),
    ],
    [
      /^scrollbar-hide$/,
      ([_]) => `.scrollbar-hide { scrollbar-width:none;-ms-overflow-style: none; }
      .scrollbar-hide::-webkit-scrollbar {display:none;}`,
    ],
  ],
  presets: [
    presetWind3(),
    presetTypography({ cssExtend }),
    presetAttributify(),
    presetIcons({ scale: 1.2, warn: true }),
    presetTheme({
      theme: {
        dark: {
          colors: { ...colorsDark },
        },
      },
    }),
  ],
  theme: {
    colors: { ...colorsLight },
    fontFamily: fonts,
  },
  shortcuts: [
    ['post-title', 'text-5 font-bold lh-7.5 m-0'],
    // 1px hairline rule between sections, the template's core visual device
    ['rule-t', 'border-t-1 border-t-solid border-t-hairline/25'],
    ['mono-label', 'font-mono text-3.25 tracking-wide c-primary'],
    ['mono-link', 'font-mono text-3.5 c-primary underline underline-offset-4 decoration-1 hover:op-60 transition-opacity'],
  ],
  transformers: [transformerDirectives(), transformerVariantGroup()],
  safelist: [
    ...themeConfig.site.socialLinks.map(social => `i-mdi-${social.name}`),
    'i-mdi-github',
    'i-mdi-linkedin',
    'i-mdi-email',
    'i-mdi-rss',
    'i-mdi-map-marker',
    'i-mdi-content-copy',
    'i-mdi-check',
  ],
})
