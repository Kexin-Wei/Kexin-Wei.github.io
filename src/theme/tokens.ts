/**
 * Design tokens derived from Wix template wh-1226 "Researcher CV (Minimal)".
 *
 * Visual language: Swiss-minimal. White page, near-black text, 1px hairline
 * rules between sections, giant uppercase display headline, monospace for
 * nav/labels/meta, pastel "pixel-art" accent tiles, generous vertical space.
 *
 * Measured from the live template:
 * - h1: univers-lt-std 700, uppercase, letter-spacing -0.05em
 * - h2/h3: neue-haas-grotesk-display-pro 400
 * - body: instrument-sans 500
 * - nav/meta: ibm-plex-mono 400
 * - text #1a1a1a (headings) / #3b3b3b (body), bg #ffffff
 * - accent tiles: green #ddf0d9, lavender ~#dccfea, gray ~#d6d6d6
 *
 * Font substitutions (self-hosted via @fontsource, license-free):
 * - univers-lt-std → Archivo (display)
 * - neue-haas-grotesk → Instrument Sans (shared with body)
 * - ibm-plex-mono → IBM Plex Mono (exact match, it's open source)
 */

export const colorsLight = {
  primary: '#1a1a1a',
  secondary: '#3b3b3b',
  background: '#ffffff',
  surface: '#f5f5f3',
  hairline: '#1a1a1a',
  accent: '#ddf0d9',
  accentAlt: '#dccfea',
  muted: '#8a8a8a',
  shadow: '#0000000A',
}

export const colorsDark = {
  primary: '#f5f5f3',
  secondary: '#c9c9c9',
  background: '#1a1a1a',
  surface: '#232323',
  hairline: '#f5f5f3',
  accent: '#3a4d36',
  accentAlt: '#463a52',
  muted: '#8a8a8a',
  shadow: '#FFFFFF0A',
}

export const fonts = {
  header: '"Archivo Variable", "Archivo", "Helvetica Neue", Arial, sans-serif',
  ui: '"Instrument Sans Variable", "Instrument Sans", "Helvetica Neue", Arial, sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
}
