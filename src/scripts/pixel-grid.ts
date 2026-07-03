/**
 * Animated halftone field replicating the Wix template's looping videos.
 *
 * Studied from the original (41s loop, sampled at 0.5s steps): the density
 * field redistributes completely within ~2s — regions flood in and drain
 * out — and every cell's SYMBOL is a function of its current density level,
 * climbing a ladder (solid square → shrinking square → diamond → dot →
 * themed icons → cross → speck → empty) as the field moves through it.
 *
 * Two noise layers moving in different directions interfere, so shapes
 * grow and decay rather than merely translating. Initializes every
 * `[data-pixel-grid]` canvas.
 */

const SIZE = 12
const STEP = SIZE + 4
const FPS = 12

// Deterministic 2D integer hash → [0, 1)
function hash(x: number, y: number, seed: number): number {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 2246822519)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t)
}

// Value noise: hashed lattice + smoothed bilinear interpolation
function noise(u: number, v: number, seed: number): number {
  const iu = Math.floor(u)
  const iv = Math.floor(v)
  const fu = smoothstep(u - iu)
  const fv = smoothstep(v - iv)
  const a = hash(iu, iv, seed)
  const b = hash(iu + 1, iv, seed)
  const c = hash(iu, iv + 1, seed)
  const d = hash(iu + 1, iv + 1, seed)
  return a + (b - a) * fu + (c - a) * fv + (a - b - c + d) * fu * fv
}

type IconDrawer = (ctx: CanvasRenderingContext2D) => void

function stroke(ctx: CanvasRenderingContext2D, width: number, draw: () => void) {
  ctx.beginPath()
  draw()
  ctx.lineWidth = width
  ctx.stroke()
}

function dot(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
}

function line(ctx: CanvasRenderingContext2D, segments: number[][]) {
  for (const [x1, y1, x2, y2] of segments) {
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
  }
}

const ICONS: Record<string, IconDrawer> = {
  robot: (ctx) => {
    ctx.beginPath()
    if (ctx.roundRect)
      ctx.roundRect(2, 4.5, 8, 6, 1)
    else
      ctx.rect(2, 4.5, 8, 6)
    ctx.lineWidth = 1.2
    ctx.stroke()
    ctx.fillRect(4, 6.6, 1.4, 1.6)
    ctx.fillRect(6.6, 6.6, 1.4, 1.6)
    stroke(ctx, 1.2, () => line(ctx, [[6, 4.5, 6, 2.4]]))
    dot(ctx, 6, 1.7, 0.9)
  },
  gear: (ctx) => {
    ctx.beginPath()
    ctx.arc(6, 6, 2.1, 0, Math.PI * 2)
    ctx.lineWidth = 1.3
    ctx.stroke()
    stroke(ctx, 1.3, () => line(ctx, [
      [6, 1, 6, 2.6],
      [6, 9.4, 6, 11],
      [1, 6, 2.6, 6],
      [9.4, 6, 11, 6],
      [2.5, 2.5, 3.6, 3.6],
      [8.4, 8.4, 9.5, 9.5],
      [9.5, 2.5, 8.4, 3.6],
      [3.6, 8.4, 2.5, 9.5],
    ]))
  },
  chip: (ctx) => {
    ctx.lineWidth = 1.2
    ctx.strokeRect(3, 3, 6, 6)
    stroke(ctx, 1, () => line(ctx, [
      [4.5, 3, 4.5, 1],
      [7.5, 3, 7.5, 1],
      [4.5, 11, 4.5, 9],
      [7.5, 11, 7.5, 9],
      [3, 4.5, 1, 4.5],
      [3, 7.5, 1, 7.5],
      [11, 4.5, 9, 4.5],
      [11, 7.5, 9, 7.5],
    ]))
  },
  wave: (ctx) => {
    ctx.beginPath()
    ctx.moveTo(0.5, 6)
    ctx.quadraticCurveTo(3, 1.5, 6, 6)
    ctx.quadraticCurveTo(9, 10.5, 11.5, 6)
    ctx.lineWidth = 1.3
    ctx.stroke()
  },
  node: (ctx) => {
    dot(ctx, 2.3, 2.6, 1.4)
    dot(ctx, 2.3, 9.4, 1.4)
    dot(ctx, 9.4, 6, 1.6)
    stroke(ctx, 1, () => line(ctx, [
      [3.5, 3.3, 8, 5.4],
      [3.5, 8.7, 8, 6.6],
    ]))
  },
  code: (ctx) => {
    ctx.beginPath()
    ctx.moveTo(4, 2.5)
    ctx.lineTo(1, 6)
    ctx.lineTo(4, 9.5)
    ctx.moveTo(8, 2.5)
    ctx.lineTo(11, 6)
    ctx.lineTo(8, 9.5)
    ctx.lineWidth = 1.3
    ctx.stroke()
  },
  lock: (ctx) => {
    ctx.beginPath()
    ctx.arc(6, 5.2, 1.9, Math.PI, 0)
    ctx.lineWidth = 1.2
    ctx.stroke()
    ctx.lineWidth = 1.2
    ctx.strokeRect(3.2, 5.2, 5.6, 5)
    dot(ctx, 6, 7.7, 0.8)
  },
}

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

function initGrid(canvas: HTMLCanvasElement) {
  const cols = Number(canvas.dataset.cols)
  const rows = Number(canvas.dataset.rows)
  const seed = Number(canvas.dataset.seed)
  const icons = (canvas.dataset.icons ?? '').split(',').filter(name => name in ICONS)
  const ctx = canvas.getContext('2d')
  if (!ctx || !cols || !rows)
    return

  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = cols * STEP * dpr
  canvas.height = rows * STEP * dpr

  // Density level: two noise layers moving in different directions, so
  // regions grow and decay (interference) instead of just sliding.
  function density(x: number, y: number, t: number): number {
    const n1 = noise(x * 0.13 + t * 0.45, y * 0.19 + t * 0.06, seed)
    const n2 = noise(x * 0.11 - t * 0.32, y * 0.16 - t * 0.1, seed + 7)
    return (n1 + n2) * 0.6
  }

  function draw(t: number) {
    if (!ctx)
      return
    const accent = getComputedStyle(canvas).color
    const muted = getComputedStyle(document.documentElement).getPropertyValue('--uno-colors-muted').trim() || '#8a8a8a'

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cols * STEP, rows * STEP)

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const d = density(x, y, t)
        const px = x * STEP
        const py = y * STEP
        const cx = px + SIZE / 2
        const cy = py + SIZE / 2

        if (d > 0.66) {
          // solid block
          ctx.fillStyle = accent
          ctx.fillRect(px, py, SIZE, SIZE)
        }
        else if (d > 0.6) {
          // shrinking square — the halftone size gradient
          const s = 7 + (d - 0.6) * 50
          ctx.fillStyle = accent
          ctx.fillRect(px + (SIZE - s) / 2, py + (SIZE - s) / 2, s, s)
        }
        else if (d > 0.54) {
          ctx.fillStyle = accent
          ctx.save()
          ctx.translate(cx, cy)
          ctx.rotate(Math.PI / 4)
          ctx.fillRect(-3.5, -3.5, 7, 7)
          ctx.restore()
        }
        else if (d > 0.48) {
          ctx.fillStyle = accent
          dot(ctx, cx, cy, 2.2)
        }
        else if (d > 0.3 && icons.length) {
          // themed icon bands: the 0.3–0.48 range is split evenly across
          // all of the grid's icons, so every icon type appears as its own
          // coherent wave that mutates with the field
          const band = Math.min(Math.floor(((0.48 - d) / 0.18) * icons.length), icons.length - 1)
          const icon = ICONS[icons[band]]
          ctx.save()
          ctx.translate(px, py)
          ctx.strokeStyle = muted
          ctx.fillStyle = muted
          icon(ctx)
          ctx.restore()
        }
        else if (d > 0.25) {
          // small cross
          ctx.strokeStyle = muted
          stroke(ctx, 1.1, () => line(ctx, [
            [cx - 2, cy - 2, cx + 2, cy + 2],
            [cx + 2, cy - 2, cx - 2, cy + 2],
          ]))
        }
        else if (d > 0.2) {
          // speck
          ctx.fillStyle = muted
          dot(ctx, cx, cy, 0.9)
        }
      }
    }
  }

  draw(0)
  if (reducedMotion)
    return

  let running = true
  let last = 0
  const start = performance.now()

  function frame(now: number) {
    if (running && now - last >= 1000 / FPS) {
      last = now
      draw((now - start) / 1000)
    }
    requestAnimationFrame(frame)
  }

  new IntersectionObserver((entries) => {
    for (const entry of entries) running = entry.isIntersecting
  }).observe(canvas)

  requestAnimationFrame(frame)
}

document.querySelectorAll<HTMLCanvasElement>('[data-pixel-grid]').forEach(initGrid)
