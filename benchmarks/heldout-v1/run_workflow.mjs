#!/usr/bin/env node
// Run a workflow script (workflow-pass1.js / workflow-pass2.js) outside Claude Code. Node 18+, no dependencies.
//   node run_workflow.mjs <script.js> <families.json> --out <result.json> [--concurrency N] [--timeout-min M] [--model-override MODEL] [--dry-run]
// The script body runs as an async function with globals agent, parallel, pipeline, log, phase and args (the parsed families file).
// agent() spawns `claude -p` (Claude Code CLI, logged in) with the prompt on stdin and returns the structured output.
// At most --concurrency agent processes run at once, however deeply the script nests parallel().
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const USAGE = 'usage: node run_workflow.mjs <script.js> <families.json> --out <result.json> [--concurrency N] [--timeout-min M] [--model-override MODEL] [--dry-run]'
const positional = []
const opt = { concurrency: 4, out: null, modelOverride: null, dryRun: false, timeoutMin: 20 }
const argv = process.argv.slice(2)
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]
  if (a === '--out') opt.out = argv[++i]
  else if (a === '--concurrency') opt.concurrency = Number(argv[++i])
  else if (a === '--model-override') opt.modelOverride = argv[++i]
  else if (a === '--timeout-min') opt.timeoutMin = Number(argv[++i])
  else if (a === '--dry-run') opt.dryRun = true
  else if (a.startsWith('--')) { console.error(`unknown option ${a}\n${USAGE}`); process.exit(2) }
  else positional.push(a)
}
if (positional.length !== 2 || !opt.out || !Number.isInteger(opt.concurrency) || opt.concurrency < 1 || !(opt.timeoutMin > 0)) { console.error(USAGE); process.exit(2) }
const [scriptPath, familiesPath] = positional

const stats = { agents: 0, failed: 0, retried: 0, cost: 0 }
const err = (...m) => console.error(...m)
const log = (...m) => err('[log]', ...m.map(x => (typeof x === 'string' ? x : JSON.stringify(x))))
const phase = (name) => err(`[phase] ${name}`)

// Counting semaphore over agent processes.
let free = opt.concurrency
const waiting = []
const acquire = () => (free > 0 ? (free--, Promise.resolve()) : new Promise(r => waiting.push(r)))
const release = () => { const next = waiting.shift(); if (next) next(); else free++ }

const children = new Set()
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { for (const c of children) c.kill('SIGKILL'); process.exit(130) })
const scratchCwd = mkdtempSync(join(tmpdir(), 'wf-agent-'))
process.on('exit', () => { try { rmSync(scratchCwd, { recursive: true, force: true }) } catch {} })

function runClaude(prompt, o) {
  const model = opt.modelOverride || o.model
  const args = ['-p']
  if (model) args.push('--model', model)
  if (o.effort && !opt.modelOverride) args.push('--effort', o.effort)
  args.push('--output-format', 'json', '--tools', '', '--no-session-persistence')
  if (o.schema) args.push('--json-schema', JSON.stringify(o.schema))
  return new Promise((resolve) => {
    const child = spawn('claude', args, { cwd: scratchCwd, stdio: ['pipe', 'pipe', 'pipe'] })
    children.add(child)
    // A hung process counts as a failed attempt, so one stuck call cannot stall the run.
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL') }, opt.timeoutMin * 60000)
    let out = '', errText = ''
    child.stdout.on('data', d => { out += d })
    child.stderr.on('data', d => { errText += d })
    child.on('error', e => { clearTimeout(timer); children.delete(child); resolve({ ok: false, why: `spawn: ${e.message}` }) })
    child.on('close', (code) => {
      clearTimeout(timer); children.delete(child)
      if (timedOut) return resolve({ ok: false, why: `timed out after ${opt.timeoutMin} min` })
      if (code !== 0) return resolve({ ok: false, why: `exit ${code}: ${(errText || out).slice(0, 200).replace(/\s+/g, ' ')}` })
      let j
      try { j = JSON.parse(out) } catch { return resolve({ ok: false, why: 'stdout is not JSON' }) }
      if (typeof j.total_cost_usd === 'number') stats.cost += j.total_cost_usd
      if (j.is_error) return resolve({ ok: false, why: `is_error: ${String(j.result || j.subtype).slice(0, 200)}` })
      if (o.schema && (j.structured_output === undefined || j.structured_output === null)) return resolve({ ok: false, why: 'no structured_output' })
      resolve({ ok: true, value: o.schema ? j.structured_output : j.result })
    })
    child.stdin.on('error', () => {})
    child.stdin.end(prompt)
  })
}

async function liveAgent(prompt, o = {}) {
  await acquire()
  try {
    stats.agents++
    const label = o.label || '(unlabelled)'
    let r = await runClaude(prompt, o)
    if (!r.ok) { stats.retried++; err(`[agent] ${label}: ${r.why}; retrying`); r = await runClaude(prompt, o) }
    if (!r.ok) { stats.failed++; err(`[agent] ${label}: failed (${r.why})`); return null }
    return r.value
  } finally { release() }
}

// ---- offline stub (--dry-run): schema-shaped answers that make the gates accept, strings vary per call ----
let stubN = 0
const stubAnswer = new Map()
const ANSWER_KIND = (p) => (/single letter of the correct option/.test(p) ? 'letter' : /a single number with no units/.test(p) ? 'number' : 'latex')
const indexed = (p) => [...p.matchAll(/^\[(\d+)\]/gm)].map(m => Number(m[1]))
function stubWriter(prompt) {
  const kind = ANSWER_KIND(prompt)
  const items = Array.from({ length: 10 }, () => {
    const n = ++stubN
    const answer = kind === 'letter' ? 'ABCD'[n % 4] : String(n)
    stubAnswer.set(n, answer)
    const head = `Dry run item ${n} zq${n}x with its own made-up setting number ${n * 7}.`
    const body = kind === 'letter' ? `${head}\nA. option one ${n}\nB. option two ${n}\nC. option three ${n}\nD. option four ${n}` : `${head} What is the value?`
    return { body, answer, solution: `stub solution ${n}`, program: kind === 'number' ? `def solve():\n    return ${n}` : '', topic: 'stub topic' }
  })
  return { items }
}
const markerOf = (text) => { const m = /Dry run item (\d+)/.exec(text); return m ? Number(m[1]) : null }
const chunks = (p) => p.split(/^\[(\d+)\]\n/m).slice(1).reduce((acc, x, i, a) => (i % 2 === 0 ? acc.concat([[Number(x), a[i + 1]]]) : acc), [])
function stubSolver(prompt) {
  return { answers: chunks(prompt).map(([index, text]) => ({ index, answer: stubAnswer.get(markerOf(text)) ?? '0', confident: true, ambiguous: false, issue: '' })) }
}
function stubCritic(prompt) {
  return { verdicts: chunks(prompt).map(([index, text]) => { const bad = markerOf(text) % 7 === 0; return { index, keep: !bad, resembles_known: false, reason: bad ? 'stub rejection' : 'stub keep' } }) }
}
function stubEquiv(prompt) {
  return { verdicts: [...prompt.matchAll(/^\[(\d+)\] item: [\s\S]*?\nwriter: (.*)$/gm)].map(m => ({ index: Number(m[1]), equivalent: true, canonical: m[2] })) }
}
function stubFromSchema(s) {
  if (!s) return 'stub'
  if (s.type === 'object') return Object.fromEntries(Object.entries(s.properties || {}).map(([k, v]) => [k, /^(confident|keep|equivalent)$/.test(k) ? true : v.type === 'boolean' ? false : stubFromSchema(v)]))
  if (s.type === 'array') return [stubFromSchema(s.items)]
  if (s.type === 'boolean') return false
  if (s.type === 'integer' || s.type === 'number') return 0
  return `stub ${++stubN}`
}
async function stubAgent(prompt, o = {}) {
  stats.agents++
  if (prompt.startsWith('You are writing ORIGINAL evaluation items')) return stubWriter(prompt)
  if (prompt.startsWith('Solve each item below independently.')) return stubSolver(prompt)
  if (prompt.startsWith('You are the quality and memorization critic')) return stubCritic(prompt)
  if (prompt.startsWith('For each index decide whether the three answers')) return stubEquiv(prompt)
  return stubFromSchema(o.schema)
}

// ---- workflow globals ----
const agent = opt.dryRun ? stubAgent : liveAgent
const run1 = async (thunk, i) => { try { return await (typeof thunk === 'function' ? thunk(i) : thunk) } catch (e) { err(`[parallel] task ${i} threw: ${e && e.message}`); return null } }
const parallel = (thunks) => Promise.all(thunks.map(run1))
// pipeline(items, ...stages): each item flows through the stages in order (stage(prev, item, index)); items run concurrently.
// A stage that returns null/undefined or throws ends that item with null.
const pipeline = (items, ...stages) => Promise.all(items.map(async (item, i) => {
  let prev = item
  for (const stage of stages) {
    try { prev = await stage(prev, item, i) } catch (e) { err(`[pipeline] item ${i} threw: ${e && e.message}`); return null }
    if (prev === null || prev === undefined) return null
  }
  return prev
}))

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const source = readFileSync(scriptPath, 'utf8').replace(/^export const meta =/m, 'const meta =')
const args = JSON.parse(readFileSync(familiesPath, 'utf8'))
const body = new AsyncFunction('agent', 'parallel', 'pipeline', 'log', 'phase', 'args', source)

const t0 = Date.now()
let result
try { result = await body(agent, parallel, pipeline, log, phase, args) } catch (e) { err(`workflow threw: ${e && e.stack}`); process.exit(1) }
writeFileSync(opt.out, JSON.stringify(result, null, 1), 'utf8')
err(`done in ${Math.round((Date.now() - t0) / 1000)}s${opt.dryRun ? ' (dry run)' : ''}: agents run ${stats.agents}, retried ${stats.retried}, failed ${stats.failed}, total_cost_usd ${stats.cost.toFixed(4)}; result -> ${opt.out}`)
process.exit(0)
