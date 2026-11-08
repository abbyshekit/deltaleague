export const meta = {
  name: 'heldout-set-v1',
  description: 'Generate and verify a held-out test set in the Delta League validation format: original items per family, two independent solvers, an adversarial critic, dedup, loop until the target count per family',
  phases: [
    { title: 'Write', detail: 'Opus writers, 10 original items per batch', model: 'opus' },
    { title: 'Solve', detail: 'two independent Sonnet solvers per batch', model: 'sonnet' },
    { title: 'Judge', detail: 'Opus memorization/quality critic; Sonnet equivalence judge when answers differ' },
  ],
}

const SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
const ITEMS = { type: 'object', properties: { items: { type: 'array', items: { type: 'object', properties: {
  body: { type: 'string' }, answer: { type: 'string' }, solution: { type: 'string' }, program: { type: 'string' }, topic: { type: 'string' } },
  required: ['body', 'answer', 'solution', 'topic'] } } }, required: ['items'] }
const SOLVE = { type: 'object', properties: { answers: { type: 'array', items: { type: 'object', properties: {
  index: { type: 'integer' }, answer: { type: 'string' }, confident: { type: 'boolean' }, ambiguous: { type: 'boolean' }, issue: { type: 'string' } },
  required: ['index', 'answer', 'confident', 'ambiguous'] } } }, required: ['answers'] }
const CRITIC = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  index: { type: 'integer' }, keep: { type: 'boolean' }, resembles_known: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['index', 'keep', 'resembles_known', 'reason'] } } }, required: ['verdicts'] }
const EQUIV = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  index: { type: 'integer' }, equivalent: { type: 'boolean' }, canonical: { type: 'string' } }, required: ['index', 'equivalent', 'canonical'] } } }, required: ['verdicts'] }

const norm = (s) => String(s ?? '').replace(/\\boxed\{([^}]*)\}/, '$1').replace(/\$/g, '').replace(/\s+/g, '').replace(/\.$/, '').replace(/^\\text\{(.*)\}$/, '$1').toLowerCase()
const numEq = (a, b) => { const x = parseFloat(a.replace(/,/g, '')), y = parseFloat(b.replace(/,/g, '')); return Number.isFinite(x) && Number.isFinite(y) && Math.abs(x - y) < 1e-9 }
const same = (fam, a, b) => norm(a) === norm(b) || (fam.numeric && numEq(norm(a), norm(b)))
const flags = (r) => (r ? { confident: r.confident ?? null, ambiguous: r.ambiguous ?? null, issue: r.issue || '' } : null)
const key = (body) => body.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().slice(0, 90)

function writerPrompt(fam, batch, topics) {
  const ex = fam.examples.map((e, i) => `Example ${i + 1} (public item, for style and difficulty only):\n${e.body}\nAnswer: ${e.answer}`).join('\n\n')
  return `You are writing ORIGINAL evaluation items for a held-out test set. Family: ${fam.name}. ${fam.spec}

Write exactly 10 new items. Each must be entirely original: a new scenario, new numbers, new names; never a known benchmark item, textbook classic, famous puzzle, or paraphrase of one. Do not reuse the examples' topics or structure. Assigned topics for this batch, one or two items each: ${topics.join('; ')}. Match the examples' difficulty and register. Batch ${batch} of this family: vary sentence structure and setting from anything you would write by default; prefer concrete, specific scenarios (a named place, object or process) over the generic phrasing a textbook would use, and keep every number and name new.

${fam.answerRules}

Return "body" WITHOUT any closing instruction line (the harness appends the standard instruction). "answer" is the final answer only, in the canonical form above. "solution" is a terse worked solution. ${fam.programRule} "topic" is the assigned topic.

${ex}`
}

function solverPrompt(fam, items, adversarial) {
  const list = items.map((it, i) => `[${i}]\n${it.body}`).join('\n\n')
  return `Solve each item below independently. ${fam.answerRules}
${adversarial ? 'You are the adversarial solver: for each item also decide whether it is ambiguous, underspecified, has more than one defensible answer, has no correct option, or contains an error; set "ambiguous": true and explain in "issue" when so. Be strict: a competition grader must be able to mark it with no judgement calls. Set "confident": true only when you have checked your answer; otherwise false with the doubt in "issue".' : 'Set "confident": true only when you have checked the answer; otherwise false with the doubt in "issue". Set "ambiguous": false unless the item cannot be answered as written.'}
Return one entry per index with the final answer in canonical form (no \\boxed, no units unless the item asks for them).

${list}`
}

function criticPrompt(fam, items) {
  const list = items.map((it, i) => `[${i}]\n${it.body}\nProposed answer: ${it.answer}`).join('\n\n')
  return `You are the quality and memorization critic for a held-out test set (family: ${fam.name}). For each item decide keep=true only if ALL hold: (1) it is not a near-duplicate of a specific item you can recall from ARC, GSM8K, MATH, MMLU, GPQA, AQuA, SVAMP, ASDiv, MathQA, AIME/AMC or a well-known textbook, meaning the same facts, numbers or objects arranged so that a model could answer by recalling that item rather than reasoning; a shared TEMPLATE or topic (seasons, frost wedging, ratio changes, average-after-n-tests, price up then down) is NOT a resemblance, because the public benchmarks are built from the same templates; set resembles_known=true only when you can state the specific recalled item in one sentence; (2) exactly one correct answer, which is the proposed one; (3) ${fam.criticRule}; (4) it is not trivially guessable from surface cues (for multiple choice: the correct option is not the only long, hedged, or detailed one); (5) the English is clean and the numbers are consistent. Give a one-sentence reason either way.

${list}`
}

function equivPrompt(fam, triples) {
  const list = triples.map(t => `[${t.index}] item: ${t.body.slice(0, 300)}\nwriter: ${t.writer}\nsolver A: ${t.a}\nsolver B: ${t.b}`).join('\n\n')
  return `For each index decide whether the three answers are mathematically the same value or expression written differently (equivalent=true) or genuinely different (false). If equivalent, give the single canonical form a strict grader using math-verify would accept, in the family's convention: ${fam.answerRules}

${list}`
}

async function runBatch(fam, batch, topics) {
  const written = await agent(writerPrompt(fam, batch, topics), { label: `write:${fam.name}:${batch}`, phase: 'Write', schema: ITEMS, model: 'opus', effort: 'high' })
  const items = (written?.items || []).filter(it => it.body && it.answer).slice(0, 10).map(it => ({ ...it, body: it.body.trim() }))
  if (!items.length) return { items: [], accepted: [], rejected: [] }
  const [a, b, c] = await parallel([
    () => agent(solverPrompt(fam, items, false), { label: `solveA:${fam.name}:${batch}`, phase: 'Solve', schema: SOLVE, model: 'sonnet' }),
    () => agent(solverPrompt(fam, items, true), { label: `solveB:${fam.name}:${batch}`, phase: 'Solve', schema: SOLVE, model: 'sonnet' }),
    () => agent(criticPrompt(fam, items), { label: `critic:${fam.name}:${batch}`, phase: 'Judge', schema: CRITIC, model: 'opus' }),
  ])
  const byIdx = (r, k) => Object.fromEntries(((r && r[k]) || []).map(x => [x.index, x]))
  const A = byIdx(a, 'answers'), B = byIdx(b, 'answers'), C = byIdx(c, 'verdicts')
  const accepted = [], rejected = [], differ = []
  items.forEach((it, i) => {
    const ra = A[i], rb = B[i], rc = C[i]
    const reasons = []
    if (!ra || !rb || !rc) reasons.push('missing verdict')
    for (const [who, r] of [['A', ra], ['B', rb]]) {
      if (!r) continue
      if (r.ambiguous === true) reasons.push(`solver ${who} ambiguous: ` + (r.issue || 'ambiguous'))
      if (r.confident !== true) reasons.push(`solver ${who} not confident: ` + (r.issue || ''))
    }
    const solver_flags = { a: flags(ra), b: flags(rb) }
    if (rc && !rc.keep) reasons.push('critic: ' + rc.reason)
    if (rc && rc.resembles_known) reasons.push('resembles known: ' + rc.reason)
    const agreeA = ra && same(fam, it.answer, ra.answer), agreeB = rb && same(fam, it.answer, rb.answer)
    if (reasons.length) rejected.push({ item: it, reasons, a: ra?.answer, b: rb?.answer, solver_flags })
    else if (agreeA && agreeB) accepted.push({ ...it, a: ra.answer, b: rb.answer, solver_flags })
    else if (fam.numeric || fam.letter) rejected.push({ item: it, reasons: [`answers disagree: writer ${it.answer}, A ${ra.answer}, B ${rb.answer}`], a: ra.answer, b: rb.answer, solver_flags })
    else differ.push({ index: i, body: it.body, writer: it.answer, a: ra.answer, b: rb.answer, it, solver_flags })
  })
  if (differ.length) {
    const e = await agent(equivPrompt(fam, differ), { label: `equiv:${fam.name}:${batch}`, phase: 'Judge', schema: EQUIV, model: 'sonnet' })
    const E = byIdx(e, 'verdicts')
    for (const d of differ) {
      const v = E[d.index]
      if (v && v.equivalent) accepted.push({ ...d.it, answer: v.canonical || d.it.answer, a: d.a, b: d.b, solver_flags: d.solver_flags, equivalence_judged: true })
      else rejected.push({ item: d.it, reasons: [`answers differ and not judged equivalent: writer ${d.writer}, A ${d.a}, B ${d.b}`], a: d.a, b: d.b, solver_flags: d.solver_flags })
    }
  }
  return { items, accepted, rejected }
}

async function runFamily(fam) {
  const seen = new Set(), accepted = [], rejected = []
  let round = 0, dry = 0, batches = 0, writtenTotal = 0
  while (accepted.length < fam.target && dry < 2 && round < 8) {
    round++
    const n = Math.min(fam.batchesPerRound, Math.ceil((fam.target - accepted.length) / 6))
    const results = (await parallel(Array.from({ length: n }, (_, j) => () => runBatch(fam, batches + j, fam.topics.slice(((batches + j) * 3) % fam.topics.length, ((batches + j) * 3) % fam.topics.length + 3))))).filter(Boolean)
    batches += n
    let fresh = 0
    for (const r of results) {
      writtenTotal += r.items.length
      rejected.push(...r.rejected)
      for (const it of r.accepted) {
        const k = key(it.body)
        if (seen.has(k)) { rejected.push({ item: it, reasons: ['duplicate of an accepted item'] }); continue }
        seen.add(k); accepted.push(it); fresh++
      }
    }
    dry = fresh < 2 ? dry + 1 : 0
    log(`${fam.name}: round ${round}, ${n} batches, +${fresh} accepted (${accepted.length}/${fam.target}); written ${writtenTotal}, rejected ${rejected.length}`)
  }
  if (accepted.length < fam.target) log(`${fam.name}: stopped at ${accepted.length}/${fam.target} after ${round} rounds (dry=${dry})`)
  return { name: fam.name, scorer: fam.scorer, accepted: accepted.slice(0, fam.target).map((it, i) => ({ id: `${fam.name}-${i}`, prompt: it.body + SUFFIX, answer: it.answer, scorer: fam.scorer, samples: 1, private: { solution: it.solution, program: it.program || '', topic: it.topic, solver_a: it.a, solver_b: it.b, solver_flags: it.solver_flags, equivalence_judged: !!it.equivalence_judged } })),
    overflow: accepted.slice(fam.target).length, rejected: rejected.map(r => ({ body: r.item.body, answer: r.item.answer, reasons: r.reasons, a: r.a, b: r.b, solver_flags: r.solver_flags || r.item.solver_flags })), rounds: round, batches, written: writtenTotal }
}

const results = (await parallel(args.families.map(fam => () => runFamily(fam)))).filter(Boolean)
const summary = Object.fromEntries(results.map(r => [r.name, { accepted: r.accepted.length, rejected: r.rejected.length, written: r.written, rounds: r.rounds, batches: r.batches }]))
log(`done: ${JSON.stringify(summary)}`)
return { summary, families: results }