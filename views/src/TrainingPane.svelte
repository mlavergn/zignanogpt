<script>
  // The training dashboard: pick a run, and its metrics.jsonl is tailed every
  // two seconds while this pane is showing. Shows the latest of each metric,
  // a chart per group of metrics present, and the latest step's samples.

  import Card from '@wui/zigwui/components/Card.svelte';
  import EmptyState from '@wui/zigwui/components/EmptyState.svelte';
  import { onMount } from 'svelte';
  import LineChart from './LineChart.svelte';
  import { metrics, runs as listRuns } from './api.js';

  /** @type {{ active: boolean }} */
  let { active } = $props();

  // Series colors: the accent first, then hues that read on the dark ground.
  const colors = ['rgb(255, 112, 52)', '#6c9cf0', '#5cc28a', '#b58be0'];

  const stats = [
    ['loss', 'loss', 4],
    ['val bpb', 'val_bpb', 4],
    ['ChatCORE', 'chatcore', 4],
    ['reward', 'reward', 3],
    ['pass@1', 'pass_at_1', 3],
    ['tok/s', 'tok_per_sec', 0],
    ['TFLOP/s', 'tflops', 3],
  ];

  const charts = [
    ['Training loss', [['loss', 'loss', 0]]],
    ['Validation', [['val_bpb', 'val bpb', 1], ['chatcore', 'ChatCORE', 2], ['pass_at_1', 'pass@1', 2]]],
    ['Reward', [['reward', 'reward', 3]]],
    ['Learning-rate multiplier', [['lrm', 'lrm', 2]]],
    ['Throughput (tok/s)', [['tok_per_sec', 'tok/s', 3]]],
  ];

  let available = $state(null);
  let run = $state('');
  let rows = $state.raw([]);
  let updated = $state('');
  let failure = $state('');
  let offset = 0;

  const series = (key) => rows.filter((r) => typeof r[key] === 'number').map((r) => [r.step, r[key]]);
  const last = (key) => {
    const points = series(key);
    return points.length ? points[points.length - 1][1] : null;
  };
  const format = (value, digits) => (value === null ? null : Math.abs(value) >= 1000 ? Math.round(value).toLocaleString() : value.toFixed(digits));

  const step = $derived.by(() => {
    const steps = rows.filter((r) => typeof r.loss === 'number' || typeof r.reward === 'number');
    return steps.length ? steps[steps.length - 1].step : null;
  });
  const tiles = $derived([['step', step === null ? null : String(step)], ...stats.map(([name, key, digits]) => [name, format(last(key), digits)]).filter(([, value]) => value !== null)]);
  const drawn = $derived(
    charts
      .map(([title, keys]) => ({ title, lines: keys.map(([key, label, color]) => ({ label, color: colors[color], points: series(key) })).filter((line) => line.points.length) }))
      .filter((chart) => chart.lines.length),
  );
  const samples = $derived.by(() => {
    const all = rows.filter((r) => typeof r.sample === 'string');
    if (!all.length) return null;
    const at = all[all.length - 1].step;
    return { step: at, texts: all.filter((s) => s.step === at).map((s) => s.sample) };
  });

  onMount(() => {
    const timer = setInterval(() => {
      if (active && !document.hidden) poll();
    }, 2000);
    return () => clearInterval(timer);
  });

  // Showing the pane refreshes the run list.
  $effect(() => {
    if (active) loadRuns();
  });

  /** Lists the runs, keeping the chosen one while it still exists. */
  async function loadRuns() {
    try {
      available = await listRuns();
      failure = '';
    } catch (error) {
      failure = error.message;
      return;
    }
    if (!available.length) return;
    const next = available.some((r) => r.name === run) ? run : available[0].name;
    if (next !== run) choose(next);
  }

  /** Starts tailing a run from its beginning. */
  function choose(name) {
    run = name;
    offset = 0;
    rows = [];
    poll();
  }

  /** Reads the chosen run's new lines. */
  async function poll() {
    if (!run) return;
    const asked = run;
    let tail;
    try {
      tail = await metrics(asked, offset);
    } catch (error) {
      failure = error.message;
      return;
    }
    if (asked !== run) return;
    failure = '';
    // A file that shrank is a new run, read again from its start.
    const base = tail.offset < offset ? [] : rows;
    offset = tail.offset;
    const added = [];
    for (const line of tail.text.split('\n')) {
      if (!line.trim()) continue;
      try {
        added.push(JSON.parse(line));
      } catch {
        // A malformed line is skipped, as the CLI's tail does.
      }
    }
    if (added.length || base !== rows) rows = [...base, ...added];
    updated = new Date().toLocaleTimeString();
  }
</script>

<section class="flex flex-col gap-4 p-4">
  <div class="flex flex-wrap items-center gap-3 text-sm text-ink-muted">
    <label class="flex items-center gap-2"
      >run
      <select
        class="rounded border border-edge bg-deep-slate py-1 pr-8 pl-2 font-mono text-ink"
        value={run}
        onchange={(event) => choose(event.currentTarget.value)}
        data-testid="train.run"
      >
        {#each available ?? [] as entry (entry.name)}<option value={entry.name}>{entry.name}</option>{/each}
      </select>
    </label>
    {#if updated}<span class="text-xs text-ink-faint">updated {updated}</span>{/if}
    {#if failure}<span class="text-xs text-red-400">{failure}</span>{/if}
  </div>

  {#if available && !available.length}
    <EmptyState title="No training runs yet" detail="zignanogpt train, sft or rl write metrics.jsonl." />
  {:else if run}
    <div class="grid grid-cols-[repeat(auto-fit,minmax(130px,1fr))] gap-3">
      {#each tiles as [name, value] (name)}
        <div class="rounded-lg border border-edge bg-deep-slate px-3 py-2">
          <div class="text-xs text-ink-muted">{name}</div>
          <div class="font-mono text-xl tabular-nums" class:text-ink-faint={value === null}>{value ?? '—'}</div>
        </div>
      {/each}
    </div>
    <div class="grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-3">
      {#each drawn as chart (chart.title)}
        <Card title={chart.title}><LineChart lines={chart.lines} /></Card>
      {/each}
    </div>
    {#if samples}
      <Card title="Samples at step {samples.step}">
        {#each samples.texts as text, index (index)}
          <p class="border-t border-edge py-1.5 text-sm whitespace-pre-wrap first:border-t-0">{text}</p>
        {/each}
      </Card>
    {/if}
  {/if}
</section>
