<script>
  // The zignanogpt web console: Chat talks to the loaded model (replies
  // streamed), Training tails the runs' metrics.jsonl.

  import Chip from '@wui/zigwui/components/Chip.svelte';
  import { onMount } from 'svelte';
  import ChatPane from './ChatPane.svelte';
  import TrainingPane from './TrainingPane.svelte';
  import { info as readInfo } from './api.js';

  const tabs = [
    { key: 'chat', label: 'Chat' },
    { key: 'train', label: 'Training' },
  ];

  let info = $state({ model: null, version: '' });
  let tab = $state('chat');

  onMount(async () => {
    try {
      info = await readInfo();
    } catch {
      // The header then shows no version; the panes report their own failures.
    }
  });
</script>

<div class="flex h-full flex-col bg-deep-black text-ink">
  <header class="glass-panel flex flex-wrap items-center gap-3 border-x-0 border-t-0 px-4 py-2.5">
    <h1 class="font-header text-base font-semibold tracking-tight">zignanogpt</h1>
    <nav class="flex gap-1">
      {#each tabs as entry (entry.key)}
        <button
          class="rounded px-3 py-1 text-sm"
          class:bg-deep-raised={tab === entry.key}
          class:text-ink={tab === entry.key}
          class:text-ink-muted={tab !== entry.key}
          onclick={() => (tab = entry.key)}
          data-testid="tab.{entry.key}">{entry.label}</button
        >
      {/each}
    </nav>
    <span class="ml-auto">
      <Chip tone={info.model ? 'accent' : 'neutral'}>{info.model ?? 'no model yet'}{info.version ? ` · v${info.version}` : ''}</Chip>
    </span>
  </header>

  <main class="mx-auto min-h-0 w-full max-w-5xl flex-1">
    <!-- Hidden rather than unmounted, so the conversation keeps its state. -->
    <div class="h-full" class:hidden={tab !== 'chat'}>
      <ChatPane hasModel={info.model !== null} />
    </div>
    <div class="h-full overflow-auto" class:hidden={tab !== 'train'}>
      <TrainingPane active={tab === 'train'} />
    </div>
  </main>
</div>
