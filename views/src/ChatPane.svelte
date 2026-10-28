<script>
  // A conversation with the loaded model. The server keeps no state: every
  // request carries the whole conversation, ending with the user's turn, and
  // the reply streams back token by token. A failed turn is dropped from the
  // conversation and noted in the transcript.

  import Conversation from '@wui/zigwui/components/Conversation.svelte';
  import { Button } from 'flowbite-svelte';
  import { chat } from './api.js';

  /** @type {{ hasModel: boolean }} */
  let { hasModel } = $props();

  // What the model sees, and what the page shows (failures included).
  let messages = [];
  let updates = $state([]);
  let draft = $state('');
  let busy = $state(false);
  let temperature = $state(0.6);
  let topK = $state(50);
  let maxTokens = $state(512);
  let input = $state(null);

  const say = (sessionUpdate, text) => ({ sessionUpdate, content: { type: 'text', text } });

  /** Sends the draft and streams the reply. */
  async function send(event) {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    busy = true;
    draft = '';
    messages.push({ role: 'user', content: text });
    updates.push(say('user_message_chunk', text));
    let reply = '';
    try {
      await chat({ messages, temperature: Number(temperature) || 0, top_k: Number(topK) || 0, max_tokens: Number(maxTokens) || 512 }, (token) => {
        reply += token;
        updates.push(say('agent_message_chunk', token));
      });
      messages.push({ role: 'assistant', content: reply });
    } catch (failure) {
      messages.pop();
      updates.push({ note: `Error: ${failure.message}` });
    } finally {
      busy = false;
      input?.focus();
    }
  }

  /** Starts over. */
  function clear() {
    if (busy) return;
    messages = [];
    updates = [];
  }

  /** Enter sends, Shift+Enter starts a new line. */
  function keydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) send(event);
  }

  const field = 'w-20 rounded border border-edge bg-deep-slate px-2 py-1 font-mono text-ink';
</script>

<section class="flex h-full flex-col gap-3 p-4">
  <div class="flex flex-wrap items-center gap-4 text-xs text-ink-muted">
    <label class="flex items-center gap-2">temperature <input class={field} type="number" step="0.1" min="0" max="2" bind:value={temperature} data-testid="chat.temperature" /></label>
    <label class="flex items-center gap-2">top-k <input class={field} type="number" min="0" bind:value={topK} data-testid="chat.topk" /></label>
    <label class="flex items-center gap-2">max tokens <input class={field} type="number" min="1" bind:value={maxTokens} data-testid="chat.max" /></label>
    <Button size="xs" color="alternative" disabled={busy} onclick={clear} data-testid="chat.clear">New chat</Button>
  </div>

  {#if updates.length === 0}
    <p class="flex-1 pt-10 text-center text-sm text-ink-muted">
      {hasModel ? 'Say something to the model.' : 'No model is loaded: train or import one, then restart the console.'}
    </p>
  {:else}
    <Conversation {updates} agent="zignanogpt" working={busy} />
  {/if}

  <form class="flex items-end gap-2 border-t border-edge pt-3" onsubmit={send}>
    <textarea
      bind:this={input}
      bind:value={draft}
      onkeydown={keydown}
      rows="2"
      class="max-h-48 min-h-11 flex-1 resize-none rounded border border-edge bg-deep-slate px-3 py-2 text-sm text-ink"
      placeholder="Message (Enter to send, Shift+Enter for a new line)"
      data-testid="chat.input"
    ></textarea>
    <Button type="submit" disabled={busy || !draft.trim()} data-testid="chat.send">Send</Button>
  </form>
</section>
