// The console's own routes (web/handler.zig). Failures carry the server's
// `{"error": …}` sentence, through zigwui's `request`.

import { request } from '@wui/zigwui/api';

/** The loaded model's label (null without one) and the version. */
export const info = () => request('/api/info');

/** Every training run with a `metrics.jsonl`, base first. */
export const runs = () => request('/api/runs');

/** The complete lines of a run's metrics after `offset`, and the next offset. */
export const metrics = (run, offset) => request(`/api/metrics?run=${encodeURIComponent(run)}&offset=${offset}`);

/**
 * Streams an assistant reply to `messages` (the whole conversation, ending
 * with the user's turn), calling `onToken` with each piece of text.
 *
 * @param {{ messages: Array<{ role: string, content: string }>, temperature: number, top_k: number, max_tokens: number }} body
 * @param {(token: string) => void} onToken
 * @returns {Promise<void>} once the reply ends; rejects with the server's error
 */
export async function chat(body, onToken) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (response.headers.get('content-type') !== 'text/event-stream') {
    const failure = await response.json().catch(() => ({}));
    throw new Error(failure.error ?? `${response.status} ${response.statusText}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    pending += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = pending.indexOf('\n\n')) >= 0) {
      const line = pending.slice(0, cut);
      pending = pending.slice(cut + 2);
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));
      if (event.token !== undefined) onToken(event.token);
      if (event.error) throw new Error(event.error);
    }
  }
}
