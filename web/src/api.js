/* Browser-to-loopback only. No cloud fallback, token persistence, or remote host support. */

export const DEFAULT_ENDPOINT = 'http://127.0.0.1:11435';
const LIMIT_BYTES = 128 * 1024;

export function normalizeEndpoint(value) {
  if (typeof value !== 'string') throw new Error('Enter a local endpoint URL.');
  let url;
  try { url = new URL(value.trim()); }
  catch { throw new Error('Use a complete local URL, for example http://127.0.0.1:11435.'); }
  const hosts = ['localhost', '127.0.0.1', '[::1]'];
  if (url.protocol !== 'http:' || !hosts.includes(url.hostname) ||
      (url.pathname !== '/' && url.pathname !== '') || url.search || url.hash ||
      url.username || url.password || !url.port || Number(url.port) < 1) {
    throw new Error('Only an explicit HTTP loopback endpoint with a port is allowed. No remote hosts, URL credentials, paths or query strings.');
  }
  return url.origin;
}

function getHeaders(token, json) {
  if (!token || typeof token !== 'string' || !token.trim()) {
    throw new Error('Paste the bearer token from your local PhiServe token file.');
  }
  return {
    Authorization: 'Bearer ' + token.trim(),
    ...(json ? { 'Content-Type': 'application/json' } : {}),
  };
}

async function checkResponse(response) {
  if (response.ok) return response;
  const hints = {
    401: 'Invalid or missing bearer token.',
    403: 'This page origin is not allowed by the PhiServe daemon.',
    404: 'Requested endpoint or model not found.',
    409: 'Another model is busy. Try again after it finishes.',
    422: 'PhiServe rejected an unsupported request.',
    429: 'Local request limit reached. Try again.',
    503: 'Model is starting, stopping or unavailable. Try again.',
  };
  throw new Error('PhiServe HTTP ' + response.status + '. ' + (hints[response.status] || 'Check local daemon logs.'));
}

export function friendlyConnectionError(error) {
  if (error?.name === 'AbortError') return 'Request cancelled.';
  if (error instanceof TypeError) {
    return 'Could not reach the local daemon. Check same-device access, exact --browser-origin, CORS, browser local-network permissions and token.';
  }
  return error instanceof Error ? error.message : 'Unexpected local connection error.';
}

export async function getJSON(endpoint, path, token, signal) {
  const base = normalizeEndpoint(endpoint);
  const response = await fetch(base + path, {
    method: 'GET', headers: getHeaders(token, false),
    redirect: 'error', cache: 'no-store', signal,
  });
  await checkResponse(response);
  return response.json();
}

/** Strict NDJSON processing. The final done:true frame is mandatory. */
export async function* parseNDJSON(chunks) {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let pending = '';
  let completed = false;
  const parse = (line) => {
    let frame;
    try { frame = JSON.parse(line); }
    catch { throw new Error('Invalid NDJSON from local PhiServe.'); }
    if (frame?.error) throw new Error('PhiServe stream: ' + String(frame.error));
    if (typeof frame?.done !== 'boolean') throw new Error('Missing NDJSON done marker.');
    if (frame.message?.content !== undefined && typeof frame.message.content !== 'string') {
      throw new Error('PhiServe returned non-text content.');
    }
    return { content: frame.message?.content || '', done: frame.done };
  };
  for await (const chunk of chunks) {
    pending += decoder.decode(chunk, { stream: true });
    if (pending.length > LIMIT_BYTES) throw new Error('Streaming frame exceeded the 128 KiB limit.');
    let split;
    while ((split = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, split).trim();
      pending = pending.slice(split + 1);
      if (!line) continue;
      const event = parse(line);
      if (event.content) yield event.content;
      if (event.done) {
        completed = true;
        break;
      }
    }
    if (completed) break;
  }
  if (!completed) {
    pending += decoder.decode();
    if (pending.trim()) {
      const event = parse(pending.trim());
      if (event.content) yield event.content;
      completed = event.done;
    }
  }
  if (!completed) throw new Error('PhiServe stream stopped without a completed response.');
}

export async function* streamChat({ endpoint, token, model, messages, signal }) {
  const base = normalizeEndpoint(endpoint);
  if (!model || !Array.isArray(messages) || !messages.length) {
    throw new Error('Select a registered model and provide a message.');
  }
  const response = await fetch(base + '/api/chat', {
    method: 'POST', headers: getHeaders(token, true),
    redirect: 'error', cache: 'no-store', signal,
    body: JSON.stringify({
      model, stream: true,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      options: { num_predict: 256, temperature: 0.7 },
    }),
  });
  await checkResponse(response);
  if (!response.body) throw new Error('This browser does not support streamed responses.');
  const reader = response.body.getReader();
  const stream = {
    async *[Symbol.asyncIterator]() {
      try {
        while (true) {
          const next = await reader.read();
          if (next.done) return;
          yield next.value;
        }
      } finally {
        reader.cancel().catch(() => {});
      }
    },
  };
  try {
    yield* parseNDJSON(stream);
  } finally {
    reader.cancel().catch(() => {});
  }
}
