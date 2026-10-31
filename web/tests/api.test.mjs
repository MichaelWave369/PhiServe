import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_ENDPOINT, friendlyConnectionError, getJSON, normalizeEndpoint, parseNDJSON, streamChat } from '../src/api.js';

const enc = new TextEncoder();
const TOKEN = 'local-token-example-not-an-actual-secret';
const entry = (content, done) => JSON.stringify({ message: { content }, done }) + '\n';

async function read(iterable) {
  let value = '';
  for await (const part of iterable) value += part;
  return value;
}

test('default endpoint is loopback and requires explicit nondefault port', () => {
  assert.equal(DEFAULT_ENDPOINT, 'http://127.0.0.1:11435');
  assert.equal(normalizeEndpoint('http://127.0.0.1:11435/'), DEFAULT_ENDPOINT);
  assert.equal(normalizeEndpoint('http://localhost:11435/'), 'http://localhost:11435');
  assert.equal(normalizeEndpoint('http://[::1]:11435'), 'http://[::1]:11435');
});

test('remote, mixed-content, URL-credential and path tricks rejected', () => {
  for (const bad of [
    'https://localhost:11435', 'http://192.168.1.25:11435',
    'https://example.com', 'http://example.com:11435',
    'http://127.0.0.1.evil.com:11435', 'http://[::2]:11435',
    'http://user:pass@localhost:11435', 'http://127.0.0.1:11435/api/chat',
    'http://127.0.0.1:11435/?token=secrets',
    'http://127.0.0.1:11435/#frag',
    'http://127.0.0.1', 'http://127.0.0.1:0',
    'file:///etc/passwd', 'not a URL',
  ]) {
    assert.throws(() => normalizeEndpoint(bad), { name: 'Error' }, bad);
  }
});

test('parser handles NDJSON split across chunks and unicode', async () => {
  const data = entry('Hello ', false) + entry('世界', false) + entry('', true);
  const bytes = enc.encode(data);
  async function* chunks() {
    yield bytes.slice(0, 21);
    yield bytes.slice(21, 40);
    yield bytes.slice(40, 57);
    yield bytes.slice(57);
  }
  assert.equal(await read(parseNDJSON(chunks())), 'Hello 世界');
});

test('parser rejects truncated or synthetic success', async () => {
  async function* truncated() { yield enc.encode(entry('partial', false)); }
  await assert.rejects(read(parseNDJSON(truncated())), /completed response/);
  async function* malformed() { yield enc.encode('{"message":{"content":"hi"}}\n'); }
  await assert.rejects(read(parseNDJSON(malformed())), /done marker/);
  async function* fatal() { yield enc.encode('{"error":"backend failed"}\n'); }
  await assert.rejects(read(parseNDJSON(fatal())), /backend failed/);
  async function* oversized() { yield enc.encode('x'.repeat(128 * 1024 + 1)); }
  await assert.rejects(read(parseNDJSON(oversized())), /128 KiB/);
});

test('authenticated GET only connects to localhost and never puts token in URL', async () => {
  const old = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, DEFAULT_ENDPOINT + '/api/tags');
      assert.equal(options.headers.Authorization, 'Bearer ' + TOKEN);
      assert.equal(options.cache, 'no-store');
      assert.equal(options.redirect, 'error');
      return Response.json({ models: [{ name: 'registered-gguf' }] });
    };
    const response = await getJSON(DEFAULT_ENDPOINT, '/api/tags', TOKEN);
    assert.equal(response.models[0].name, 'registered-gguf');
    await assert.rejects(getJSON(DEFAULT_ENDPOINT, '/api/tags', ''), /token/i);
  } finally {
    globalThis.fetch = old;
  }
});

test('streamChat sends only configured model and text messages, with bearer header', async () => {
  const old = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, DEFAULT_ENDPOINT + '/api/chat');
      assert.equal(options.headers.Authorization, 'Bearer ' + TOKEN);
      assert.equal(options.headers['Content-Type'], 'application/json');
      const payload = JSON.parse(options.body);
      assert.equal(payload.model, 'registered-gguf');
      assert.equal(payload.stream, true);
      assert.deepEqual(payload.messages, [{ role: 'user', content: 'Hello' }]);
      assert.equal(payload.options.num_predict, 256);
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(enc.encode(entry('Real simulated transport', false)));
          controller.enqueue(enc.encode(entry('', true)));
          controller.close();
        },
      });
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } });
    };
    const output = await read(streamChat({
      endpoint: DEFAULT_ENDPOINT, token: TOKEN, model: 'registered-gguf',
      messages: [{ role: 'user', content: 'Hello' }],
    }));
    assert.equal(output, 'Real simulated transport');
  } finally {
    globalThis.fetch = old;
  }
});

test('HTTP auth failures are specific and cannot return success text', async () => {
  const old = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response('Unauthorized', { status: 401 });
    await assert.rejects(getJSON(DEFAULT_ENDPOINT, '/api/tags', TOKEN), /Invalid or missing bearer token/);
  } finally {
    globalThis.fetch = old;
  }
});

test('browser network restrictions explain local permission without token leakage', () => {
  const error = friendlyConnectionError(new TypeError('Failed to fetch'));
  assert.match(error, /same-device access/);
  assert.doesNotMatch(error, /token-example/);
  assert.equal(friendlyConnectionError({ name: 'AbortError' }), 'Request cancelled.');
});
