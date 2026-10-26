import assert from 'node:assert/strict'
import { afterEach, describe, it } from 'node:test'
import { classifyProbe, DiffusionCapability, type ProbeTarget } from '../src/capability.ts'

const realFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = realFetch })

/** Verbatim Inception errors, captured from the live endpoint through a gateway. */
const BASELINE = '{"error":{"code":"invalid_request_error","message":"inception: 400 Bad Request: {\\"error\\":{\\"message\\":[{\\"type\\":\\"value_error\\",\\"loc\\":[\\"body\\",\\"messages\\"],\\"msg\\":\\"Value error, At least one message is required\\"}]}}"}}'
const KNOWS_FIELD = '{"error":{"code":"invalid_request_error","message":"inception: 400 Bad Request: {\\"error\\":{\\"message\\":[{\\"type\\":\\"value_error\\",\\"loc\\":[\\"body\\",\\"messages\\"]},{\\"type\\":\\"bool_parsing\\",\\"loc\\":[\\"body\\",\\"diffusing\\"],\\"msg\\":\\"Input should be a valid boolean, unable to interpret input\\"}]}}"}}'
const EXTRA_KEY = '{"detail":[{"type":"extra_forbidden","loc":["body","diffusing"],"msg":"Extra inputs are not permitted"}]}'

describe('classifyProbe', () => {
  it('confirms a route that validates the field', () => {
    assert.equal(classifyProbe(400, KNOWS_FIELD), 'yes')
  })

  it('refuses a strict route that calls the field an extra key', () => {
    assert.equal(classifyProbe(422, EXTRA_KEY), 'no')
  })

  it('stays undecided when the field is simply ignored', () => {
    assert.equal(classifyProbe(400, BASELINE), 'unknown')
  })

  it('stays undecided on auth failures and unexpected successes', () => {
    assert.equal(classifyProbe(401, '{"error":{"message":"Incorrect API key provided"}}'), 'unknown')
    assert.equal(classifyProbe(200, 'data: {"choices":[]}'), 'unknown')
  })

  it('stays undecided on transport-level server errors', () => {
    assert.equal(classifyProbe(500, '{"error":{"message":"boom"}}'), 'unknown')
  })
})

describe('DiffusionCapability', () => {
  const TARGET: ProbeTarget = { provider: 'gw', model: 'mercury-3', baseURL: 'http://127.0.0.1:9/v1', apiKey: 'sk-test' }

  it('drops an accepted probe instead of consuming its stream', async () => {
    const encoder = new TextEncoder()
    let closeBody = (): void => {}
    globalThis.fetch = async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[]}\n\n'))
        // Stays open, like an SSE generation the endpoint just started.
        closeBody = () => { try { controller.close() } catch (_alreadyClosed) {} }
      },
    }), { status: 200, headers: { 'content-type': 'text/event-stream' } })

    const capability = new DiffusionCapability()
    const verdict = await Promise.race([
      capability.verdict(TARGET),
      new Promise<'timeout'>(resolve => { setTimeout(() => { resolve('timeout') }, 400) }),
    ])
    closeBody()
    // A 2xx means the endpoint started a generation; it must be abandoned
    // immediately rather than read to completion (which would bill tokens).
    assert.equal(verdict, 'unknown')
  })
})
