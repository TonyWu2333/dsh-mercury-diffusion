import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { LlmError } from '@deepseek-ai/dsh-llm'
import { readChunk } from '../src/sse.ts'

const malformed = (error: unknown) => error instanceof LlmError && error.code === 'MALFORMED_RESPONSE'

describe('readChunk', () => {
  it('reads content, tool calls and finish reason from the first choice', () => {
    const chunk = readChunk({
      choices: [{
        delta: { content: 'hi', tool_calls: [{ id: 'call_1', function: { name: 'search', arguments: '{"q"' } }] },
        finish_reason: 'tool_calls',
      }],
    })
    assert.deepEqual(chunk, {
      content: 'hi',
      toolCalls: [{ index: 0, id: 'call_1', name: 'search', arguments: '{"q"' }],
      finishReason: 'tool_calls',
    })
  })

  it('reads a usage-only chunk with nested token details', () => {
    const chunk = readChunk({
      choices: [],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        prompt_tokens_details: { cached_tokens: 4 },
        completion_tokens_details: { reasoning_tokens: 2 },
      },
    })
    assert.deepEqual(chunk, {
      toolCalls: [],
      usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15, cachedTokens: 4, reasoningTokens: 2 },
    })
  })

  it('rejects malformed fields', () => {
    assert.throws(() => readChunk([]), malformed)
    assert.throws(() => readChunk({ choices: {} }), malformed)
    assert.throws(() => readChunk({ choices: [{ finish_reason: 'aborted' }] }), malformed)
    assert.throws(() => readChunk({ choices: [{ delta: { content: 42 } }] }), malformed)
    assert.throws(() => readChunk({ choices: [{ delta: { tool_calls: [{ index: -1 }] } }] }), malformed)
    assert.throws(() => readChunk({ usage: { prompt_tokens: 1.5 } }), malformed)
  })

  it('classifies an in-band error object', () => {
    assert.throws(
      () => readChunk({ error: { message: 'Rate limit reached', type: 'rate_limit_error' } }),
      (error: unknown) => error instanceof LlmError && error.code === 'RATE_LIMIT',
    )
  })
})
