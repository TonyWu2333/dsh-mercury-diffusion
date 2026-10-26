import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { LlmError, ReasoningEffortId, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { serialize } from '../src/serialize.ts'

/** Loop-built histories carry durable fields the serializer never reads; request inputs cover the cases. */
function request(messages: unknown[], extra: Partial<GenerateOptions> = {}): GenerateOptions {
  return { provider: 'inception', model: 'mercury-3', messages: messages as GenerateOptions['messages'], ...extra }
}

describe('serialize', () => {
  it('maps every role and keeps tool calls with their results', () => {
    const body = serialize(request([
      { role: 'system', content: [{ type: 'text', text: 'Be brief.' }] },
      { role: 'user', content: [{ type: 'text', text: 'List files' }] },
      { role: 'assistant', content: [
        { type: 'reasoning', text: 'hidden' },
        { type: 'text', text: 'Checking.' },
        { type: 'tool-call', id: ToolCallId('call_1'), name: 'bash', arguments: '{"command":"ls"}' },
      ] },
      { role: 'tool', toolCallId: ToolCallId('call_1'), content: [{ type: 'text', text: 'a.md' }] },
    ]), { diffusing: false })
    assert.deepEqual(body.messages, [
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'List files' },
      { role: 'assistant', content: 'Checking.', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: '{"command":"ls"}' } }] },
      { role: 'tool', tool_call_id: 'call_1', content: 'a.md' },
    ])
    assert.equal(body.diffusing, undefined)
    assert.deepEqual(body.stream_options, { include_usage: true })
  })

  it('requests diffusing and forwards controls and tools', () => {
    const body = serialize(request([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }], {
      system: 'sys', maxTokens: 512, temperature: 0.7, stop: ['END'], reasoningEffort: ReasoningEffortId('high'),
      tools: [{ name: 'bash', description: 'Run a command', parameters: { type: 'object' } }],
    }), { diffusing: true })
    assert.equal(body.diffusing, true)
    assert.equal(body.messages[0]?.role, 'system')
    assert.equal(body.max_completion_tokens, 512)
    assert.equal(body.temperature, 0.7)
    assert.deepEqual(body.stop, ['END'])
    assert.equal(body.reasoning_effort, 'high')
    assert.deepEqual(body.tools, [{ type: 'function', function: { name: 'bash', description: 'Run a command', parameters: { type: 'object' } } }])
  })

  it('omits developer tool-update messages because the complete tools are sent', () => {
    const body = serialize(request([
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { role: 'developer', content: [{ type: 'tool-addition', toolName: 'bash' }] },
    ], {
      tools: [{ name: 'bash', description: 'Run a command', parameters: { type: 'object' } }],
    }), { diffusing: true })
    assert.deepEqual(body.messages, [{ role: 'user', content: 'hi' }])
    assert.deepEqual(body.tools, [{
      type: 'function',
      function: { name: 'bash', description: 'Run a command', parameters: { type: 'object' } },
    }])
  })

  it('rejects values Mercury cannot honor', () => {
    const user = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]
    assert.throws(() => serialize(request(user, { temperature: 0.2 }), { diffusing: false }),
      (error: unknown) => error instanceof LlmError && error.code === 'UNSUPPORTED_OPTION')
    assert.throws(() => serialize(request([{ role: 'user', content: [{ type: 'image', attachment: {} }] }]), { diffusing: false }),
      (error: unknown) => error instanceof LlmError && error.code === 'UNSUPPORTED_CONTENT')
  })
})
