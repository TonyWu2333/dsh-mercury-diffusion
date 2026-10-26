/** Map a provider-neutral request onto an Inception streaming chat-completions body. */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, RequestMessage } from '@deepseek-ai/dsh-llm'
import type { WireMessage, WireRequest, WireToolCall } from './wire-types.ts'

/** Sampling range accepted by Mercury chat models. */
const TEMPERATURE_MIN = 0.5
const TEMPERATURE_MAX = 1

function unsupported(what: string): never {
  throw new LlmError(`Mercury chat completions cannot represent ${what}`, 'UNSUPPORTED_CONTENT')
}

/**
 * Join the text blocks of user, system, or tool-result content. Reasoning and
 * tool-call blocks carry no input meaning there and are skipped; image and file
 * blocks never arrive because the route declares text-only input and request
 * assembly projects files to handle text.
 */
function inputText(blocks: readonly ContentBlock[], role: string): string {
  const parts: string[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text.length > 0) parts.push(block.text)
        break
      case 'reasoning':
      case 'tool-call':
        break
      default:
        return unsupported(`${role} content ${block.type}`)
    }
  }
  return parts.join('\n\n')
}

function assistantMessage(blocks: readonly ContentBlock[]): WireMessage | undefined {
  const text: string[] = []
  const calls: WireToolCall[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text.length > 0) text.push(block.text)
        break
      // Mercury reasons internally and accepts no reasoning replay.
      case 'reasoning':
        break
      case 'tool-call':
        calls.push({ id: block.id, type: 'function', function: { name: block.name, arguments: block.arguments } })
        break
      default:
        return unsupported(`assistant content ${block.type}`)
    }
  }
  if (text.length === 0 && calls.length === 0) return undefined
  return {
    role: 'assistant',
    content: text.length === 0 ? null : text.join(''),
    ...calls.length === 0 ? {} : { tool_calls: calls },
  }
}

function wireMessage(message: RequestMessage): WireMessage | undefined {
  switch (message.role) {
    case 'system': {
      const content = inputText(message.content, 'system')
      return content.length === 0 ? undefined : { role: 'system', content }
    }
    case 'user': {
      const content = inputText(message.content, 'user')
      return content.length === 0 ? undefined : { role: 'user', content }
    }
    case 'assistant':
      return assistantMessage(message.content)
    case 'tool':
      // An empty result still answers its call id.
      return { role: 'tool', tool_call_id: message.toolCallId, content: inputText(message.content, 'tool-result') }
    // The route has no tool-update mode, so the complete current declarations
    // are sent in `tools`; in-history tool changes have no wire meaning. dsh's
    // own projection strips these, but it runs after this interceptor.
    case 'developer':
      return undefined
    default:
      return assertNever(message)
  }
}

function assertNever(value: never): never {
  throw new Error(`unexpected message role ${JSON.stringify(value)}`)
}

/** Per-request transport choices resolved by the bridge. */
export interface SerializeOptions {
  /** Ask for complete-text denoising frames instead of deltas. */
  readonly diffusing: boolean
}

/**
 * Serialize one complete streaming request.
 * @param options - provider-neutral request after runtime projection.
 * @param transport - diffusion choice for this request.
 * @returns the chat-completions JSON body.
 * @throws {LlmError} `UNSUPPORTED_CONTENT` or `UNSUPPORTED_OPTION` for values Mercury cannot honor.
 */
export function serialize(options: GenerateOptions, transport: SerializeOptions): WireRequest {
  const messages: WireMessage[] = []
  if (options.system !== undefined && options.system.length > 0) messages.push({ role: 'system', content: options.system })
  for (const message of options.messages) {
    const wire = wireMessage(message)
    if (wire !== undefined) messages.push(wire)
  }
  if (options.temperature !== undefined && (options.temperature < TEMPERATURE_MIN || options.temperature > TEMPERATURE_MAX)) {
    throw new LlmError(
      `Mercury accepts temperature from ${TEMPERATURE_MIN} to ${TEMPERATURE_MAX}; got ${options.temperature}`,
      'UNSUPPORTED_OPTION',
    )
  }
  const effort = options.reasoningEffort === undefined ? undefined : String(options.reasoningEffort)
  return {
    model: options.model,
    messages,
    stream: true,
    ...transport.diffusing ? { diffusing: true as const } : {},
    stream_options: { include_usage: true },
    ...options.maxTokens === undefined ? {} : { max_completion_tokens: options.maxTokens },
    ...options.temperature === undefined ? {} : { temperature: options.temperature },
    ...options.stop === undefined || options.stop.length === 0 ? {} : { stop: options.stop },
    ...effort === undefined ? {} : { reasoning_effort: effort },
    ...options.tools === undefined || options.tools.length === 0 ? {} : {
      tools: options.tools.map(tool => ({
        type: 'function' as const,
        function: { name: tool.name, description: tool.description, parameters: tool.parameters },
      })),
    },
  }
}
