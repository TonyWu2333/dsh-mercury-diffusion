/**
 * Zero-cost capability probe for the diffusion transport.
 *
 * Whether a route honors `diffusing` is a property of the upstream, not of the
 * model name. Probing it with a real generation would bill tokens, so the
 * question is instead answered at the validation layer: the probe sends one
 * request that is invalid for a structural reason (`messages: []`) together
 * with `diffusing` carrying a deliberately wrong TYPE. A server that knows the
 * field reports a type error naming it; a server that does not either says
 * nothing about it or rejects it as an extra key. Either way the request is
 * refused before any token exists.
 *
 * Confirmed against Inception (direct and through the Magpie gateway):
 *   messages: []                                    -> 400 At least one message is required
 *   + diffusing: "probe"                            -> 400 ... loc ["body","diffusing"] bool_parsing
 */

import { attributionHeaders } from '@deepseek-ai/dsh-llm'

/** One endpoint/model pair to interrogate. */
export interface ProbeTarget {
  /** Provider route key, part of the cache identity. */
  provider: string
  /** Model id sent as `model`. */
  model: string
  /** API root serving `/chat/completions`. */
  baseURL: string
  /** Bearer token for the probe. */
  apiKey: string
}

/** What one probe concluded. `unknown` means the endpoint could not answer. */
export type DiffusionVerdict = 'yes' | 'no' | 'unknown'

/** Wrong-typed value for `diffusing`; only a route that knows the field complains about it. */
const WRONG_TYPE = 'probe-not-a-boolean'

/** Give up on a probe that says nothing within this window. */
const PROBE_TIMEOUT_MS = 15_000

/** How long an inconclusive probe is trusted before the next conversation re-asks. */
const UNKNOWN_TTL_MS = 60_000

/** How much of the error body to inspect. */
const MAX_BODY = 8_000

/** A validation error whose location names the `diffusing` body field. */
const NAMES_FIELD = /"loc"\s*:\s*\[\s*"body"\s*,\s*"diffusing"\s*\]|body\.diffusing/iu

/** A strict-schema refusal of an unrecognized key. */
const EXTRA_KEY = /extra_(forbidden|inputs|fields)|unexpected keyword|unknown field/iu

/** A type complaint, which only an endpoint that validates the field can produce. */
const TYPE_COMPLAINT = /bool|boolean|parsing|not a valid|invalid type|type_error/iu

/**
 * Classify one probe response.
 * @param status - HTTP status of the probe.
 * @param body - response text, already capped.
 * @returns the verdict this response proves.
 */
export function classifyProbe(status: number, body: string): DiffusionVerdict {
  if (status === 401 || status === 403) return 'unknown'
  // A 2xx means the endpoint accepted an empty conversation, so the probe never
  // reached the validation layer it relies on; never treat that as evidence.
  if (status >= 200 && status < 300) return 'unknown'
  if (EXTRA_KEY.test(body)) return NAMES_FIELD.test(body) ? 'no' : 'unknown'
  if (NAMES_FIELD.test(body)) return 'yes'
  if (/diffusing/iu.test(body) && TYPE_COMPLAINT.test(body)) return 'yes'
  return 'unknown'
}

/**
 * Per-route diffusion capability, probed on demand and cached for the process.
 * Concurrent callers share one in-flight probe so a burst of requests costs one.
 */
export class DiffusionCapability {
  private readonly verdicts = new Map<string, { verdict: DiffusionVerdict; expiresAt: number }>()
  private readonly inflight = new Map<string, Promise<DiffusionVerdict>>()

  private static key(target: { provider: string; model: string }): string {
    return `${target.provider}::${target.model}`
  }

  /**
   * Read the cached verdict without probing.
   * @param provider - provider route key.
   * @param model - model id.
   * @returns the last verdict, or undefined when the pair was never probed.
   */
  verdictOf(provider: string, model: string): DiffusionVerdict | undefined {
    const cached = this.verdicts.get(`${provider}::${model}`)
    if (cached === undefined) return undefined
    if (cached.expiresAt > Date.now()) return cached.verdict
    this.verdicts.delete(`${provider}::${model}`)
    return undefined
  }

  /**
   * The verdict for one target, probing when it is not cached yet.
   * @param target - endpoint, model, and credential.
   * @returns the settled verdict; `unknown` is cached briefly and retried after.
   */
  async verdict(target: ProbeTarget): Promise<DiffusionVerdict> {
    const key = DiffusionCapability.key(target)
    const cached = this.verdictOf(target.provider, target.model)
    if (cached !== undefined) return cached
    const running = this.inflight.get(key)
    if (running !== undefined) return await running
    const pending = this.probe(target).then((verdict) => {
      // A measured property of the route holds for the process. An evasive or
      // unreachable endpoint may answer differently next time, so it is kept
      // only long enough to stop every conversation open from re-probing — an
      // ignored `diffusing` is the common OpenAI-compatible case.
      this.verdicts.set(key, {
        verdict,
        expiresAt: verdict === 'unknown' ? Date.now() + UNKNOWN_TTL_MS : Number.POSITIVE_INFINITY,
      })
      return verdict
    }).finally(() => { this.inflight.delete(key) })
    this.inflight.set(key, pending)
    return await pending
  }

  /**
   * Send the validation-only probe and classify the refusal.
   * @param target - endpoint, model, and credential.
   * @returns acceptance, refusal, or `unknown`.
   */
  private async probe(target: ProbeTarget): Promise<DiffusionVerdict> {
    const controller = new AbortController()
    const timer = setTimeout(() => { controller.abort(new Error('probe timeout')) }, PROBE_TIMEOUT_MS)
    try {
      const response = await fetch(`${target.baseURL.replace(/\/+$/u, '')}/chat/completions`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          ...attributionHeaders(),
          'content-type': 'application/json',
          accept: 'application/json',
          authorization: `Bearer ${target.apiKey}`,
        },
        // Both parts matter: empty messages make the request refusable before
        // generation, and the wrong type is what only a knowing route reports.
        body: JSON.stringify({
          model: target.model,
          messages: [],
          stream: true,
          diffusing: WRONG_TYPE,
        }),
      })
      // A 2xx would mean the endpoint started a generation; drop it immediately
      // and without reading the body, which for a streaming endpoint would
      // otherwise run to completion and bill tokens.
      if (response.ok) {
        void response.body?.cancel().catch(() => {})
        controller.abort(new Error('probe accepted; cancelled'))
        return 'unknown'
      }
      const body = (await response.text()).slice(0, MAX_BODY)
      return classifyProbe(response.status, body)
    } catch (_probeFailed) {
      return 'unknown'
    } finally {
      clearTimeout(timer)
    }
  }
}
