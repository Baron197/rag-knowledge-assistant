# Rate Limits

Nimbus enforces rate limits per API key to protect platform stability and your
sender reputation.

## Default limits by plan

- **Free:** 10 requests per second, 1,000 messages per month.
- **Growth:** 50 requests per second, 100,000 messages per month.
- **Scale:** 200 requests per second, custom monthly volume.

Limits are applied using a token-bucket algorithm, so short bursts above your
per-second rate are tolerated as long as the average stays within budget.

## Reading the headers

Every response includes rate-limit headers:

- `X-RateLimit-Limit` — your current per-second ceiling.
- `X-RateLimit-Remaining` — requests left in the current window.
- `X-RateLimit-Reset` — seconds until the window resets.

## Handling 429 responses

When you exceed the limit, Nimbus returns HTTP `429 Too Many Requests` with a
`Retry-After` header indicating how many seconds to wait. Clients should back
off exponentially and retry after the suggested delay. Messages that receive a
`429` are **not** queued — you must resend them.

## Requesting higher limits

Growth and Scale customers can request a temporary or permanent limit increase
from **Settings → Usage → Request increase**. Increases are typically approved
within one business day.
