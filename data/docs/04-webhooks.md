# Webhooks

Webhooks let Nimbus notify your application about delivery events instead of you
polling the API.

## Events

Nimbus emits the following event types:

- `message.queued` — accepted and waiting to send.
- `message.delivered` — confirmed delivered to the recipient.
- `message.bounced` — permanently failed (for example, invalid address).
- `message.opened` — recipient opened the email (email channel only).

## Configuring an endpoint

Add an HTTPS endpoint at **Settings → Webhooks**. Nimbus sends a POST request
with a JSON body for each event. Your endpoint must respond with HTTP `200`
within 5 seconds; otherwise the delivery is considered failed.

## Retries

Failed webhook deliveries are retried with exponential backoff for up to 24
hours. After 24 hours of failures, the endpoint is automatically disabled and
the account owner is emailed.

## Verifying signatures

Each webhook request includes an `X-Nimbus-Signature` header containing an
HMAC-SHA256 signature of the raw request body, computed with your webhook
signing secret. Always verify this signature before trusting the payload to
prevent spoofed events. The signing secret is found on the Webhooks settings
page and can be rotated independently of your API keys.
