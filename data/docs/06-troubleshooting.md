# Troubleshooting

## My messages are not being delivered

First check whether you are using a **test** key. Test keys (`nk_test_…`) log
messages in the **Logs** tab but never deliver them. Switch to a live key and
make sure your account email is verified.

## I get 401 Unauthorized

This means the API key is missing, malformed, or has been deleted. Confirm the
key is sent as `Authorization: Bearer <key>` and that it has not been rotated
out. Deleted keys stop working immediately.

## I get 402 Payment Required

You have hit your monthly spend cap or a payment has failed. Raise or remove the
cap under Settings → Billing, or update your payment method.

## I get 429 Too Many Requests

You are sending faster than your plan's per-second limit. Respect the
`Retry-After` header and back off exponentially. Consider requesting a limit
increase if this happens consistently.

## Webhook events are not arriving

Verify your endpoint returns HTTP `200` within 5 seconds and is reachable over
HTTPS. Endpoints that fail for 24 hours are automatically disabled — re-enable
them from Settings → Webhooks after fixing the issue.

## Email opens are not tracked

Open tracking is available on the email channel only and requires the recipient
to load images. Plain-text emails and recipients with images disabled will not
generate `message.opened` events.
