# Billing & Plans

## Plans

- **Free** — $0/month, 1,000 messages, community support.
- **Growth** — $49/month, 100,000 messages included, email support, then
  $0.40 per 1,000 additional messages.
- **Scale** — custom pricing, volume discounts, 99.95% uptime SLA, and a
  dedicated support engineer.

## How usage is counted

One outbound message to one recipient counts as one message, regardless of
channel. A single API request that fans out to 50 recipients counts as 50
messages. Messages sent with test keys are never billed.

## Overages

On the Growth plan, sending beyond your included 100,000 messages is billed at
$0.40 per 1,000 messages and added to your next invoice. There is no hard stop
by default, but you can set a **monthly spend cap** under **Settings → Billing**;
once the cap is reached, further live sends return HTTP `402 Payment Required`
until the next billing cycle or until you raise the cap.

## Changing plans

Upgrades take effect immediately and are prorated. Downgrades take effect at the
start of your next billing cycle so you keep paid features until then.

## Invoices and payment

Invoices are issued monthly and charged to the card on file. You can download
PDF invoices and update payment details under **Settings → Billing**. Failed
payments are retried over 7 days before the account is suspended.
