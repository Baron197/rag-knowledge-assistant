# Getting Started with Nimbus

Nimbus is a cloud API platform for sending transactional email, SMS, and push
notifications from a single API. This guide gets you from zero to your first
delivered message in about five minutes.

## Create an account

Sign up at https://app.nimbus.dev/signup using your work email. New accounts
start on the **Free** plan, which includes 1,000 messages per month at no cost.
You must verify your email address before you can send live messages; until
then, your account operates in **sandbox mode** and messages are logged but not
delivered.

## Send your first message

1. Open the dashboard and copy your **test** API key from **Settings → API Keys**.
2. Make a POST request to `https://api.nimbus.dev/v1/messages` with the key in
   the `Authorization` header.
3. Include `channel`, `to`, and `body` fields in the JSON request.

A successful request returns HTTP `202 Accepted` with a `message_id` you can use
to track delivery. Messages sent with a test key always appear in the **Logs**
tab but are never delivered to real recipients.

## Going live

To send to real recipients, switch from your test key to a **live** key and
ensure your account email is verified. Live keys are only shown once at creation
time, so store them in a secret manager rather than in source control.

## Where to next

- Authentication and key rotation: see "Authentication".
- Limits on how fast you can send: see "Rate Limits".
- Getting notified about delivery: see "Webhooks".
