# Authentication

Nimbus uses API keys to authenticate every request. There is no username or
password on the API itself.

## API key types

- **Test keys** (`nk_test_…`) operate in sandbox mode. Requests are validated and
  logged, but no message is delivered. Use them in development and CI.
- **Live keys** (`nk_live_…`) send real messages and count against your plan
  quota and billing.

## Passing your key

Send the key as a Bearer token in the `Authorization` header:

```
Authorization: Bearer nk_live_xxxxxxxxxxxxxxxx
```

Requests without a valid key return HTTP `401 Unauthorized`. Requests with a key
that lacks permission for the requested action return HTTP `403 Forbidden`.

## Rotating and revoking keys

You can create multiple keys and give each a label (for example, "production
worker" or "staging"). To rotate a key, create a new one, deploy it, and then
delete the old key from **Settings → API Keys**. Deleting a key takes effect
immediately and any request using it afterward returns `401`.

If a key is leaked, revoke it immediately by deleting it. Nimbus also supports
**scoped keys** that can be restricted to specific channels (email only, for
example) to limit blast radius.

## Resetting a compromised key

If you believe a key has been exposed, delete it from the dashboard and create a
replacement. There is no "reset" that keeps the same key string — rotation
always produces a new value. Update your secret manager and redeploy.
