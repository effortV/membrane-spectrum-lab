# Community Cloud through a private network and restricted OpenSSH

Target: GitHub + Streamlit Community Cloud (*.streamlit.app).
Data and computation stay on the Windows server. Both interfaces use one
authoritative database through the server API. SSH encryption does not make
RFC1918 addresses reachable from the Internet.

## Optional private-network route

Tailscale can provide reachability across NAT. The application still uses
regular, restricted Windows OpenSSH, not the Tailscale SSH server feature.

- Install an official signed client; preserve existing DNS and routes.
- The owner personally authorizes the server device in the intended account.
- Do not use an exit node, advertise the LAN, or expose a public Funnel.
- A Cloud Linux userspace daemon may provide a loopback-only SOCKS5 proxy.
  This must be provisioned and tested on the actual Community Cloud runtime.
- SOCKS5 applies only to SSH, never ALL_PROXY for provider/literature requests.
- Pin the server host key before public-key authentication.
- Use a dedicated non-administrator SSH identity, never provider API keys or
  an administrator private key in the cloud.
- Scope network grants to cloud identity -> server SSH port only; do not
  overwrite an existing shared policy or leave wildcard access to the server.
- Restricted OpenSSH permits only local forwarding to 127.0.0.1:8771; shell,
  SFTP, reverse forwarding, passwords and other target ports stay disabled.
- Credentials belong in Streamlit Secrets/protected server files, never GitHub,
  prompts, diagnostic output or logs.

## Separate publication from backend activation

The frontend can be published before the server route is ready. Without valid
secrets and connectivity it displays a connection-setup message and starts no
database or research task. Before adding live credentials, verify app access is
limited to trusted viewers and test real Cloud -> SSH -> FastAPI reads.
Do not infer Internet reachability from a LAN-only SSH test.

Device authorization, persistent auth-key creation and network-policy changes
require the account owner's participation. Check plan eligibility and allocations;
never automatically buy a subscription or promise unlimited free infrastructure.
Handle credential expiry explicitly; failure must not cause paid upgrades or
automatic model-job resubmission.

## Primary references

- https://tailscale.com/docs/install/windows/msi
- https://tailscale.com/docs/concepts/userspace-networking
- https://tailscale.com/docs/features/access-control/auth-keys
- https://tailscale.com/pricing
- https://www.iana.org/assignments/iana-ipv4-special-registry
