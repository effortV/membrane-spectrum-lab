# Streamlit Cloud UI -> pinned SSH -> server FastAPI

This is the user's requested deployment architecture. Community Cloud runs
`streamlit_cloud.py` and installs ONLY `requirements.txt`; Windows runs the full
project from `pyproject.toml`. Neither PDFs/NF/RO data nor provider API keys belong
in GitHub or Cloud Secrets. There is ONE server SQLite database, not a cloud copy.

The cloud client opens an authenticated, fingerprint-pinned `direct-tcpip` SSH
channel to server `127.0.0.1:8771`. It uses HTTP inside this encrypted channel;
FastAPI is never bound to the LAN or Internet. No user-owned domain is used.
The non-administrator SSH account `xpscloud` cannot run shell/SFTP sessions,
forward to other ports, use password authentication, or open reverse tunnels.

## Server startup

Provision a token on the server (its value is deliberately not printed):

```powershell
Set-Location D:\zzh\XPS-agent
& .\.venv\Scripts\python.exe .\scripts\provision-backend.py
```

The token file is under the authoritative data workspace:
`D:\data\XPS-agent\workspace\state\connections\backend_token.txt`.
It is protected with Windows ACLs. Do not put its contents in chat or logs.

Start both API and local UI in ONE worker process, including before a browser
visits the local UI:

```powershell
& .\.venv\Scripts\python.exe .\scripts\run-server.py --address 127.0.0.1 --port 8501
```

The existing hidden `run-ui-task.ps1`/`serve-ui.py` launcher also uses this entry.
The old direct Streamlit command remains supported, but initializes the API
only when the first app session runs. For Cloud availability use the combined
launcher. Do not start a separate multi-worker uvicorn process with this DB.

Local UI and API share BackgroundTasks, the paid-operation lock, and the
conversation runtime ID. Idempotency receipts persist in `backend_requests`;
an uncertain HTTP write or server restart NEVER resubmits paid work automatically.
Progress/history, literature batch receipts and conversation context are stored
on the same server. Model output uses the existing math-format repair code.

## Cloud configuration

Create a private code-only GitHub repository, deploy `streamlit_cloud.py` on
Community Cloud, and verify app access is private/trusted before granting access
to scientific data or billable actions. Repository privacy alone does not prove
app privacy. Each model action requires confirmation, but a shared backend token
is not a per-user authorization system for arbitrary public visitors.

Use `.streamlit/secrets.example.toml` as a template. Actual Secrets contain ONLY
the dedicated SSH identity, verified host fingerprint, server-reachable address,
SSH port, and backend token. An administrator key is never used by Cloud.
Connection failure stops the frontend without starting any local database,
model job or scientific compute. No global proxy is configured for literature
or SiliconFlow traffic.

SSH encryption is not NAT traversal. An RFC1918 address is still private even
when displayed on the router WAN page. A Cloud-reachable SSH route is necessary.
If the authorized Tailscale private-network approach is used, see SSH_NETWORK.md;
the optional loopback SOCKS5 settings are supported by the SSH client, but account
enrollment, userspace daemon provisioning, scoped network policy and actual
Community Cloud reachability must be verified separately. Do not infer cloud
connectivity from LAN-only SSH tests.

## Validation and activation

Run tests against a synthetic isolated staging directory before activation.
Set the existing migration hold, verify no task is running, back up code and
the live database consistently, then validate exact process and source ownership
before restarting the project service. Do not kill unrelated Python processes.
Verify health, authentication, pinned SSH reads, access restrictions, and unchanged
scientific rows/model-call counts. Remove the maintenance hold only as part of
a verified activation or restoration. Keep backups under D:\data, not GitHub.

Primary documentation:

- https://fastapi.tiangolo.com/advanced/events/
- https://docs.paramiko.org/en/stable/api/transport.html
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
