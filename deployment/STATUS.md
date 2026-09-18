# Deployment status

Target: GitHub + Streamlit Community Cloud, with streamlit_cloud.py on main.
The cloud is a thin interface only. Data, provider credentials, paid jobs and
the authoritative database remain on the Windows server.

The server FastAPI is active on loopback, sharing one process and runtime with
the local UI. Restricted, pinned SSH reads and all seven cloud-interface pages
passed LAN smoke checks. The isolated full regression passed 169 tests; no paid
model calls were submitted. LAN checks do not prove Community Cloud reachability.

This source release contains no scientific data, database, actual credentials
or machine-specific network addresses. Connection templates are placeholders.
Missing secrets or an unreachable server stops the frontend without creating
a replacement database or automatically resubmitting paid actions.

Cloud publication and end-to-end server connectivity are separate milestones.
Verify the deployed app, trusted viewer access, actual connection credentials
and Cloud-to-server SSH reachability before enabling research operations.

Operational backups, device enrollment receipts and credentials stay outside
this repository. No user-owned domain or public database is required.
