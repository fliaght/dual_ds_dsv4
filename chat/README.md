# Chat UI

Streamlit front-end for the dual-Spark OpenAI-compatible API. This directory
owns the UI, FIFO queue, browser history, streaming client, host monitor,
server-side logs and tests.

- [Run, configure and inspect logs](../Docs/runbooks/chat.md)
- [Module responsibilities](../Knowledge/code-layout.md)
- [Offline and live checks](../Docs/runbooks/checks.md)
- [Project status](../Docs/current.md)

Commands run from the repository root. The model must already be running for
real chats. History resides in each browser; complete conversations are also
logged on the server.
