# Rung 5 browser bridge: explicit opt-in only

PhiServe intentionally binds to `127.0.0.1`, even with browser bridge enabled. Browsers have an origin boundary, and local model APIs can expose sensitive prompts and responses. **CORS is not authentication.** Rung 5 therefore requires an operator-generated bearer token to authorize browser access.

## Operator setup (Windows PowerShell)

~~~powershell
phiserve auth create --output "$HOME\\.phiserve\\bridge-token.txt"
phiserve daemon --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --browser-origin "https://michaelwave369.github.io" --api-key-file "$HOME\\.phiserve\\bridge-token.txt"
~~~

The key file must be kept private. It is created exclusively; existing files cannot be overwritten. On POSIX it uses file mode 0600. Windows file security inherits parent ACLs, so store the token inside your private user profile and review Windows file permissions where appropriate.

For an authenticated local client:

~~~powershell
$token = (Get-Content "$HOME\\.phiserve\\bridge-token.txt" -Raw).Trim()
Invoke-RestMethod -Uri "http://127.0.0.1:11435/api/tags" -Headers @{ Authorization = "Bearer $token" }
~~~

A trusted browser uses an identical Bearer header. Browser mode protects GET and POST routes, including health and model listing. OPTIONS preflight is anonymous but only grants GET and POST from the **exact authorized Origin**, with Authorization and Content-Type request headers. If the browser sends `Access-Control-Request-Private-Network: true`, the server grants it only for this pre-approved origin. No wildcard, opaque origin, cross-site credentials, arbitrary headers, or network interface binding is allowed. Cross-origin callers from other origins are denied even with a token in browser mode. Without the token, requests receive 401.

An HTTPS-hosted browser UI may be able to reach `http://127.0.0.1:11435` **on the same physical device**, depending on browser and local-network permission policies. This does not expose your PC to a phone or Quest on your LAN. A future trusted gateway must separately implement TLS, user authorization, network policy, revocation, and safety controls.

The token is a **bearer secret**: any code that can read it can use your local service. Treat local browser applications and extensions as trusted only after review. Never put the token in URLs, analytics, logs, public screenshots, or source control. Rotating the token requires shutting down the daemon and starting with a different key file (no live key rotation yet).

## Limitations

- No remotely reachable socket, mutual TLS, authenticated identity, session-scoped model permission, or per-app rate limit.
- CORS is a browser read boundary, not an authorization mechanism; the token protects local HTTP requests.
- Browser Private Network Access / Local Network Access compatibility is platform/browser-dependent and **not validated on user hardware by CI**.
- Upstream still relies on a separate real llama.cpp installation and compatible local GGUF.
