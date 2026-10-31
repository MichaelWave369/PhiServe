# Φ PhiServe React Console

A static, responsive React interface for the local-only PhiServe model daemon. It can be hosted on GitHub Pages as **HTML/JS/CSS only**. The hosted website does not serve models, retain a token, or connect to a central inference API.

Expected public site **after deployment is enabled**:

https://michaelwave369.github.io/PhiServe/

This is a *project Pages path*, with its own directory but the same **origin** as other projects under \`michaelwave369.github.io\`. PhiServe's CORS allowlist is origin-scoped, **not path-scoped**. Use the trust boundary intentionally.

## What works without PhiServe

- Fully rendered responsive console and orbital architecture overview.
- **Clearly labeled demo** model library, chat interactions and setup/field instructions.
- Demo does not invoke an LLM or fabricate local inference.
- Source and documentation links.

## What works after a local daemon is authorized

- Connect with \`http://127.0.0.1:11435\` (or an explicit local loopback URL with a port).
- GET \`/health\`, \`/api/tags\`, \`/api/ps\`; inspect real registered GGUF and currently resident model.
- Select a registered model, send text prompts to \`POST /api/chat\`, decode real NDJSON frames, and cancel a streaming request.
- Refresh local state. A 409 means an active different model is busy; 503 indicates a transition or unavailable backend.
- Token is held in React state **only**. It is never written to localStorage, sessionStorage, cookies, URL query strings, or the repository.
- A browser refresh or Disconnect clears the token. No automatic network calls are made on initial page load.

## Starting the daemon on your own computer

First register an already-held licensed GGUF and ensure a separately installed, trusted \`llama-server\` binary is available. Then run:

~~~powershell
phiserve models list
phiserve auth create --output "$HOME\.phiserve\bridge-token.txt"
phiserve daemon --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20 --browser-origin "https://michaelwave369.github.io" --api-key-file "$HOME\.phiserve\bridge-token.txt"
~~~

Do **not** copy a real token into GitHub source, screenshots, browser developer-console scripts, URL parameters, or support tickets. Paste it into the local React console's masked input on the **same PC** running PhiServe.

The server continues to bind **only 127.0.0.1**. An HTTPS page may encounter browser Local Network Access / Private Network Access restrictions and permission prompts. This is browser- and OS-dependent, and not verified on real hardware in CI. A phone or Quest visiting GitHub Pages cannot access a PC's \`127.0.0.1\`; the loopback resolves to the phone/Quest instead. A separately designed TLS-authenticated gateway is needed for remote-device use.

**Security:** Hosting under GitHub Pages shares the \`michaelwave369.github.io\` origin with other projects. Review all applications and scripts you host under that origin. The bearer token is powerful. Do not embed untrusted scripts or broad analytics in this console. The frontend includes **no third-party font, tracking, image, or inference requests**.

## Development

From this directory:

~~~bash
npm install --no-audit --no-fund
npm run test
npm run dev
npm run build
~~~

The Vite production base path is \`/PhiServe/\`. For local Vite development, the daemon must trust the dev server's **exact** origin (for example \`http://127.0.0.1:5173\`), not the GitHub Pages origin. Stop the production daemon and restart it with that origin explicitly if necessary.

## Deployment

The \`.github/workflows/web-pages.yml\` workflow builds the console and publishes \`web/dist\` to GitHub Pages **on pushes to main**. The \`.github/workflows/web-ci.yml\` workflow checks the tests and build on pull requests. If GitHub Pages isn't configured for this repo, choose **Settings → Pages → Build and deployment → GitHub Actions**. The workflow also requests Pages enablement.

Opening a PR is **not** publishing. Confirm the deploy workflow succeeds after merge before claiming the URL is live.

## What is intentionally not here

There is no model weight upload/download, no experimental PhiStream chat adapter, no remote LAN endpoint, no browser-run \`nvidia-smi\`, no background GPU telemetry, and no promise of CUDA hardware validation. The field lab only presents local operator commands and links to the real hardware acceptance docs.
