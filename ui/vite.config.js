import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { spawn, execSync } from 'node:child_process'
import http from 'node:http'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const BACKEND_PORT = 8741
const BACKEND_DIR = resolve(__dirname, '..')  // repo root (one level up from ui/)

// ── Jarvis Launcher Plugin ────────────────────────────────────────────
// Manages the Jarvis backend lifecycle from the Vite dev server.
// Intercepts /api/v1/system/* so the UI can start/stop Jarvis.
// All other /api requests are proxied to the backend on port 8741.
function jarvisLauncher() {
  let backendProcess = null

  // ── Port-level process detection ──────────────────────────────────
  // Finds the PID listening on BACKEND_PORT regardless of who started it.
  function findPidOnPort() {
    try {
      if (process.platform === 'win32') {
        const out = execSync(
          `netstat -ano | findstr :${BACKEND_PORT} | findstr LISTENING`,
          { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'ignore'] },
        )
        for (const line of out.trim().split('\n')) {
          const pid = line.trim().split(/\s+/).pop()
          if (pid && pid !== '0') return Number(pid)
        }
      } else {
        const out = execSync(`lsof -ti:${BACKEND_PORT}`, {
          encoding: 'utf-8', stdio: ['pipe', 'pipe', 'ignore'],
        })
        const pid = parseInt(out.trim(), 10)
        if (pid > 0) return pid
      }
    } catch { /* nothing on port */ }
    return null
  }

  // Kill whatever process holds BACKEND_PORT (our child or an orphan).
  function killPortProcess() {
    // 1. Kill our own child if we have one
    if (backendProcess) {
      try {
        if (process.platform === 'win32') {
          execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' })
        } else {
          backendProcess.kill('SIGTERM')
        }
      } catch { /* already gone */ }
      backendProcess = null
    }

    // 2. Kill any remaining process on the port (orphan from previous run)
    const pid = findPidOnPort()
    if (pid) {
      try {
        if (process.platform === 'win32') {
          execSync(`taskkill /pid ${pid} /T /F`, { stdio: 'ignore' })
        } else {
          process.kill(pid, 'SIGTERM')
        }
        console.log(`  Killed orphan process on port ${BACKEND_PORT} (PID ${pid})`)
      } catch { /* already gone */ }
    }
  }

  // Quick HTTP health-check
  function checkBackend() {
    return new Promise((resolve) => {
      const req = http.get(
        `http://127.0.0.1:${BACKEND_PORT}/api/v1/health`,
        { timeout: 2000 },
        () => resolve(true),
      )
      req.on('error', () => resolve(false))
      req.on('timeout', () => { req.destroy(); resolve(false) })
    })
  }

  // Fetch real status payload from the running backend
  function fetchBackendStatus() {
    return new Promise((resolve) => {
      const req = http.get(
        `http://127.0.0.1:${BACKEND_PORT}/api/v1/system/status`,
        { timeout: 2000 },
        (res) => {
          let data = ''
          res.on('data', (chunk) => (data += chunk))
          res.on('end', () => {
            try { resolve(JSON.parse(data)) } catch { resolve(null) }
          })
        },
      )
      req.on('error', () => resolve(null))
      req.on('timeout', () => { req.destroy(); resolve(null) })
    })
  }

  // Spawn the Jarvis backend as a child process
  function startBackend() {
    if (backendProcess) return
    console.log('\n  Starting Jarvis backend...\n')
    backendProcess = spawn('python', ['-m', 'jarvis', '--no-cli'], {
      cwd: BACKEND_DIR,
      stdio: ['ignore', 'inherit', 'inherit'],
    })
    backendProcess.on('error', (err) => {
      console.error('  Failed to start Jarvis:', err.message)
      backendProcess = null
    })
    backendProcess.on('exit', (code) => {
      console.log(`\n  Jarvis backend exited (code ${code})\n`)
      backendProcess = null
    })
  }

  // Stop: kill child + any orphan on the port
  function stopBackend() {
    console.log('\n  Stopping Jarvis backend...\n')
    killPortProcess()
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

  function readBody(req) {
    return new Promise((resolve) => {
      let data = ''
      req.on('data', (chunk) => (data += chunk))
      req.on('end', () => resolve(data))
    })
  }

  return {
    name: 'jarvis-launcher',

    configureServer(server) {
      // On startup: adopt or kill any pre-existing backend
      const existingPid = findPidOnPort()
      if (existingPid) {
        console.log(`  Jarvis already running on port ${BACKEND_PORT} (PID ${existingPid}) — killing to take ownership...`)
        killPortProcess()
      }

      server.middlewares.use(async (req, res, next) => {
        // ── GET /api/v1/system/status ──────────────────────────────
        if (req.url === '/api/v1/system/status' && req.method === 'GET') {
          res.setHeader('Content-Type', 'application/json')
          const data = await fetchBackendStatus()
          if (data) {
            data.status = 'running'
            res.end(JSON.stringify(data))
          } else {
            res.end(JSON.stringify({
              status: 'stopped',
              timestamp: Date.now() / 1000,
            }))
          }
          return
        }

        // ── POST /api/v1/system/start ─────────────────────────────
        if (req.url === '/api/v1/system/start' && req.method === 'POST') {
          await readBody(req)
          res.setHeader('Content-Type', 'application/json')

          // Already running under our control?
          if (backendProcess && await checkBackend()) {
            res.end(JSON.stringify({ status: 'ok', message: 'Backend läuft bereits' }))
            return
          }

          // Kill any orphan, then spawn fresh
          killPortProcess()
          await sleep(500)
          startBackend()

          // Wait up to 30s for backend to respond
          let ready = false
          for (let i = 0; i < 30; i++) {
            await sleep(1000)
            if (await checkBackend()) { ready = true; break }
            if (!backendProcess) break
          }

          if (ready) {
            res.end(JSON.stringify({ status: 'ok', message: 'Jarvis gestartet' }))
          } else {
            res.statusCode = 504
            res.end(JSON.stringify({ error: 'Backend Timeout — Check Terminal für Fehler' }))
          }
          return
        }

        // ── POST /api/v1/system/stop ──────────────────────────────
        if (req.url === '/api/v1/system/stop' && req.method === 'POST') {
          await readBody(req)
          res.setHeader('Content-Type', 'application/json')
          stopBackend()
          await sleep(1000)
          // Verify it's actually gone
          const stillAlive = await checkBackend()
          if (stillAlive) {
            // Retry harder
            killPortProcess()
            await sleep(500)
          }
          res.end(JSON.stringify({ status: 'ok', message: 'Jarvis gestoppt' }))
          return
        }

        next()
      })

      // Clean up when Vite exits
      server.httpServer?.on('close', () => stopBackend())
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), jarvisLauncher()],
  server: {
    proxy: {
      '/api': { target: `http://localhost:${BACKEND_PORT}`, changeOrigin: true },
      '/ws':  { target: `ws://localhost:${BACKEND_PORT}`, ws: true },
    },
  },
})
