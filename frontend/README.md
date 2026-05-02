# AOB Frontend (Next.js 15)

This is the judge-optimized UI for the Autonomous Oncology Board demo. It connects to the FastAPI backend and presents the live multi-agent pipeline, debate transcript, VRAM dashboard, and final clinical report.

## Prerequisites
- Node.js 18+
- A running backend at `http://localhost:8000` (or your AMD host)

## Environment
Create a `.env.local` file in this folder:

```
NEXT_PUBLIC_API_URL=http://<AMD_HOST>:8000
```

## Run locally

```
npm install
npm run dev
```

Open `http://localhost:3000`.

## Build

```
npm run build
npm run start
```

## Routes
- `/` landing page with demo launch cards and live VRAM widget
- `/analyze/[jobId]` live SSE pipeline view
- `/report/[jobId]` final clinical report with debate and PFS chart

## Deployment (Vercel)
- Root directory: `frontend`
- Build command: `npm run build`
- Env var: `NEXT_PUBLIC_API_URL=http://<AMD_HOST>:8000`
