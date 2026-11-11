# Dashboard (React + FastAPI)

Two tabs: **Live run** (pick one camera + run parameters, Start/Stop, annotated video with alert overlay and live stats) and
**Alert history** (per-camera list of every limit violation: start time, duration, peak, snapshot, evidence clip). Alert history is saved in `samples/webapp/` and survives restarts.

## Run
```bash
cd people_counter
pip install -r requirements.txt                      # adds fastapi + uvicorn
(cd webapp/frontend && npm install && npm run build)
python -m webapp.server                              # http://localhost:8000
```
Development with hot reload: run the server as above, then `cd webapp/frontend && npm run dev` (http://localhost:5173).

## Parameters
The Live tab mirrors `python -m peoplecount run`, one camera at a time, and shows the equivalent terminal command under the form.
Source, model, device, conf, imgsz, threads, target-fps, tracker, realtime, threshold, hold, clear, cooldown, webhook,
no-clips, clip-pre/post, merge-gap, save video. Web-only: "Loop video". Files must be in `videos/`; webcam index or `rtsp://` also work.

## Notes
* Outputs (annotated video, alerts.jsonl, episodes.json, counts.csv, clips) go to `samples/<video>/` exactly like the CLI.
* The browser polls one JPEG at a time instead of MJPEG, so it keeps working across run restarts.
* API: `GET /api/options`, `POST /api/run`, `POST /api/stop`, `GET /api/status`, `GET /api/frame`, `GET /api/alerts?source=`, `GET /api/files/<path>`.
