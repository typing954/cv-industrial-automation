import { useEffect, useMemo, useState } from "react";
import { LiveImage, post, usePoll } from "./api.jsx";

// CLI flag for each form field (used to show the equivalent terminal command).
const FLAGS = { model: "--model", device: "--device", conf: "--conf", threads: "--threads", imgsz: "--imgsz",
  threshold: "--threshold", hold: "--hold", clear: "--clear", cooldown: "--cooldown", target_fps: "--target-fps",
  tracker: "--tracker", clip_pre: "--clip-pre", clip_post: "--clip-post", merge_gap: "--merge-gap", webhook: "--webhook" };

const num = (v) => (v === "" || v == null ? null : Number(v));

function cliCommand(f, defaults) {
  const a = ["python -m peoplecount run", f.source ? JSON.stringify(f.source).replace(/^"|"$/g, "") : "<source>"];
  for (const [k, flag] of Object.entries(FLAGS)) {
    const v = f[k];
    if (v === "" || v == null || v === defaults[k]) continue;
    a.push(`${flag} ${v}`);
  }
  if (f.realtime) a.push("--realtime");
  if (f.no_clips) a.push("--no-clips");
  if (!f.save_video) a.push("--no-save");
  return a.join(" ") + (f.loop ? "    # + loop (dashboard only)" : "");
}

function Field({ label, hint, children }) {
  return <label className="field"><span className="lab">{label}{hint && <small title={hint}> ⓘ</small>}</span>{children}</label>;
}

function Check({ label, hint, checked, onChange }) {
  return <label className="check" title={hint}><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />{label}</label>;
}

function Section({ title, open, children }) {
  return <details className="sect" open={open}><summary>{title}</summary><div className="sect-body">{children}</div></details>;
}

export default function RunView({ status }) {
  const { data: opts } = usePoll("/api/options", 60000);
  const [f, setF] = useState(null);                 // form values
  const [custom, setCustom] = useState(false);      // custom source (webcam index / RTSP) instead of a video file
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {                                 // initialise once options arrive
    if (!opts || f || !status) return;                // wait for options AND status so a running run's parameters are shown
    const p = status.params;
    setF({ ...opts.defaults, source: opts.videos[0] ?? "", ...(p ?? {}) });
    if (p && !opts.videos.includes(p.source)) setCustom(true);
  }, [opts, f, status]);

  const set = (k) => (e) => setF((p) => ({ ...p, [k]: e?.target ? e.target.value : e }));
  const cmd = useMemo(() => (f && opts ? cliCommand(f, opts.defaults) : ""), [f, opts]);

  if (!opts || !f) return <p className="empty">Connecting…</p>;

  const state = status?.state ?? "idle";
  const running = state === "loading" || state === "running";
  const model = opts.models.find((m) => m.name === f.model);

  const start = async () => {
    setErr(""); setBusy(true);
    try {
      await post("/api/run", { ...f, threads: num(f.threads), imgsz: num(f.imgsz), target_fps: num(f.target_fps),
        conf: Number(f.conf), threshold: Number(f.threshold), hold: Number(f.hold), clear: Number(f.clear),
        cooldown: Number(f.cooldown), clip_pre: Number(f.clip_pre), clip_post: Number(f.clip_post),
        merge_gap: Number(f.merge_gap), webhook: f.webhook || null });
    } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  const stop = async () => { setBusy(true); try { await post("/api/stop"); } catch (e) { setErr(e.message); } setBusy(false); };

  const N = (k, props) => <input type="number" value={f[k] ?? ""} onChange={set(k)} {...props} />;

  return (
    <div className="run">
      <aside className="form">
        <Section title="Source" open>
          <Field label="Camera / video">
            <select value={custom ? "__custom" : f.source} onChange={(e) => {
              if (e.target.value === "__custom") { setCustom(true); setF((p) => ({ ...p, source: "" })); }
              else { setCustom(false); setF((p) => ({ ...p, source: e.target.value })); }
            }}>
              {opts.videos.map((v) => <option key={v} value={v}>{v}</option>)}
              <option value="__custom">Webcam / RTSP…</option>
            </select>
          </Field>
          {custom && <Field label="Webcam index or stream URL" hint="e.g. 0 or rtsp://user:pass@host/stream">
            <input type="text" value={f.source} onChange={set("source")} placeholder="0  or  rtsp://…" /></Field>}
          <Check label="Pace at native FPS (--realtime)" hint="Files play in real time and frames are dropped when the model is too slow"
            checked={f.realtime} onChange={set("realtime")} />
          <Check label="Loop video when it ends" hint="Dashboard only; the CLI stops at the end of the file"
            checked={f.loop} onChange={set("loop")} />
        </Section>

        <Section title="Model & hardware" open>
          <Field label="Model (--model)" hint={model?.note}>
            <select value={f.model} onChange={set("model")}>
              {opts.models.map((m) => <option key={m.name} value={m.name}>{m.name}{m.note ? ` – ${m.note}` : ""}</option>)}
            </select>
          </Field>
          <Field label="Device (--device)">
            <select value={f.device} onChange={set("device")}>{opts.devices.map((d) => <option key={d}>{d}</option>)}</select>
          </Field>
          <div className="row2">
            <Field label="Confidence (--conf)">{N("conf", { min: 0.05, max: 0.95, step: 0.05 })}</Field>
            <Field label="Image size (--imgsz)">{N("imgsz", { min: 160, step: 32, placeholder: String(model?.imgsz ?? 640) })}</Field>
          </div>
          <div className="row2">
            <Field label="CPU threads (--threads)">{N("threads", { min: 1, placeholder: "auto" })}</Field>
            <Field label="Detect FPS (--target-fps)" hint="Detections per second; blank = every frame">{N("target_fps", { min: 1, placeholder: "every frame" })}</Field>
          </div>
          <Field label="Tracker (--tracker)">
            <select value={f.tracker} onChange={set("tracker")}>{opts.trackers.map((t) => <option key={t}>{t}</option>)}</select>
          </Field>
        </Section>

        <Section title="Alert rule" open>
          <Field label="Alert when people exceed (--threshold)">{N("threshold", { min: 0 })}</Field>
          <div className="row2">
            <Field label="Hold s (--hold)" hint="Seconds over the limit before ALERT">{N("hold", { min: 0, step: 0.5 })}</Field>
            <Field label="Clear s (--clear)" hint="Seconds under the limit before CLEARED">{N("clear", { min: 0, step: 0.5 })}</Field>
          </div>
          <Field label="Cooldown s (--cooldown)" hint="Minimum seconds between alerts; 0 = record every episode">{N("cooldown", { min: 0 })}</Field>
          <Field label="Webhook URL (--webhook)"><input type="text" value={f.webhook ?? ""} onChange={set("webhook")} placeholder="optional" /></Field>
        </Section>

        <Section title="Clips & output">
          <Check label="Cut evidence clips (off = --no-clips)" checked={!f.no_clips} onChange={(v) => setF((p) => ({ ...p, no_clips: !v }))} />
          <div className="row3">
            <Field label="Pre s">{N("clip_pre", { min: 0 })}</Field>
            <Field label="Post s">{N("clip_post", { min: 0 })}</Field>
            <Field label="Merge gap s">{N("merge_gap", { min: 0 })}</Field>
          </div>
          <Check label="Save full annotated video (off = --no-save)" checked={f.save_video} onChange={set("save_video")} />
        </Section>

        <div className="actions">
          {running
            ? <button className="btn stop" onClick={stop} disabled={busy}>Stop</button>
            : <button className="btn go" onClick={start} disabled={busy || !f.source}>Start</button>}
        </div>
        {err && <p className="formerr">{err}</p>}
        <pre className="cmd" title="Equivalent terminal command">{cmd}</pre>
      </aside>

      <section className="player">
        <div className={"stage" + (status?.alerting ? " alerting" : "")}>
          {state === "idle" && <p className="empty">Choose a camera and parameters, then press Start.</p>}
          {state === "loading" && <p className="empty">Loading model…</p>}
          {state !== "idle" && <LiveImage alt="live" />}
          {state === "error" && <div className="overlay-msg bad">{status.error}</div>}
        </div>
        <div className="statbar">
          <span className={`state ${state}`}>{state}</span>
          <span>People <b>{status?.count ?? 0}</b> / {status?.threshold ?? f.threshold}</span>
          <span>Max <b>{status?.max_people ?? 0}</b></span>
          <span>{status?.fps ?? 0} FPS</span>
          <span>Dropped {status?.dropped ?? 0}</span>
          <span>Time {status?.elapsed_s ?? 0}s</span>
          {status?.alerting && <span className="over">OVER CAPACITY</span>}
        </div>
      </section>
    </div>
  );
}
