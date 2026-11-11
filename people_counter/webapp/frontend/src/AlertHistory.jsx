import { useState } from "react";
import { usePoll, fileUrl } from "./api.jsx";

const fmtTime = (s) => new Date(s * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "medium" });
const fmtDur = (s) => {
  if (s == null) return "–";
  s = Math.round(s);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
};
const LABEL = { active: "Active", cleared: "Cleared", interrupted: "Interrupted" };

export default function AlertHistory() {
  const [sel, setSel] = useState("all");
  const [zoom, setZoom] = useState(null);
  const { data, error } = usePoll(`/api/alerts${sel === "all" ? "" : `?source=${encodeURIComponent(sel)}`}`, 3000);
  const rows = data?.alerts ?? [];
  const sources = Object.entries(data?.sources ?? {});
  const total = sources.reduce((n, [, c]) => n + c, 0);

  return (
    <div className="history">
      <aside className="feedlist">
        <button className={sel === "all" ? "item on" : "item"} onClick={() => setSel("all")}>
          <span>All cameras</span><span className="n">{total}</span>
        </button>
        {sources.map(([s, c]) => (
          <button key={s} className={sel === s ? "item on" : "item"} onClick={() => setSel(s)}>
            <span>{s}</span><span className="n has">{c}</span>
          </button>
        ))}
      </aside>
      <section className="rows">
        {error && <p className="offline">Backend offline</p>}
        {!error && data && rows.length === 0 && <p className="empty">No alerts recorded yet. Start a run on the Live tab.</p>}
        {rows.length > 0 && (
          <table>
            <thead>
              <tr><th>Camera</th><th>Status</th><th>Limit exceeded at</th><th>Duration</th><th>Peak / limit</th><th>Model</th><th>Snapshot</th><th>Clip</th></tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.id} className={a.status === "active" ? "active-row" : ""}>
                  <td>{a.source}</td>
                  <td><span className={`pill ${a.status}`}>{LABEL[a.status] ?? a.status}</span></td>
                  <td>{fmtTime(a.started_at)}</td>
                  <td>{a.status === "active" ? "ongoing" : fmtDur(a.duration_s)}</td>
                  <td><b>{a.peak_count}</b> / {a.threshold}</td>
                  <td>{a.model}</td>
                  <td>{a.snapshot ? <img className="thumb" src={fileUrl(a.snapshot)} alt="" onClick={() => setZoom({ img: a.snapshot })} /> : "–"}</td>
                  <td>{a.clip ? <button className="link" onClick={() => setZoom({ clip: a.clip })}>▶ Play</button> : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          {zoom.img ? <img src={fileUrl(zoom.img)} alt="" /> : <video src={fileUrl(zoom.clip)} controls autoPlay onClick={(e) => e.stopPropagation()} />}
        </div>
      )}
    </div>
  );
}
