import { useState } from "react";
import { usePoll } from "./api.jsx";
import RunView from "./RunView.jsx";
import AlertHistory from "./AlertHistory.jsx";

export default function App() {
  const [tab, setTabState] = useState(location.hash === "#history" ? "history" : "live");
  const setTab = (t) => { setTabState(t); history.replaceState(null, "", t === "history" ? "#history" : "#"); };
  const { data: status, error } = usePoll("/api/status", 1000);
  const alerting = status?.alerting;

  return (
    <div className="app">
      <header className="topbar">
        <h1>People Counter</h1>
        <nav>
          <button className={tab === "live" ? "tab on" : "tab"} onClick={() => setTab("live")}>
            Live run {alerting && <span className="badge">ALERT</span>}
          </button>
          <button className={tab === "history" ? "tab on" : "tab"} onClick={() => setTab("history")}>Alert history</button>
        </nav>
        <div className="meta">
          {error ? <span className="offline">Backend offline</span> : status?.device && <span>{status.params?.model} · {status.device}</span>}
        </div>
      </header>
      {/* RunView stays mounted so the form keeps its values when you look at the history */}
      <main style={{ display: tab === "live" ? "block" : "none" }}><RunView status={status} /></main>
      {tab === "history" && <main><AlertHistory /></main>}
    </div>
  );
}
