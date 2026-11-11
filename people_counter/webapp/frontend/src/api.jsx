import { useEffect, useRef, useState } from "react";

// Polls a JSON endpoint; keeps the last good data on failure and exposes `error` ("backend offline").
export function usePoll(url, ms) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (alive) { setData(j); setError(null); }
      } catch (e) {
        if (alive) setError(e.message);
      }
    };
    tick();
    const id = setInterval(tick, ms);
    return () => { alive = false; clearInterval(id); };
  }, [url, ms]);
  return { data, error };
}

export async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = j.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join("; ") : `HTTP ${r.status}`);
  }
  return j;
}

// Latest annotated frame, fetched in a loop (~10 fps). Short requests keep working across run restarts.
export function LiveImage({ alt, ...rest }) {
  const ref = useRef(null);
  useEffect(() => {
    let alive = true, prev = null;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    (async () => {
      while (alive) {
        try {
          const r = await fetch("/api/frame", { cache: "no-store" });
          if (r.ok && alive && ref.current) {
            const url = URL.createObjectURL(await r.blob());
            ref.current.src = url;
            if (prev) URL.revokeObjectURL(prev);
            prev = url;
          }
        } catch { /* backend down: retry */ }
        await sleep(100);
      }
      if (prev) URL.revokeObjectURL(prev);
    })();
    return () => { alive = false; };
  }, []);
  return <img ref={ref} alt={alt} {...rest} />;
}

export const fileUrl = (p) => `/api/files/${p}`;
