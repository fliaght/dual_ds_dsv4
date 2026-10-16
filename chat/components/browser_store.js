// The visitor's chats and settings, kept in THIS browser only (Streamlit custom component v2, no iframe).
//
//   IndexedDB "dsv4chat": store "convs" = whole conversations, store "meta" = the chat list (title, dates)
//   localStorage "dsv4chat:*": client_id (anonymous browser id), settings, active (the open chat)
//
// Python sends the state it wants on every run (browser.py, app.store_data); each part is applied once, in order:
//   deletes: [id]                    -> removed from both stores
//   conv: {conv, meta, legacy?}      -> written when its id@updated is new; legacy: drop the old localStorage copy
//   prefs: {settings, active}        -> localStorage
//   load: nonce                      -> trigger "loaded" {nonce, ok, error, client_id, settings, active, conv, index, legacy}
//   open: {nonce, id}                -> trigger "opened" {nonce, id, conv, error}  (after the writes above: an ack)
//   ask: nonce | null                -> the load/open Python is still waiting for
// Writes never trigger a rerun, so saving never interrupts a reply that is streaming.
// Streamlit can lose a component event: one sent while the page is finishing a run is dropped from the
// rerun it causes, and merged reruns keep only button clicks.  So while Python still asks for an answer it
// is sent again, after 0.25, 0.5, 1, 2, 3, 4, 5 and 6 s (Python ignores repeats).
const PREFIX = "dsv4chat:";
const G = (window.__dsv4store = window.__dsv4store ||
  { done: new Set(), chain: Promise.resolve(), db: null, error: null, prefs: null, asking: null, answer: null, retry: 0 });

const msg = (e) => String((e && (e.message || e.name)) || e);
function lsGet(k) { try { return window.localStorage.getItem(PREFIX + k); } catch (e) { return null; } }
function lsSet(k, v) { try { window.localStorage.setItem(PREFIX + k, v); } catch (e) { G.error = msg(e); } }
function lsDel(k) { try { window.localStorage.removeItem(PREFIX + k); } catch (e) { /* storage blocked */ } }
function lsJson(k) { try { return JSON.parse(lsGet(k) || "null"); } catch (e) { return null; } }

function clientId() {
  let id = lsGet("client_id");
  if (!id) {
    id = window.crypto && crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "")
      : Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
    lsSet("client_id", id);
  }
  return id;
}

function openDb() {
  if (!G.db) {
    G.db = new Promise((resolve, reject) => {
      if (!window.indexedDB) { reject(new Error("this browser has no IndexedDB")); return; }
      const req = indexedDB.open("dsv4chat", 1);
      req.onupgradeneeded = () => {
        for (const name of ["convs", "meta"]) {
          if (!req.result.objectStoreNames.contains(name)) req.result.createObjectStore(name, { keyPath: "id" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error("IndexedDB is blocked by another tab"));
    });
    G.db.catch(() => { G.db = null; });          // try again on the next operation
  }
  return G.db;
}

// One transaction; resolves with the request's result once it is committed.
async function run(stores, mode, body) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(stores, mode);
    const req = body(tx);
    tx.oncomplete = () => resolve(req ? req.result : undefined);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("transaction aborted"));
  });
}
const both = ["convs", "meta"];

const RETRY_MS = [250, 250, 500, 1000, 1000, 1000, 1000, 1000];

function answer(name, nonce, value, setTriggerValue) {
  const a = (G.answer = { name, nonce, value, tries: 0 });
  setTriggerValue(name, value);
  const again = () => {
    clearTimeout(G.retry);
    if (a.tries >= RETRY_MS.length) return;
    G.retry = setTimeout(() => {
      if (G.answer !== a || G.asking !== a.nonce) return;      // answered: Python stopped asking
      a.tries += 1;
      setTriggerValue(a.name, a.value);
      again();
    }, RETRY_MS[a.tries]);
  };
  again();
}

async function step(work) {            // a failed write is shown and retried on the next run; reads still answer
  try { await work(); } catch (e) { G.error = msg(e); }
}

async function apply(data, setTriggerValue) {
  for (const id of data.deletes || []) {
    if (G.done.has("del:" + id)) continue;
    await step(async () => {
      await run(both, "readwrite", (tx) => { tx.objectStore("convs").delete(id); tx.objectStore("meta").delete(id); });
      G.done.add("del:" + id);
    });
  }
  const c = data.conv;
  if (c && !G.done.has("put:" + c.meta.id + "@" + c.meta.updated)) {
    await step(async () => {
      await run(both, "readwrite", (tx) => { tx.objectStore("convs").put(c.conv); tx.objectStore("meta").put(c.meta); });
      G.done.add("put:" + c.meta.id + "@" + c.meta.updated);
      if (c.legacy) lsDel("conversation");
      G.error = null;
    });
  }
  const p = data.prefs;
  if (p && JSON.stringify(p) !== G.prefs) {
    if (p.settings) lsSet("settings", JSON.stringify(p.settings));
    if (p.active) lsSet("active", p.active); else lsDel("active");
    G.prefs = JSON.stringify(p);
  }
  if (data.load && !G.done.has(data.load)) {
    G.done.add(data.load);
    const out = { nonce: data.load, ok: true, error: null, client_id: clientId(), settings: lsJson("settings"),
                  active: lsGet("active"), conv: null, index: [], legacy: lsJson("conversation") };
    try {
      out.index = await run(["meta"], "readonly", (tx) => tx.objectStore("meta").getAll());
      if (out.active) out.conv = (await run(["convs"], "readonly", (tx) => tx.objectStore("convs").get(out.active))) || null;
    } catch (e) { out.ok = false; out.error = msg(e); }
    answer("loaded", data.load, out, setTriggerValue);
  }
  const o = data.open;
  if (o && !G.done.has(o.nonce)) {
    G.done.add(o.nonce);
    const out = { nonce: o.nonce, id: o.id || null, conv: null, error: null };
    try {
      if (o.id) out.conv = (await run(["convs"], "readonly", (tx) => tx.objectStore("convs").get(o.id))) || null;
    } catch (e) { out.error = msg(e); }
    answer("opened", o.nonce, out, setTriggerValue);
  }
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const note = parentElement.querySelector(".note");
  const paint = () => {
    if (!note) return;
    note.textContent = G.error ? "Not saved in this browser: " + G.error : (data && data.note) || "";
    note.classList.toggle("error", !!G.error);
  };
  paint();
  if (!data) return;
  if ("ask" in data) G.asking = data.ask;      // only the page's main store instance says what it waits for
  G.chain = G.chain
    .then(() => apply(data, setTriggerValue))
    .catch((e) => { G.error = msg(e); })
    .then(paint);
}
