// Copy / Edit / Regenerate under one reply (Streamlit custom component v2, no iframe).
// data: {text, later}: the reply's markdown, and how many messages follow it.
// Copy stays in the browser; Edit and Regenerate send the trigger "action" ("edit" | "regen") to Python.
// Regenerating an earlier reply removes the messages after it, so that takes a second click.
const timers = new WeakMap();

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall back below */ }
  }
  // plain http (the LAN address) has no Clipboard API: copy through a hidden textarea
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:fixed;top:0;left:0;opacity:0;pointer-events:none";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
  ta.remove();
  return ok;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const d = data || {};
  const later = d.later || 0;
  const flash = (button, label, ms) => {
    clearTimeout(timers.get(button));
    button.textContent = label;
    timers.set(button, setTimeout(() => {
      button.textContent = button.dataset.label;
      button.className = "";
      timers.delete(button);
    }, ms));
  };
  for (const b of parentElement.querySelectorAll("button")) {
    b.dataset.label = b.dataset.label || b.textContent;
    if (!timers.get(b)) b.textContent = b.dataset.label;
    const action = b.dataset.a;
    if (action === "copy") {
      b.disabled = !d.text;
      b.onclick = async () => {
        const ok = await copyText(d.text || "");
        b.className = ok ? "done" : "error";
        flash(b, ok ? "Copied" : "Copy failed", 1600);
      };
    } else if (action === "edit") {
      b.onclick = () => setTriggerValue("action", "edit");
    } else if (action === "regen") {
      b.onclick = () => {
        if (later > 0 && b.className !== "armed") {
          b.className = "armed";
          flash(b, `Regenerate and remove the ${later} later message${later > 1 ? "s" : ""}?`, 5000);
          return;
        }
        clearTimeout(timers.get(b));
        setTriggerValue("action", "regen");
      };
    }
  }
}
