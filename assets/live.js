// Blue Onyx Prism: live config for every page. Polls GET /v1/config (ETag / If-None-Match, every
// 2 s while the tab is visible and on focus), keeps the header's "pending changes - restart to
// apply" indicator current, and hands each new payload to the page scripts
// (PrismLive.onConfig). Also the form helpers the pages use to refresh fields the user has not
// edited while keeping (and marking) the ones they have.
(function () {
  "use strict";
  var PERIOD = 2000;
  var listeners = [], etag = null, last = null, busy = null;

  function emit(data, prev) {
    listeners.forEach(function (fn) { try { fn(data, prev); } catch (e) { if (window.console) console.error(e); } });
  }
  // Fetch /v1/config; resolves to the payload (the last one on 304 or error).
  function poll(force) {
    if (busy) return busy;
    var h = {"Accept": "application/json"};
    if (etag && !force) h["If-None-Match"] = etag;
    busy = fetch("/v1/config", {headers: h, cache: "no-store"}).then(function (r) {
      if (r.status === 304) return last;
      if (!r.ok) throw new Error("HTTP " + r.status);
      etag = r.headers.get("ETag");
      return r.json().then(function (j) { var prev = last; last = j; emit(j, prev); return j; });
    }).catch(function () { return last; }).then(function (j) { busy = null; return j; });
    return busy;
  }

  // Header: "N pending changes - restart to apply".
  function header(j) {
    var box = document.getElementById("pending-restart");
    if (!box) return;
    var n = j.pendingCount || 0, a = box.querySelector("a");
    box.hidden = n === 0;
    a.textContent = n + " pending change" + (n === 1 ? "" : "s") + " — restart to apply";
    a.title = ((j.pending && j.pending.all) || []).join("\n");
  }
  listeners.push(header);

  // Restart (existing POST /config/restart flow), then reload once the next generation answers.
  function restart(btn) {
    if (!confirm("Restart the server now? It reloads the config file and recompiles the enabled models.")) return;
    var gen = last ? last.generation : null, epoch = last ? last.epoch : null;
    if (btn) { btn.disabled = true; btn.textContent = "Restarting…"; }
    var done = function () {
      setTimeout(function again() {
        fetch("/v1/config", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (j) {
          if (j.generation !== gen || j.epoch !== epoch) location.reload(); else setTimeout(again, 700);
        }, function () { setTimeout(again, 700); });
      }, 700);
    };
    fetch("/config/restart", {method: "POST"}).then(done, done);
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-restart]");
    if (!b || b.type === "submit") return;
    e.preventDefault();
    restart(b);
  });

  // ---- form helpers -------------------------------------------------------------------------
  // A control differs from what the page last showed from the server (its default state).
  function edited(el) {
    if (el.type === "checkbox" || el.type === "radio") return el.checked !== el.defaultChecked;
    if (el.tagName === "SELECT") {
      for (var i = 0; i < el.options.length; i++) if (el.options[i].selected !== el.options[i].defaultSelected) return true;
      return false;
    }
    if (el.type === "number") {
      var a = el.value.trim(), b = el.defaultValue.trim();
      if (a === "" || b === "") return a !== b;
      return parseFloat(a) !== parseFloat(b);
    }
    return el.value !== el.defaultValue;
  }
  function current(el) {
    if (el.type === "checkbox") return el.checked;
    return el.value;
  }
  function baseline(el) {
    if (el.type === "checkbox") return el.defaultChecked;
    if (el.tagName === "SELECT") {
      for (var i = 0; i < el.options.length; i++) if (el.options[i].defaultSelected) return el.options[i].value;
      return el.options.length ? el.options[0].value : "";
    }
    return el.defaultValue;
  }
  function same(el, a, b) {
    if (el.type === "checkbox") return !!a === !!b;
    a = a === null || a === undefined ? "" : String(a); b = b === null || b === undefined ? "" : String(b);
    if (el.type === "number" && a.trim() !== "" && b.trim() !== "") return parseFloat(a) === parseFloat(b);
    return a === b;
  }
  // Set value and default (the new "shown by the server" state).
  function set(el, v) {
    if (el.type === "checkbox") { el.checked = !!v; el.defaultChecked = !!v; return; }
    v = v === null || v === undefined ? "" : String(v);
    if (el.tagName === "SELECT") {
      var found = false;
      for (var i = 0; i < el.options.length; i++) {
        var o = el.options[i], on = o.value === v; o.selected = on; o.defaultSelected = on; if (on) found = true;
      }
      if (!found) { var n = new Option(v + " — not a known option", v, true, true); el.add(n); }
      return;
    }
    el.value = v; el.defaultValue = v;
  }
  function flash(el) {
    if (!el) return;
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }
  function note(el, text) {
    var holder = el.closest("label, td") || el.parentNode, n = holder.querySelector(".live-note");
    if (!text) { if (n) n.remove(); el.classList.remove("edited-conflict"); return; }
    if (!n) { n = document.createElement("span"); n.className = "live-note"; holder.appendChild(n); }
    n.textContent = text; el.classList.add("edited-conflict");
  }
  // Refresh one control to the server value `v`: unedited -> take it (flash when it changed);
  // edited -> keep the user's value; mark it when the server changed it too. Returns true when
  // the control now shows the server value (the caller then adopts the new base).
  function refresh(el, v, describe) {
    if (!edited(el)) {
      var changed = !same(el, current(el), v);
      set(el, v); note(el, null); el.classList.remove("edited");
      if (changed) flash(el);
      return true;
    }
    if (same(el, current(el), v)) { set(el, v); note(el, null); el.classList.remove("edited"); return true; }
    if (!same(el, baseline(el), v)) note(el, "Changed elsewhere to " + (describe ? describe(v) : String(v)) + "; saving asks which to keep.");
    else note(el, null);
    return false;
  }
  function markEdits(form) {
    form.addEventListener("input", function (e) { if (e.target.name) e.target.classList.toggle("edited", edited(e.target)); });
    form.addEventListener("change", function (e) { if (e.target.name) e.target.classList.toggle("edited", edited(e.target)); });
  }

  window.PrismLive = {
    onConfig: function (fn) { listeners.push(fn); if (last) fn(last, null); },
    poll: poll,
    restart: restart,
    last: function () { return last; },
    ui: {edited: edited, current: current, baseline: baseline, same: same, set: set, flash: flash, note: note, refresh: refresh, markEdits: markEdits}
  };
  poll();
  setInterval(function () { if (!document.hidden) poll(); }, PERIOD);
  window.addEventListener("focus", function () { poll(); });
  document.addEventListener("visibilitychange", function () { if (!document.hidden) poll(); });
})();
