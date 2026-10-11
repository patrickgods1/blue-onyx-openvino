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

  // ---- reloads that keep the scroll position --------------------------------------------------
  // A reload the page triggers itself (a restart finished, a model finished loading) comes
  // back where the user was: the position is kept in sessionStorage for this path and restored
  // once, after load (and again while asynchronously rendered content is still growing).
  var SCROLL_KEY = "prism:scroll:" + location.pathname;
  function reloadKeep(url) {
    try { sessionStorage.setItem(SCROLL_KEY, JSON.stringify({x: window.scrollX, y: window.scrollY, t: Date.now()})); } catch (e) { /* private mode */ }
    if (url) location.replace(url); else location.reload();
  }
  (function restoreScroll() {
    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem(SCROLL_KEY) || "null"); sessionStorage.removeItem(SCROLL_KEY); } catch (e) { saved = null; }
    if (!saved || Date.now() - saved.t > 10 * 60 * 1000) return;
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
    var tries = 0, user = false;
    ["wheel", "touchstart", "keydown", "mousedown"].forEach(function (t) { window.addEventListener(t, function () { user = true; }, {once: true, passive: true}); });
    function go() {
      if (user) return;
      window.scrollTo(saved.x, saved.y);
      if (Math.abs(window.scrollY - saved.y) > 2 && ++tries < 20) setTimeout(go, 150);
    }
    if (document.readyState === "complete") go(); else window.addEventListener("load", go);
  })();

  // ---- restart ----------------------------------------------------------------------------------
  // A banner pinned to the top of the window while the server restarts (it does not move the
  // page).
  function restarting(text) {
    var b = document.getElementById("restart-banner");
    if (!b) { b = document.createElement("div"); b.id = "restart-banner"; b.className = "restart-banner"; b.setAttribute("role", "status"); b.setAttribute("aria-live", "polite"); document.body.appendChild(b); }
    b.textContent = text || "Restarting\u2026 the page reloads (at the same place) when the server answers again.";
    return b;
  }
  // Wait for the next registry generation, then reload in place.
  function waitRestart(gen, epoch) {
    setTimeout(function again() {
      fetch("/v1/config", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (j) {
        if (j.generation !== gen || j.epoch !== epoch) reloadKeep(); else setTimeout(again, 700);
      }, function () { setTimeout(again, 700); });
    }, 700);
  }
  function afterRestart() { waitRestart(last ? last.generation : null, last ? last.epoch : null); }
  // Restart (existing POST /config/restart flow), then reload once the next generation answers.
  function restart(btn) {
    if (!confirm("Restart the server now? It reloads the config file and recompiles the enabled models.")) return;
    var gen = last ? last.generation : null, epoch = last ? last.epoch : null;
    if (btn) { btn.disabled = true; btn.textContent = "Restarting…"; }
    restarting();
    var done = function () { waitRestart(gen, epoch); };
    fetch("/config/restart", {method: "POST", headers: {"Accept": "application/json"}}).then(done, done);
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

  // ---- view state across re-renders ------------------------------------------------------------
  // Pages that poll re-render tables; without care every poll resets what the user did with
  // them. `keep(root, sig, render)` skips the render when the section's data signature `sig` is
  // unchanged, defers it while the user is busy in it (selecting text, pressing the mouse, e.g.
  // dragging a scrollbar, pointing at a chart), and otherwise carries over, by stable key, the
  // scroll offsets of every `.scroll` wrapper (and any element with `data-keep`), the open state
  // of every <details>, each table's sort column and direction, and the focused control.
  // Keys: `data-keep`, else the id, else the element's kind and position in the section.
  // `patch(box, items)` does the same per child (keyed cards): only changed children are
  // rebuilt, in place.
  var pointerIn = null;
  document.addEventListener("pointerdown", function (e) { pointerIn = e.target; }, true);
  ["pointerup", "pointercancel", "dragend"].forEach(function (t) { document.addEventListener(t, function () { pointerIn = null; }, true); });
  var KEPT = ".scroll, details, table, [data-keep]";
  function keyed(root) {
    var out = {}, seen = {};
    root.querySelectorAll(KEPT).forEach(function (e) {
      var k = e.dataset.keep || (e.id ? "#" + e.id : null);
      if (!k) { var kind = e.tagName.toLowerCase() + (e.classList.contains("scroll") ? ".scroll" : ""); seen[kind] = (seen[kind] || 0) + 1; k = kind + ":" + seen[kind]; }
      if (!(k in out)) out[k] = e;
    });
    return out;
  }
  function busyIn(root) {
    if (pointerIn && root.contains(pointerIn)) return true;
    var s = window.getSelection && window.getSelection();
    if (s && !s.isCollapsed && s.rangeCount && root.contains(s.getRangeAt(0).commonAncestorContainer)) return true;
    return !!root.querySelector("svg:hover, .keep-hover:hover");
  }
  function capture(root) {
    var st = {els: {}, focus: null};
    var map = keyed(root);
    Object.keys(map).forEach(function (k) {
      var e = map[k], v = {};
      if (e.scrollLeft || e.scrollTop) { v.left = e.scrollLeft; v.top = e.scrollTop; }
      if (e.tagName === "DETAILS") v.open = e.open;
      if (e.tagName === "TABLE") {
        var hs = e.tHead ? e.tHead.rows[0].cells : [];
        for (var i = 0; i < hs.length; i++) if (hs[i].dataset.dir) v.sort = [i, hs[i].dataset.dir];
      }
      st.els[k] = v;
    });
    var a = document.activeElement;
    if (a && a !== document.body && root.contains(a)) {
      var all = root.querySelectorAll(a.tagName);
      st.focus = {tag: a.tagName, name: a.name || null, text: a.textContent, index: Array.prototype.indexOf.call(all, a)};
    }
    return st;
  }
  // Sort a table's body by column `col` (numeric when the cells carry data-v).
  function sortTable(table, col, asc) {
    var tb = table.tBodies[0]; if (!tb) return;
    var rows = Array.prototype.slice.call(tb.rows);
    table.querySelectorAll("th").forEach(function (h) { delete h.dataset.dir; });
    var th = table.tHead && table.tHead.rows[0].cells[col]; if (th) th.dataset.dir = asc ? "asc" : "desc";
    rows.sort(function (a, b) {
      var x = a.cells[col], y = b.cells[col]; if (!x || !y) return 0;
      var vx = x.dataset.v, vy = y.dataset.v;
      if (vx !== undefined && vy !== undefined) { vx = parseFloat(vx); vy = parseFloat(vy); if (isNaN(vx)) vx = Infinity; if (isNaN(vy)) vy = Infinity; return asc ? vx - vy : vy - vx; }
      return asc ? x.textContent.localeCompare(y.textContent) : y.textContent.localeCompare(x.textContent);
    });
    rows.forEach(function (r) { tb.appendChild(r); });
  }
  function restore(root, st) {
    var map = keyed(root);
    Object.keys(st.els).forEach(function (k) {
      var e = map[k], v = st.els[k]; if (!e) return;
      if (v.sort) sortTable(e, v.sort[0], v.sort[1] === "asc");
      if (v.open !== undefined && e.open !== v.open) e.open = v.open;
    });
    // Scroll after every <details> is open again (closed ones have no layout).
    Object.keys(st.els).forEach(function (k) {
      var e = map[k], v = st.els[k]; if (!e || v.left === undefined) return;
      e.scrollLeft = v.left; e.scrollTop = v.top;
    });
    var f = st.focus;
    if (f && (!document.activeElement || document.activeElement === document.body)) {
      var cands = Array.prototype.filter.call(root.querySelectorAll(f.tag), function (e) { return (e.name || null) === f.name && e.textContent === f.text; });
      var pick = cands.length ? cands[0] : root.querySelectorAll(f.tag)[f.index];
      if (pick && pick.focus) try { pick.focus({preventScroll: true}); } catch (e) { pick.focus(); }
    }
  }
  // Re-render `root` with `render()` keeping its view state. `sig` (any string; undefined =
  // always render) skips unchanged data. Returns true when it rendered; false when skipped or
  // deferred (a deferred render is retried with the next call, its signature not taken).
  function keep(root, sig, render) {
    if (!root) return false;
    if (sig !== undefined && root._keepSig === sig) return false;
    if (root._keepSig !== undefined && busyIn(root)) return false;
    var st = capture(root);
    render();
    restore(root, st);
    root._keepSig = sig;
    return true;
  }
  // Keyed children of `box`: items [{key, sig, build}] in order; `build()` returns the new
  // element. Unchanged children (same sig) are left alone, changed ones replaced in place with
  // their view state, missing ones added, the rest removed.
  function patch(box, items) {
    var old = {};
    Array.prototype.forEach.call(box.children, function (c) { if (c.dataset.patchKey !== undefined) old[c.dataset.patchKey] = c; });
    var prev = null;
    items.forEach(function (it) {
      var cur = old[it.key]; delete old[it.key];
      if (!cur || (cur._keepSig !== it.sig && !busyIn(cur))) {
        var n = it.build(); n.dataset.patchKey = it.key; n._keepSig = it.sig;
        if (cur) { var st = capture(cur); cur.replaceWith(n); restore(n, st); }
        cur = n;
      }
      var want = prev ? prev.nextSibling : box.firstChild;
      if (want !== cur) box.insertBefore(cur, want);
      prev = cur;
    });
    Object.keys(old).forEach(function (k) { old[k].remove(); });
  }

  // ---- forms without leaving the page ----------------------------------------------------------
  // `<form data-ajax>`: submitted with fetch (Accept: application/json) instead of navigating,
  // so the page keeps its scroll position; the clicked button is disabled while the request
  // runs and the answer is shown next to it. The page then updates the section in place: it
  // gets a "prism:action" event on the form ({ok, json, button}). Without JS the form posts
  // normally (its action carries the section's #fragment). `data-confirm` on a button asks
  // first.
  function inlineMsg(form, by) {
    var at = (by && by.closest(".actions, td, .inline-row")) || form, m = at.querySelector(".form-msg, .action-msg");
    if (!m) { m = document.createElement("span"); m.className = "action-msg"; m.setAttribute("aria-live", "polite"); at.appendChild(m); }
    return m;
  }
  function say(m, cls, text) { m.className = (m.classList.contains("form-msg") ? "form-msg " : "action-msg ") + cls; m.textContent = text; }
  function submitAjax(form, by) {
    var data = new FormData(form);
    if (by && by.name) data.append(by.name, by.value);
    var url = (by && by.getAttribute("formaction")) || form.getAttribute("action") || location.pathname;
    url = url.split("#")[0];
    var m = inlineMsg(form, by), label = by ? by.textContent : null;
    if (by) by.disabled = true;
    say(m, "muted", "working…");
    var fire = function (ok, j) { form.dispatchEvent(new CustomEvent("prism:action", {bubbles: true, detail: {ok: ok, json: j, button: by, message: m}})); };
    return fetch(url, {method: "POST", body: new URLSearchParams(data), headers: {"Accept": "application/json"}})
      .then(function (r) { return r.json().then(function (j) { j._status = r.status; return j; }); })
      .then(function (j) {
        if (by) { by.disabled = false; by.textContent = label; }
        say(m, j.success ? "ok-text" : "error-text", j.message || (j.success ? "done" : "failed"));
        poll(true);
        fire(!!j.success, j);
      })
      .catch(function (e) { if (by) by.disabled = false; say(m, "error-text", "failed: " + e); fire(false, null); });
  }
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (e.defaultPrevented || !form.matches || !form.matches("form[data-ajax]")) return;
    var by = e.submitter;
    e.preventDefault();
    if (by && by.hasAttribute("data-restart")) { restart(by); return; }
    submitAjax(form, by);
  });

  window.PrismLive = {
    onConfig: function (fn) { listeners.push(fn); if (last) fn(last, null); },
    poll: poll,
    restart: restart,
    restarting: restarting,
    afterRestart: afterRestart,
    reloadKeep: reloadKeep,
    submit: submitAjax,
    last: function () { return last; },
    keep: keep,
    patch: patch,
    sortTable: sortTable,
    view: {capture: capture, restore: restore, busy: busyIn},
    ui: {edited: edited, current: current, baseline: baseline, same: same, set: set, flash: flash, note: note, refresh: refresh, markEdits: markEdits}
  };
  poll();
  setInterval(function () { if (!document.hidden) poll(); }, PERIOD);
  window.addEventListener("focus", function () { poll(); });
  document.addEventListener("visibilitychange", function () { if (!document.hidden) poll(); });
})();
