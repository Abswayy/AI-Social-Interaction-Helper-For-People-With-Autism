// People page: everyone the glasses know, their attributes, and every conversation.
(() => {
  const $ = (id) => document.getElementById(id);
  let people = [], unassigned = [], current = null, editing = false;

  const api = async (path, opts = {}) => {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
    return body;
  };
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k.startsWith("on")) n[k] = v; else if (v != null) n.setAttribute(k, v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(c));
    return n;
  };
  const date = (t) => t ? new Date(t * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";
  const time = (t) => t ? new Date(t * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : "";
  const ago = (t) => {
    if (!t) return "";
    const d = (Date.now() / 1000 - t) / 86400;
    return d < 1 ? "today" : d < 2 ? "yesterday" : d < 30 ? `${Math.floor(d)} days ago` : date(t);
  };
  const dur = (s) => s < 60 ? `${s} sec` : `${Math.round(s / 60)} min`;
  const toast = (msg) => {
    const t = el("div", { class: "toast", role: "status" }, msg);
    document.body.append(t); setTimeout(() => t.remove(), 2200);
  };
  const avatar = (p, big) => p.has_photo
    ? el("img", { class: "avatar" + (big ? " big" : ""), src: `/api/people/${p.id}/photo`, alt: "" })
    : el("div", { class: "avatar" + (big ? " big" : ""), "aria-hidden": "true" }, (p.name || "?").trim()[0].toUpperCase());

  // ---------------- list ----------------
  async function loadList() {
    [people, unassigned] = await Promise.all([api("/api/people"), api("/api/conversations/unassigned")]);
    renderList();
  }
  function matches(p, q) {
    if (!q) return true;
    const pr = p.profile || {};
    return [p.name, p.notes, p.latest_summary, pr.relationship, pr.designation, pr.recall_hint, ...(pr.key_facts || []), ...(pr.interests || [])]
      .join(" ").toLowerCase().includes(q);
  }
  function renderList() {
    const q = $("search").value.trim().toLowerCase();
    const list = $("list"); list.innerHTML = "";
    const shown = people.filter((p) => matches(p, q));
    for (const p of shown) {
      const sub = (p.profile || {}).recall_hint || (p.named ? `${p.conversation_count} conversation${p.conversation_count === 1 ? "" : "s"}` : "Added automatically, no name yet");
      list.append(el("button", { class: "person-row", "aria-current": String(current === String(p.id)), onclick: () => (location.hash = p.id) },
        avatar(p), el("span", {}, el("div", { class: "name" }, p.name), el("div", { class: "sub" }, sub))));
    }
    if (unassigned.length && !q) {
      list.append(el("button", { class: "person-row", "aria-current": String(current === "unassigned"), onclick: () => (location.hash = "unassigned") },
        el("div", { class: "avatar", "aria-hidden": "true" }, "?"),
        el("span", {}, el("div", { class: "name" }, "No face seen"),
          el("div", { class: "sub" }, `${unassigned.length} conversation${unassigned.length === 1 ? "" : "s"} to assign`))));
    }
    if (!shown.length && !(unassigned.length && !q))
      list.append(el("p", { class: "muted small" }, q ? "Nobody matches that search." :
        "No one yet. People appear here when the glasses see a face, or when someone registers."));
  }
  $("search").oninput = renderList;

  // ---------------- detail ----------------
  async function route() {
    current = location.hash.slice(1) || null; editing = false;
    renderList();
    if (!current) return;
    if (current === "unassigned") return renderUnassigned();
    try { renderPerson(await api(`/api/people/${current}`)); }
    catch { $("detail").replaceChildren(el("div", { class: "panel empty" }, "This person no longer exists.")); }
  }
  window.addEventListener("hashchange", route);

  function factPanel(title, items, emptyText) {
    const list = (items || []).filter(Boolean);
    return el("div", { class: "panel" }, el("h3", {}, title),
      list.length ? el("ul", {}, list.map((x) => el("li", {}, x))) : el("p", { class: "muted small" }, emptyText));
  }

  function renderPerson(p) {
    const pr = p.profile || {};
    const others = people.filter((x) => x.id !== p.id);
    const head = el("div", { class: "panel detail-head" },
      avatar(p, true),
      el("div", {},
        el("h1", {}, p.name, " ", !p.named ? el("span", { class: "tag auto" }, "Added automatically") : null),
        el("p", { class: "meta" }, [pr.relationship, pr.designation, `First met ${date(p.first_met)}`,
          `${p.conversation_count} conversation${p.conversation_count === 1 ? "" : "s"}`,
          p.last_seen ? `last seen ${ago(p.last_seen)}` : null].filter(Boolean).join(", "))),
      el("div", { class: "row" }, el("button", { class: "primary", onclick: () => renderEdit(p) }, "Edit details")));

    const recall = pr.recall_hint ? el("p", { class: "recall" }, pr.recall_hint) : null;
    const facts = el("div", { class: "facts" },
      factPanel("About them", pr.key_facts, "Nothing learned yet. Facts appear after a conversation."),
      factPanel("Interests", pr.interests, "None noted yet."),
      factPanel("How to talk with them", pr.communication_tips, "No tips yet."),
      factPanel("Ask about next time", pr.follow_ups, "Nothing pending."),
      el("div", { class: "panel" }, el("h3", {}, "Your notes"),
        p.notes ? el("p", {}, p.notes) : el("p", { class: "muted small" }, "Add your own notes with Edit details. The AI reads them but never changes them.")));

    const timeline = el("div", { class: "panel" }, el("h2", {}, "Conversations"),
      p.conversations.length
        ? el("div", { class: "timeline" }, p.conversations.map((c) => convItem(c, p.id)))
        : el("p", { class: "muted small", style: "margin-top:8px" }, "No saved conversations with this person yet."));

    const merge = el("select", { "aria-label": "Person to merge" }, el("option", { value: "" }, "Choose a person…"),
      others.map((o) => el("option", { value: o.id }, o.name)));
    const manage = el("details", { class: "panel" }, el("summary", {}, el("strong", {}, "Fix duplicates or remove")),
      el("div", { class: "stack", style: "margin-top:12px" },
        el("p", { class: "small muted" }, "If the glasses registered the same person twice, merge the other record into this one. Faces and conversations are combined."),
        el("div", { class: "row" }, merge, el("button", { onclick: async () => {
          if (!merge.value) return;
          const other = others.find((o) => String(o.id) === merge.value);
          if (!confirm(`Merge ${other.name} into ${p.name}? This can't be undone.`)) return;
          await api(`/api/people/${p.id}/merge/${merge.value}`, { method: "POST" });
          toast("Merged"); await loadList(); route();
        } }, "Merge into this person")),
        el("div", {}, el("button", { class: "danger", onclick: async () => {
          if (!confirm(`Delete ${p.name}? Their face data is removed. Conversations are kept under "No face seen".`)) return;
          await api(`/api/people/${p.id}`, { method: "DELETE" });
          toast("Deleted"); location.hash = ""; await loadList();
          $("detail").replaceChildren(el("div", { class: "panel empty" }, "Deleted."));
        } }, "Delete this person"))));

    $("detail").replaceChildren(head, recall, facts, timeline, manage);
  }

  function convItem(c, personId) {
    const body = el("div", { class: "script" });
    const det = el("details", {}, el("summary", {}, "Details and transcript"),
      el("div", { class: "cols" },
        c.key_points.length ? el("div", {}, el("strong", {}, "Key points"), el("ul", {}, c.key_points.map((x) => el("li", {}, x)))) : null,
        c.follow_ups.length ? el("div", {}, el("strong", {}, "Promises and plans"), el("ul", {}, c.follow_ups.map((x) => el("li", {}, x)))) : null),
      body);
    det.addEventListener("toggle", async () => {
      if (!det.open || body.childElementCount) return;
      const full = await api(`/api/conversations/${c.id}`);
      body.replaceChildren(...(full.transcript.length
        ? full.transcript.map((t) => el("div", { class: t.speaker }, el("b", {}, t.speaker === "me" ? "Me: " : "Them: "), t.text))
        : [el("p", { class: "muted small" }, "No transcript stored.")]));
    });
    const move = el("select", { "aria-label": "Move conversation to" },
      el("option", { value: "" }, "Move to…"),
      people.filter((x) => x.id !== personId).map((o) => el("option", { value: o.id }, o.name)),
      personId != null ? el("option", { value: "none" }, "No face seen") : null);
    move.onchange = async () => {
      if (!move.value) return;
      await api(`/api/conversations/${c.id}`, { method: "PATCH", body: JSON.stringify({ person_id: move.value === "none" ? null : +move.value }) });
      toast("Conversation moved"); await loadList(); route();
    };
    return el("article", { class: "conv" },
      el("div", { class: "when" }, el("strong", {}, date(c.ended_at)), time(c.started_at), el("br"), dur(c.duration_s)),
      el("div", {},
        el("h3", {}, c.title || "Conversation"),
        c.mood ? el("p", { class: "small muted" }, `They seemed ${c.mood}`) : null,
        c.topics.length ? el("div", { class: "topics" }, c.topics.map((t) => el("span", { class: "tag" }, t))) : null,
        el("p", {}, c.summary || el("span", { class: "muted" }, "No summary (the AI was unavailable). The transcript is saved.")),
        det,
        el("div", { class: "actions" }, move, el("button", { class: "link", onclick: async () => {
          if (!confirm("Delete this conversation?")) return;
          await api(`/api/conversations/${c.id}`, { method: "DELETE" }); toast("Conversation deleted"); await loadList(); route();
        } }, "Delete"))));
  }

  function renderEdit(p) {
    const pr = p.profile || {};
    const field = (label, input, hint) => el("label", { class: "field" }, label, hint ? el("span", { class: "hint" }, hint) : null, input);
    const text = (v) => el("input", { type: "text", value: v || "" });
    const lines = (v) => { const t = el("textarea", { rows: 4 }); t.value = (v || []).join("\n"); return t; };
    const f = {
      name: text(p.name), relationship: text(pr.relationship), designation: text(pr.designation),
      recall_hint: text(pr.recall_hint),
      notes: (() => { const t = el("textarea", { rows: 4 }); t.value = p.notes || ""; return t; })(),
      key_facts: lines(pr.key_facts), interests: lines(pr.interests),
      communication_tips: lines(pr.communication_tips), follow_ups: lines(pr.follow_ups),
    };
    const split = (t) => t.value.split("\n").map((s) => s.trim()).filter(Boolean);
    const save = async () => {
      await api(`/api/people/${p.id}`, { method: "PATCH", body: JSON.stringify({
        name: f.name.value, notes: f.notes.value,
        profile: { relationship: f.relationship.value, designation: f.designation.value,
          recall_hint: f.recall_hint.value,
          key_facts: split(f.key_facts), interests: split(f.interests),
          communication_tips: split(f.communication_tips), follow_ups: split(f.follow_ups) } }) });
      toast("Changes saved"); await loadList(); route();
    };
    $("detail").replaceChildren(el("div", { class: "panel stack" },
      el("div", { class: "row" }, avatar(p, true), el("h1", {}, `Edit ${p.name}`)),
      el("div", { class: "edit-grid" },
        field("Name", f.name), field("How you know them", f.relationship, "e.g. neighbour, classmate"),
        field("Role or title", f.designation, "Shown with their name on the glasses, e.g. teacher"),
        field("Reminder line", f.recall_hint, "Kept on this page. Not shown on the glasses")),
      field("Your notes", f.notes, "Private notes. The AI uses them but never overwrites them."),
      el("div", { class: "edit-grid" },
        field("About them", f.key_facts, "One fact per line"), field("Interests", f.interests, "One per line"),
        field("How to talk with them", f.communication_tips, "One tip per line"),
        field("Ask about next time", f.follow_ups, "One per line")),
      el("div", { class: "row" }, el("button", { class: "primary", onclick: save }, "Save changes"),
        el("button", { onclick: route }, "Cancel"))));
  }

  function renderUnassigned() {
    $("detail").replaceChildren(
      el("div", { class: "panel" }, el("h1", {}, "No face seen"),
        el("p", { class: "muted", style: "margin-top:6px" }, "These conversations happened while no face was recognised. Move each one to the right person.")),
      el("div", { class: "panel" }, unassigned.length
        ? el("div", { class: "timeline" }, unassigned.map((c) => convItem(c, null)))
        : el("p", { class: "muted" }, "Nothing left to assign.")));
  }

  api("/health").then((h) => {
    $("dbState").textContent = h.database === "postgres" ? "Cloud database" : "Saved on this computer";
    $("dbState").className = "pill ok";
  }).catch(() => {});
  loadList().then(route);
})();
