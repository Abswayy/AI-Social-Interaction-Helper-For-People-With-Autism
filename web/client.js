// Laptop client: streams webcam + mic to the server and renders captions on the
// camera, in the spot a glasses lens would use.
// A glasses client only has to reproduce: (1) send binary [1]+PCM16 16kHz and
// [2]+JPEG frames, (2) send JSON settings/control (incl. ptt), (3) draw the JSON events.
(() => {
  const $ = (id) => document.getElementById(id);
  const MSG_AUDIO = 1, MSG_FRAME = 2;
  const FRAME_INTERVAL_MS = 333;      // ~3 fps is plenty for recognition
  let ws, stream, frameTimer, turnTimer, pttDown = false;

  // ---------- settings (persisted in this browser) ----------
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem("scg_" + k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem("scg_" + k, JSON.stringify(v)); } catch {} },
  };
  // older versions stored only a loudness number
  if (!store.get("voice_profile") && store.get("self_voice_db") != null)
    store.set("voice_profile", { me: { db: store.get("self_voice_db"), f0: null } });

  const KEYS = ["num_replies", "social", "emotion", "turn_hints", "max_words", "fade_idle"];
  const WAKE_MS = 8000;
  let mode = store.get("speaker_mode") || "auto";

  function readSettings() {
    return {
      num_replies: +$("num_replies").value, social: $("social").checked, emotion: $("emotion").checked,
      turn_hints: $("turn_hints").checked, max_words: +$("max_words").value,
      speaker_mode: mode, voice_profile: store.get("voice_profile"),
    };
  }
  for (const k of KEYS) {
    const v = store.get(k), el = $(k);
    if (v !== null) el.type === "checkbox" ? (el.checked = v) : (el.value = v);
    el.addEventListener("change", () => {
      store.set(k, el.type === "checkbox" ? el.checked : el.value);
      if (k === "fade_idle") {
        if (el.checked && (view.replies.length || view.pending || view.social)) wake();
        else applyFade();
      } else send({ type: "settings", settings: readSettings() });
    });
  }

  // ---------- speaker mode ----------
  function setMode(m) {
    mode = m; store.set("speaker_mode", m);
    $("modeAuto").setAttribute("aria-pressed", m === "auto");
    $("modeManual").setAttribute("aria-pressed", m === "manual");
    $("modeHelp").textContent = m === "auto"
      ? "The app recognises your voice by its pitch and loudness. Holding the button below still overrides it."
      : "Everything is treated as the other person unless you hold the button while you talk.";
    send({ type: "settings", settings: { speaker_mode: m } });
    showVoiceState();
  }
  $("modeAuto").onclick = () => setMode("auto");
  $("modeManual").onclick = () => setMode("manual");

  function showVoiceState() {
    const p = store.get("voice_profile"), ready = p && p.me;
    const el = $("voiceState");
    if (mode === "manual") { el.textContent = "Manual: hold to talk"; el.className = "pill ok"; }
    else if (ready) { el.textContent = "Automatic: voice set up"; el.className = "pill ok"; }
    else { el.textContent = "Automatic: set up your voice"; el.className = "pill bad"; }
    $("calibInfo").textContent = ready
      ? `Your voice: ${p.me.f0 ? Math.round(p.me.f0) + " Hz pitch, " : ""}${p.me.db.toFixed(0)} dB.` +
        (p.other ? " Also learned the other person's voice." : "")
      : "Speak normally for 5 seconds. Only needed once.";
  }

  // ---------- hold to talk ----------
  function ptt(down) {
    if (down === pttDown || $("ptt").disabled) return;
    pttDown = down;
    $("ptt").classList.toggle("down", down);
    $("ptt").textContent = down ? "Listening to you…" : "Hold while you speak (or hold Space)";
    send({ type: "ptt", down });
  }
  const pb = $("ptt");
  pb.addEventListener("pointerdown", (e) => { e.preventDefault(); pb.setPointerCapture(e.pointerId); ptt(true); });
  pb.addEventListener("pointerup", () => ptt(false));
  pb.addEventListener("pointercancel", () => ptt(false));
  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !typing()) { e.preventDefault(); ptt(true); }
    if (e.key === "Escape") document.body.classList.remove("hudonly");
  });
  document.addEventListener("keyup", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); ptt(false); } });
  window.addEventListener("blur", () => ptt(false));

  // ---------- connection ----------
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
  function sendBin(kind, buf) {
    if (!ws || ws.readyState !== 1 || ws.bufferedAmount > 512 * 1024) return; // backpressure
    const u = new Uint8Array(buf.byteLength + 1); u[0] = kind; u.set(new Uint8Array(buf), 1);
    ws.send(u);
  }
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { setConn(true); send({ type: "settings", settings: readSettings() }); };
    ws.onclose = () => { setConn(false); if (stream) setTimeout(connect, 1500); };
    ws.onmessage = (e) => handle(JSON.parse(e.data));
  }
  function setConn(on) { $("conn").textContent = on ? "Connected" : "Offline"; $("conn").className = "pill " + (on ? "ok" : "bad"); }

  // ---------- capture ----------
  async function start() {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      // Gain control off: automatic speaker detection uses loudness.
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
    });
    $("video").srcObject = stream; $("camPlaceholder").hidden = true;
    await startAudio(stream);
    startFrames();
    connect();
    $("start").disabled = true;
    for (const id of ["end", "calib", "ptt"]) $(id).disabled = false;
    $("rec").hidden = false;
  }

  async function startAudio(stream) {
    const audioCtx = new AudioContext();
    const worklet = `
      class Down extends AudioWorkletProcessor {
        constructor() { super(); this.ratio = sampleRate / 16000; this.acc = []; this.pos = 0; }
        process(inputs) {
          const ch = inputs[0][0]; if (!ch) return true;
          while (this.pos < ch.length) {               // linear-interpolation resample to 16 kHz
            const i = Math.floor(this.pos), f = this.pos - i;
            const a = ch[i], b = i + 1 < ch.length ? ch[i + 1] : a;
            this.acc.push(a + (b - a) * f); this.pos += this.ratio;
          }
          this.pos -= ch.length;
          if (this.acc.length >= 640) {                // 40 ms per message
            const out = new Int16Array(this.acc.length);
            for (let j = 0; j < out.length; j++) out[j] = Math.max(-1, Math.min(1, this.acc[j])) * 32767;
            this.port.postMessage(out.buffer, [out.buffer]); this.acc = [];
          }
          return true;
        }
      }
      registerProcessor("down", Down);`;
    await audioCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([worklet], { type: "text/javascript" })));
    const node = new AudioWorkletNode(audioCtx, "down");
    node.port.onmessage = (e) => sendBin(MSG_AUDIO, e.data);
    audioCtx.createMediaStreamSource(stream).connect(node);
  }

  function startFrames() {
    const canvas = document.createElement("canvas"), video = $("video");
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext("2d");
    frameTimer = setInterval(() => {
      if (!video.videoWidth) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => b && b.arrayBuffer().then((buf) => sendBin(MSG_FRAME, buf)), "image/jpeg", 0.75);
    }, FRAME_INTERVAL_MS);
  }

  // ---------- glasses lens ----------
  // The card that stays up is the name and who they are. A starter from memory
  // may appear once underneath, in the same place as later reply lines.
  const view = { personId: null, person: null, turn: null, emotion: "", social: "", replies: [], pending: false, note: "" };
  let noteTimer = 0;
  let spot = { place: "bottom", side: "left" };
  let spotHold = 0;
  const SPOTS = [
    { place: "bottom", side: "left" },
    { place: "bottom", side: "right" },
    { place: "top", side: "left" },
    { place: "top", side: "right" },
  ];

  function memoryFrom(m) {
    const rel = (m.relationship || "").trim();
    const role = (m.designation || "").trim();
    const same = rel && role && rel.toLowerCase() === role.toLowerCase();
    return {
      name: (m.name || "Someone").trim(),
      isNew: !!m.new,
      relationship: [rel, same ? "" : role].filter(Boolean).join(" · "),
      context: "",
      detail: "",
      prompt: "",
      promptLabel: "",
    };
  }

  function setNote(text, ms) {
    view.note = text || "";
    clearTimeout(noteTimer);
    if (text && ms) noteTimer = setTimeout(() => { view.note = ""; renderLens(); }, ms);
    renderLens();
  }

  // The caption box stays dim until a stall or a suggestion, then rests again.
  // "Your turn" and "let them finish" are never dimmed.
  let awakeUntil = 0;
  let wakeTimer = 0;
  function paintFade() {
    const plate = $("lensPlate");
    if (!plate) return;
    const fade = !!($("fade_idle") && $("fade_idle").checked);
    const bright = !fade || Date.now() < awakeUntil;
    // Important so a cached stylesheet animation cannot pin the card at full strength.
    plate.style.setProperty("opacity", bright ? "1" : "0.18", "important");
    $("hud").classList.toggle("resting", fade && !bright);
  }
  function wake() {
    if (!$("fade_idle").checked) {
      clearTimeout(wakeTimer);
      awakeUntil = 0;
      paintFade();
      return;
    }
    awakeUntil = Date.now() + WAKE_MS;
    clearTimeout(wakeTimer);
    wakeTimer = setTimeout(paintFade, WAKE_MS);
    paintFade();
  }
  function applyFade() {
    if (!$("fade_idle").checked) {
      clearTimeout(wakeTimer);
      awakeUntil = 0;
    }
    paintFade();
  }

  function renderLens() {
    const p = view.person;
    $("hudPerson").textContent = p ? p.name : "";
    $("hudRel").textContent = p ? p.relationship : "";
    $("hudRel").hidden = !p || !p.relationship;
    $("hudNew").hidden = !p || !p.isNew;
    $("hudEmotion").textContent = view.emotion || "";
    $("hudEmotion").hidden = !view.emotion;
    const showWho = !!((p && p.name) || view.emotion);
    $("whoRow").hidden = !showWho;

    $("hudHint").textContent = p ? p.context : "";
    $("hudHint").hidden = !p || !p.context;
    $("hudDetail").textContent = p ? p.detail : "";
    $("hudDetail").hidden = !p || !p.detail;

    const ask = $("hudAsk");
    ask.replaceChildren();
    if (p && p.prompt) {
      if (p.promptLabel) {
        const k = document.createElement("span");
        k.className = "k";
        k.textContent = p.promptLabel;
        ask.append(k);
      }
      ask.append(document.createTextNode(p.prompt));
      ask.hidden = false;
    } else ask.hidden = true;

    const turn = $("hudTurn");
    turn.textContent = view.turn ? view.turn.text : "";
    turn.className = "lens-turn" + (view.turn && view.turn.state !== "your_turn" ? " " + view.turn.state : "");
    turn.hidden = !view.turn;

    const socialOn = !!view.social;
    $("hudSocial").textContent = socialOn ? view.social : "";
    $("hudSocial").hidden = !socialOn;

    const replies = $("hudReplies");
    replies.replaceChildren();
    if (view.pending) {
      const row = document.createElement("div");
      row.className = "pending";
      row.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
      replies.append(row);
    } else {
      view.replies.forEach((r, i) => {
        const row = document.createElement("div");
        row.className = "reply";
        if (view.replies.length > 1) {
          const n = document.createElement("span");
          n.className = "n";
          n.textContent = String(i + 1);
          row.append(n);
        }
        row.append(document.createTextNode(r));
        replies.append(row);
      });
    }
    replies.hidden = !view.pending && !view.replies.length;

    $("hudNote").textContent = view.note || "";
    $("hudNote").hidden = !view.note;

    const compact = view.replies.length > 0;
    $("hud").classList.toggle("has-replies", compact);
    const visible = showWho || (p && (p.context || p.detail || p.prompt)) || view.turn || socialOn || view.pending || view.replies.length || view.note;
    $("hud").hidden = !visible;
    applyFade();
  }

  function clearHud() {
    view.personId = null; view.person = null; view.turn = null; view.emotion = ""; view.social = "";
    view.replies = []; view.pending = false; view.note = "";
    clearTimeout(turnTimer); clearTimeout(noteTimer);
    applySpot({ place: "bottom", side: "left" });
    renderLens();
  }

  function turnLabel(state, text) {
    if (state === "your_turn") return "your turn";
    if (state === "wait") return "let them finish";
    return text || "";
  }
  function showTurn(m) {
    clearTimeout(turnTimer);
    view.turn = null;
    if (["your_turn", "wait", "info"].includes(m.state)) {
      view.turn = { state: m.state, text: turnLabel(m.state, m.text) };
      turnTimer = setTimeout(() => { view.turn = null; renderLens(); }, m.state === "info" ? 8000 : 4000);
    }
    renderLens();
  }

  function region(s) {
    return {
      x: s.side === "left" ? 0.03 : 0.40,
      y: s.place === "bottom" ? 0.60 : 0.03,
      w: 0.57,
      h: 0.34,
    };
  }
  function overlaps(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }
  function applySpot(s) {
    spot = s; spotHold = 0;
    const lens = $("hud");
    lens.dataset.place = s.place;
    lens.dataset.side = s.side;
  }
  // The preview is mirrored. Keep the caption off the face, and don't chase small movements.
  function placeLens(b) {
    const face = {
      x: (1 - b[0] - b[2]) - 0.03,
      y: b[1] - 0.02,
      w: b[2] + 0.06,
      h: b[3] + 0.04,
    };
    const open = SPOTS.filter((s) => !overlaps(region(s), face));
    const preferred = open[0] || spot;
    if (overlaps(region(spot), face)) {
      spotHold = 0;
      if (preferred.place !== spot.place || preferred.side !== spot.side) applySpot(preferred);
      return;
    }
    if (preferred.place === spot.place && preferred.side === spot.side) { spotHold = 0; return; }
    if (++spotHold >= 4) applySpot(preferred);
  }
  function drawFace(b) {
    const box = $("facebox");
    if (!b) { box.hidden = true; return; }
    const x = (1 - b[0] - b[2]) * 100;
    Object.assign(box.style, { left: x + "%", top: b[1] * 100 + "%", width: b[2] * 100 + "%", height: b[3] * 100 + "%" });
    box.hidden = false;
    placeLens(b);
  }

  // ---------- events from server ----------
  function handle(m) {
    switch (m.type) {
      case "status":
        if (m.ready) { if (view.note) { view.note = ""; renderLens(); } }
        else setNote(m.message || "");
        break;
      case "face": drawFace(m.bbox); break;
      case "person": {
        const same = view.personId != null && view.personId === m.id;
        view.personId = m.id;
        view.person = memoryFrom(m);
        // A new face replaces the card. The same person (a rename) keeps a starter already up.
        if (!same) {
          view.replies = []; view.pending = false; view.emotion = ""; view.social = "";
        }
        renderLens();
        break;
      }
      case "turn": showTurn(m); break;
      case "transcript":
        addLine(m);
        if (m.hesitation) wake();
        break;
      case "transcript_fixed": fixLine(m); break;
      case "cues":
        // The wearer answered, or the subject changed. Mood can stay.
        // Leaving the lines up does not keep the box bright.
        if (m.dismissed) {
          view.pending = false; view.replies = []; view.social = "";
          renderLens();
          break;
        }
        if (m.cleared) { view.pending = false; renderLens(); break; }
        if (m.pending) { view.pending = true; wake(); renderLens(); break; }
        view.pending = false;
        view.emotion = (m.emotion || "").trim();
        const social = (m.social || "").trim();
        const replies = Array.isArray(m.replies) ? m.replies.filter(Boolean) : [];
        const changed = social !== view.social || replies.length !== view.replies.length
          || replies.some((r, i) => r !== view.replies[i]);
        view.social = social;
        view.replies = replies;
        if (changed && (replies.length || social)) wake();
        renderLens();
        break;
      case "calibrated":
        $("calib").textContent = "Set up my voice"; $("calib").disabled = false;
        if (m.profile) store.set("voice_profile", m.profile);
        else $("calibInfo").textContent = "Didn't hear enough speech. Try again and keep talking for 5 seconds.";
        if (m.profile) showVoiceState();
        break;
      case "voice_profile": store.set("voice_profile", m.profile); showVoiceState(); break;
      case "conversation_saved": addSaved(m); clearHud(); break;
      case "error": console.warn(m); setNote((m.message || "Something went wrong").slice(0, 90), 7000); break;
    }
  }

  const SRC = { button: "you held the button", auto: "detected by voice", corrected: "you corrected this", calibration: "voice setup" };
  function addLine(m) {
    $("transcriptEmpty")?.remove();
    const b = document.createElement("button");
    b.className = `line ${m.speaker}` + (m.conf < 0.45 ? " low" : "");
    b.dataset.index = m.index;
    b.title = "Click to switch between Me and Them";
    b.innerHTML = `<span class="who"></span><span><span class="text"></span><span class="src"></span></span>`;
    b.querySelector(".text").textContent = m.text;
    setWho(b, m.speaker, m.source);
    b.onclick = () => send({ type: "fix_speaker", index: +b.dataset.index });
    $("transcript").appendChild(b); b.scrollIntoView({ block: "nearest" });
  }
  function setWho(b, speaker, source) {
    b.classList.toggle("me", speaker === "me"); b.classList.toggle("them", speaker === "them");
    b.querySelector(".who").textContent = speaker === "me" ? "Me" : "Them";
    b.querySelector(".src").textContent = SRC[source] || "";
  }
  function fixLine(m) {
    const b = $("transcript").querySelector(`[data-index="${m.index}"]`);
    if (b) setWho(b, m.speaker, "corrected");
  }
  function addSaved(m) {
    const d = document.createElement("div"); d.className = "saved-item";
    const title = document.createElement("strong"); title.textContent = `Saved: ${m.title || "Conversation"}`;
    const p = document.createElement("p"); p.className = "muted small";
    p.textContent = (m.name ? m.name + ". " : "") + (m.summary || "");
    const a = document.createElement("a"); a.href = m.person_id ? `/people#${m.person_id}` : "/people#unassigned";
    a.textContent = "Open in People"; a.className = "small";
    d.append(title, p, a);
    $("saved").prepend(d);
    $("transcript").innerHTML = '<p class="muted small" id="transcriptEmpty">Nothing yet.</p>';
  }

  // ---------- buttons ----------
  $("start").onclick = () => start().catch((e) => alert("Camera or microphone could not start: " + e.message));
  $("end").onclick = () => send({ type: "end_conversation" });
  $("calib").onclick = () => {
    send({ type: "calibrate_start" }); $("calib").disabled = true;
    let left = 5;
    $("calib").textContent = `Keep talking… ${left}`;
    const t = setInterval(() => {
      left -= 1;
      if (left > 0) $("calib").textContent = `Keep talking… ${left}`;
      else { clearInterval(t); $("calib").textContent = "Checking…"; send({ type: "calibrate_stop" }); }
    }, 1000);
  };
  $("hudonly").onclick = () => document.body.classList.toggle("hudonly");

  setMode(mode);
  applyFade();
  setInterval(paintFade, 1000);
})();
