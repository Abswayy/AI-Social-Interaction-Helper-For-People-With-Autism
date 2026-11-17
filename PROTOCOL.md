# Device ↔ server protocol

One WebSocket per device: `ws(s)://<server>:8000/ws`. This is the only thing a
new piece of hardware has to implement. The laptop page (`web/client.js`) is a
complete reference client.

## Device → server

**Binary messages**: the first byte is the type.

| Byte 0 | Payload | Rate |
|---|---|---|
| `0x01` | Audio: mono **16 kHz signed 16-bit little-endian PCM**, any chunk size (20–100 ms is ideal) | continuous |
| `0x02` | Video frame: one **JPEG**, ~640×480 is plenty | 2–4 fps |

Turn **off** automatic gain control on the mic if you can (automatic speaker
detection uses loudness). Noise suppression or beam-forming from the glasses' DSP is welcome.

**Text (JSON) messages**

```json
{"type": "settings", "settings": {"num_replies": 2, "social": true, "emotion": true,
                                  "turn_hints": true, "max_words": 10,
                                  "speaker_mode": "auto",
                                  "voice_profile": {"me": {"db": -18.5, "f0": 128.0}, "other": null}}}
{"type": "ptt", "down": true}        // hold-to-talk pressed (touchpad / button / Space)
{"type": "ptt", "down": false}       // released
{"type": "fix_speaker", "index": 4}  // swap Me/Them on transcript line 4
{"type": "calibrate_start"}          // wearer then talks normally for ~5 s
{"type": "calibrate_stop"}
{"type": "end_conversation"}         // optional; conversations also end automatically
{"type": "rename", "person_id": 3, "name": "Ravi"}
```

`speaker_mode`: `"auto"` (voice pitch + loudness; `ptt` still overrides) or
`"manual"` (only `ptt` marks the wearer's speech). Settings can be sent again at
any time with only the keys that changed.

On glasses, map hold-to-talk to whatever exists: a touchpad long-press, a
temple button, or a ring/clicker. Send `ptt` down on press and up on release.

## Server → device (JSON)

| type | fields | what to show |
|---|---|---|
| `status` | `ready`, `message` | – |
| `face` | `visible`, `bbox` [x,y,w,h] 0–1, `person_id` | nothing on glasses (debug only) |
| `person` | `id`, `name`, `named`, `new`, `relationship`, `designation`, `conversations` | **name and who they are** (relationship, role if different). No facts, follow-ups, or reminder line |
| `turn` | `state`: `listening` / `speaking` / `your_turn` / `wait` / `info`, `text` | show `text` for `your_turn`, `wait` (~4 s) and `info` (~8 s) |
| `transcript` | `index`, `speaker` (`me`/`them`), `text`, `conf`, `source` (`auto`/`button`/`calibration`) | optional |
| `transcript_fixed` | `index`, `speaker` | update that line |
| `cues` | `replies` [..], `social`, `emotion`, optional `kind` (`icebreaker`). `pending: true` = a suggestion is being written. `cleared: true` = drop that placeholder. `dismissed: true` = take the lines down. A reply that would invent a fact is replaced with a generic reaction, so `replies` is empty only when suggestions are turned off | **main captions**. One reply after 5 s of silence or a few umms, on a question or a finished statement. At 8 s, if the talk is still stuck, that reply stays and one ice breaker is added. Also **one starter** when a person with memory is first recognised this conversation |
| `calibrated` | `profile` (or `null` if too little speech) | store it on the device, send it back in `settings` next time |
| `voice_profile` | `profile` | the model learned something; store it like `calibrated` |
| `conversation_saved` | `person_id`, `conversation_id`, `name`, `title`, `summary` | clear captions |
| `error` | `where`, `message` | small warning |

## Typical glasses display (monocular, ~3 lines)

```
Ravi · Neighbour · Teacher
1. How did the exam go?
```

## REST API (People page, phone companion apps)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/people` | everyone, with profile |
| GET | `/api/people/{id}` | one person + conversation list |
| GET | `/api/people/{id}/photo` | face photo (JPEG) |
| PATCH | `/api/people/{id}` | `{name, notes, profile: {relationship, designation, recall_hint, key_facts[], interests[], communication_tips[], follow_ups[]}}` |
| DELETE | `/api/people/{id}` | remove person and face data (conversations kept, unassigned) |
| POST | `/api/people/{keep}/merge/{drop}` | combine duplicates |
| POST | `/api/register` | `{name, relationship, photos: [base64 JPEG]}` |
| GET | `/api/conversations/unassigned` | conversations with no recognised face |
| GET | `/api/conversations/{id}` | full record incl. transcript |
| PATCH | `/api/conversations/{id}` | `{person_id}` (or `null`) |
| DELETE | `/api/conversations/{id}` | delete one conversation |
| GET | `/api/export` | everything as JSON (no face data) |
| GET | `/health` | models ready, database type, people count |
