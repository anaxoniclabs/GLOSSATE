# Issues — GLOSSATE v0.1.0

Source PRD: `docs/glossate-prd.md`

| #   | Title                                                                 | Status | Type | Blocked by              |
| --- | --------------------------------------------------------------------- | ------ | ---- | ----------------------- |
| 001 | [Project scaffold with working info stub](001-project-scaffold.md)    | todo   | AFK  | —                       |
| 002 | [Cue model + segmenter](002-cue-model-segmenter.md)                   | todo   | AFK  | 001                     |
| 003 | [Serializer (SRT / VTT / JSON)](003-serializer.md)                    | todo   | AFK  | 001                     |
| 004 | [Output path construction + collision](004-output-paths.md)           | todo   | AFK  | 001                     |
| 005 | [Device detection + preflight checks](005-device-preflight.md)        | todo   | AFK  | 001                     |
| 006 | [UI theme + ASCII art + info panel](006-ui-theme-ascii-components.md) | todo   | AFK  | 001                     |
| 007 | [ffmpeg audio extraction](007-ffmpeg-audio-extraction.md)             | todo   | AFK  | 005                     |
| 008 | [Model download + cache check](008-model-download.md)                 | todo   | AFK  | 001                     |
| 009 | [GlossateProgress (live stage display)](009-glossate-progress.md)     | todo   | AFK  | 006                     |
| 010 | [Transcriber (mlx-whisper wrapper)](010-transcriber.md)               | todo   | AFK  | 005, 002                |
| 011 | [Translator (mlx-lm, XML batch, retry)](011-translator.md)            | todo   | AFK  | 002                     |
| 012 | [Public API + Session + exception translation](012-public-api-session.md) | todo | AFK | 003, 004, 007, 008, 010, 011 |
| 013 | [CLI (argparse, lang detection, info)](013-cli.md)                    | todo   | AFK  | 006, 009, 012           |

## Dependency graph

```
001 (scaffold)
├── 002 (segmenter) ──────────────────────────┐
├── 003 (serializer) ────────────────────┐    │
├── 004 (output paths) ─────────────────┤    │
├── 005 (device/preflight) ──┬── 007 ───┤    │
│                            └── 010 ───┤    │
├── 006 (UI) ──── 009 ────────────────────┐  │
└── 008 (download) ──────────────────────┤  │
                                          012 (API)
                                          └── 013 (CLI) ← also needs 006, 009
```

## Recommended build order

Start these in parallel after 001 lands: **002, 003, 004, 005, 006, 008**

Then in parallel: **007** (needs 005), **009** (needs 006), **010** (needs 005+002), **011** (needs 002)

Then: **012** (needs 003+004+007+008+010+011)

Finally: **013** (needs 006+009+012)
