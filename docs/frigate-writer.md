# The Frigate writer

`raptor_vision.frigate` changes Frigate zones and masks safely. Kestrel uses it when it is installed. An Osprey-only
install uses it to write the PTZ camera's zones after the owner approves them in the console. Nobody copies or pastes
config.

## One writer per Frigate

Only one program may write to a given Frigate: Kestrel if you run it, otherwise Osprey. If both run, Osprey hands its
zone proposals to Kestrel and does not write. Two writers would each plan against a config the other is about to change.
The writer refuses a stale plan, so a collision is safe but wasteful. Inside one process, a lock keeps two `Writer`s for
the same Frigate address from interleaving.

## What it does

1. `plan(changes)` reads the live config and builds a diff and the exact API calls. It changes nothing.
2. `apply(plan, approved_by=..., backup_dir=...)` writes the old config to a new timestamped file (never overwritten),
   checks the config has not changed since the plan, sends the change, reads it back, and returns a `RollbackHandle`.
   If Frigate rejects the change or the read-back differs, it rolls back by itself and raises.
3. `rollback(handle)` undoes the change through the API and verifies. If that fails it puts the backup file back with
   `/api/config/save` (saved to disk; use `restart=True` to restart Frigate so it takes effect).

The handle converts to JSON (`to_json`, `from_json`) so you can store it. A `.handle.json` file is also written next to
each backup.

## Safety rules

- **Read-only by default.** Writing needs `Writer(..., allow_write=True)`. `FrigateClient` has no public write method.
- **Version gate.** Frigate's version must be in `supported` (default `0.18`, `0.17`: current stable and the one
  before). Anything else is refused. Masks are only written on 0.18 and later (earlier versions have unnamed masks);
  zones work on both.
- **Locked items are never touched.** An item is locked if its id ends in `_locked`, or its friendly name contains
  `[locked]` (any case), or the caller passes `locked=("front", "front/rv_gate")` (a camera, or camera/item).
  To lock something we created, rename it or edit its friendly name in Frigate.
- **Not ours means hands off.** We create items whose id starts with `rv_`. An existing item without that prefix is
  only changed if the `Change` says `allow_foreign=True`. Locked still wins.
- **Privacy masks always ask.** A `Change(..., privacy=True)` puts the plan in `needs_approval`, and `apply` raises
  unless `approved_by` is given. Nothing applies a privacy mask on its own.
- **Stale plans are refused.** The plan stores a hash of Frigate's raw config. `apply` re-reads it and refuses if it
  differs. Writing needs an admin token, because the raw config is admin-only.
- **Backups first.** The raw YAML is saved before any call is made.
- **No restart.** Zones and masks go through `PUT /api/config/set` with `requires_restart: 0` and the matching
  `update_topic`, so they take effect live. The writer never restarts Frigate unless you ask in `rollback`.
- Changes are all or nothing: a plan with any refused change raises and no call is made.

## Frigate endpoints used (checked against the Frigate 0.18.3 source, `frigate/api/app.py`)

| Call | Used for |
|---|---|
| `GET /api/version` | version gate |
| `GET /api/config` | live config, read-back |
| `GET /api/config/raw` | backup and stale check (admin). Frigate sends the YAML as a JSON string; both forms are read |
| `GET /api/config/schema.json` | `FrigateClient.schema()` (for callers; the writer validates by read-back) |
| `PUT /api/config/set` | JSON body `{config_data, requires_restart: 0, update_topic}`. Frigate flattens it to dotted paths, edits the YAML, validates, restores the old file itself if invalid, then publishes the update. A `null` value deletes a key |
| `POST /api/config/save?save_option=saveonly` or `restart` | rollback fallback; text/plain body is the whole YAML |

Paths written: `cameras.<cam>.zones.<id>`, `cameras.<cam>.motion.mask.<id>`, `cameras.<cam>.objects.mask.<id>`
(update topics `.../zones`, `.../motion`, `.../objects`). Zone and mask fields: `coordinates` (relative
`"x,y,x,y,..."`), `enabled`, `friendly_name`, and `objects` for zones.

## Limits

- No removal of existing items yet (rollback removes what a plan created).
- Rollback restores the fields a plan set to their earlier values. A field the plan added to an existing item (for example a
  friendly name) is not cleared; the backup file has the exact original.
- Frigate setups that keep config in git will see drift after an apply. Commit the result or use the backup to compare.
- Tests use a stub server, never a real Frigate.
