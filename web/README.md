# Rock Walker — browser game

A zero-setup little game to actually *play* with the avatar `ncg_cli avatar` produces. Pure
three.js in one HTML file — no install, no build, runs in any modern browser.

## Run
1. Put `rock_game.html` and a character `.glb` in the same folder.
2. Rename the glb to **`rock.glb`** (e.g. `rock_nrm_textured.glb` → `rock.glb`), or just **drag any
   `.glb` onto the page**.
3. Open `rock_game.html`. Because it loads a local file, serve the folder (browsers block `file://`
   fetch of the glb):
   ```bash
   cd web && python3 -m http.server 8000     # then open http://localhost:8000/rock_game.html
   ```
   (Drag-and-drop works even from `file://`.)

## Play
**W A S D** walk · **mouse-drag** look · **scroll** zoom · **Space** jump · collect the glowing coins.

The character's baked animation plays as locomotion (speeds up while moving). Textured + normal-mapped
glbs (`*_textured.glb`) show the recovered face/skin albedo and photometric-normal surface detail.

## Notes
- Best asset: `rock_<prefix>_textured.glb` (UV albedo + normal map, from `avatar --identity --uv-texture`).
- `rock_char.glb` (vertex-colored, animated) also works.
- This is the instant-play option; `apps/ncg_game` is the native Vulkan version (needs the Vulkan SDK).
