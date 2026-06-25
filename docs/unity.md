# Using the avatar in Unity

The pipeline outputs a **binary glTF (`.glb`)**: a rigged (55-joint SMPL-X skeleton), vertex-colored
mesh with a PBR material. Unity imports it via a free glTF package.

## 1. Make the avatar from photos (one command)

```bash
tools/make_avatar.sh my_photo.jpg --out my_avatar.glb        # one photo
tools/make_avatar.sh front.heic side.heic --out my_avatar.glb # several (uses the first for color)
```
(HEIC is fine — it's converted automatically.) You get `my_avatar.glb` locally.

## 2. Install a glTF importer in Unity (once)

Unity doesn't import `.glb` natively. Add **glTFast** (Unity's official glTF package):

- **Window → Package Manager → `+` → Add package by name…**
- Enter: `com.unity.cloud.gltfast`  → Add.
  (On older Unity, use `com.atteneder.gltfast`.)

## 3. Drop the avatar in

- Drag `my_avatar.glb` into your project's **Assets/** folder.
- glTFast imports it as a prefab containing a **SkinnedMeshRenderer** (the body), its **skeleton**
  (the 55 joints), and the material. Drag the prefab into the scene.
- **Vertex colors**: the mesh stores per-vertex color in `COLOR_0`. With URP, assign a material
  whose shader reads vertex color (e.g. a Shader Graph with a *Vertex Color* node into Base Color),
  or glTFast's default lit shader, to see the photo coloring. (Built-in RP: any vertex-color shader.)

## 4. Animate it

The skeleton is standard SMPL-X. To drive it:
- Apply any **SMPL-X / SMPL `.bvh` or humanoid animation** retargeted to the skeleton, or
- Configure the imported model as **Humanoid** (Inspector → Rig → Animation Type: Humanoid → map the
  bones) and use **Mixamo** or any humanoid AnimationClip.

## Notes / current limitations

- The mesh is the **body** (SMPL-X topology) with photo-sampled vertex color — a clean, lightweight,
  animatable avatar. Clothing geometry and a baked PBR texture atlas are future work.
- For **relighting** in-engine, the avatar carries albedo-like vertex color; Unity's lighting then
  shades it. The full intrinsic-albedo path (from `ncg_cli delight`) can be baked in as a texture
  later for physically-correct relighting.
- A native real-time **animate+relight splat** runtime (147 FPS, `ncg_cli runtime`) exists outside
  the engine; a native Unity plugin for it is the next systems step.
