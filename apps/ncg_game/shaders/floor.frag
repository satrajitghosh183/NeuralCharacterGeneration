#version 450
layout(location = 0) in vec3 vWorld;
layout(location = 0) out vec4 outColor;

// Anti-aliased 1-unit grid with a fade toward the horizon.
void main() {
  vec2 g = abs(fract(vWorld.xz) - 0.5) / fwidth(vWorld.xz);
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  vec3 base = vec3(0.16, 0.18, 0.21);
  vec3 grid = vec3(0.33, 0.36, 0.41);
  float fade = clamp(1.0 - length(vWorld.xz) / 40.0, 0.0, 1.0);
  outColor = vec4(mix(base, grid, line * fade), 1.0);
}
