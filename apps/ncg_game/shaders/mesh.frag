#version 450
layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(set = 0, binding = 0) uniform Cam {
  mat4 view;
  mat4 proj;
  vec4 lightDir;
  vec4 camPos;
} cam;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 N = normalize(vNormal);
  vec3 L = normalize(-cam.lightDir.xyz);
  float diff = max(dot(N, L), 0.0);
  // Soft two-sided fill so back-facing/unlit regions stay visible.
  float fill = 0.35 + 0.65 * diff + 0.15 * max(dot(N, -L), 0.0);
  outColor = vec4(vColor * fill, 1.0);
}
