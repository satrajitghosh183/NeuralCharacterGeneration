#version 450
layout(location = 0) in vec3 inPos;
layout(set = 0, binding = 0) uniform Cam {
  mat4 view;
  mat4 proj;
  vec4 lightDir;
  vec4 camPos;
} cam;
layout(location = 0) out vec3 vWorld;

void main() {
  vWorld = inPos;
  gl_Position = cam.proj * cam.view * vec4(inPos, 1.0);
}
