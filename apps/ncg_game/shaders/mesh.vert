#version 450
// Skinned-mesh vertex shader: linear blend skinning with joint matrices that already fold in the
// inverse-bind transform (computed CPU-side from the glTF animation each frame).
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in uvec4 inJoints;
layout(location = 4) in vec4 inWeights;

layout(set = 0, binding = 0) uniform Cam {
  mat4 view;
  mat4 proj;
  vec4 lightDir;
  vec4 camPos;
} cam;
layout(set = 0, binding = 1) readonly buffer Joints { mat4 m[]; } joints;
layout(push_constant) uniform Push { mat4 model; } pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;

void main() {
  float wsum = inWeights.x + inWeights.y + inWeights.z + inWeights.w;
  mat4 skin = mat4(1.0);
  if (wsum > 1e-4) {
    skin = inWeights.x * joints.m[inJoints.x] + inWeights.y * joints.m[inJoints.y]
         + inWeights.z * joints.m[inJoints.z] + inWeights.w * joints.m[inJoints.w];
  }
  mat4 world = pc.model * skin;
  vec4 wp = world * vec4(inPos, 1.0);
  gl_Position = cam.proj * cam.view * wp;
  vNormal = normalize(mat3(world) * inNormal);
  vColor = inColor;
}
