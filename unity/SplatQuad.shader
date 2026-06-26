// Minimal Gaussian-splat renderer for the skinned avatar (Built-in Render Pipeline). Draws one
// camera-facing quad per splat with a Gaussian alpha falloff, reading the per-frame skinned
// position/rotation buffers from GaussianSplatRenderer. Isotropic billboard + alpha blend, no
// depth sort — enough to see the skinned avatar move with the rig. For full fidelity (anisotropic
// projection + back-to-front sort) feed the skinned buffers to a dedicated GS plugin instead.
Shader "NCG/SplatQuad"
{
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            StructuredBuffer<float3> _Pos;
            StructuredBuffer<float4> _Rot;     // (unused in the isotropic billboard; kept for parity)
            StructuredBuffer<float3> _Scale;
            StructuredBuffer<float3> _Color;
            StructuredBuffer<float>  _Opacity;
            float _SizeScale;
            float4x4 _Local2World;

            struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float3 col : TEXCOORD1; float op : TEXCOORD2; };

            static const float2 QUAD[6] = {
                float2(-1,-1), float2(1,-1), float2(1,1),
                float2(-1,-1), float2(1,1), float2(-1,1)
            };

            v2f vert(uint vid : SV_VertexID, uint iid : SV_InstanceID)
            {
                float3 centerWorld = mul(_Local2World, float4(_Pos[iid], 1.0)).xyz;
                float3 s = _Scale[iid];
                float r = max(max(s.x, s.y), s.z) * 3.0 * _SizeScale;  // ~3-sigma extent
                float2 c = QUAD[vid];
                float3 right = float3(UNITY_MATRIX_V._m00, UNITY_MATRIX_V._m01, UNITY_MATRIX_V._m02);
                float3 up    = float3(UNITY_MATRIX_V._m10, UNITY_MATRIX_V._m11, UNITY_MATRIX_V._m12);
                float3 wp = centerWorld + (right * c.x + up * c.y) * r;
                v2f o;
                o.pos = UnityWorldToClipPos(wp);
                o.uv = c;
                o.col = _Color[iid];
                o.op = _Opacity[iid];
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float d2 = dot(i.uv, i.uv);                 // 0 at center, 1 at quad edge
                float a = exp(-0.5 * d2 * 9.0) * i.op;       // gaussian over the 3-sigma quad
                return float4(i.col, a);
            }
            ENDCG
        }
    }
}
