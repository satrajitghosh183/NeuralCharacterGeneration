// GaussianSplatRenderer — drop-in Unity component that renders the skinned Gaussian-splat avatar
// produced by `ncg_cli avatar` (rock_char.ply + rock_char.ply.skin) and deforms it every frame with
// the SAME skeleton that drives the rigged mesh (rock_char.glb). So whatever moves the skeleton —
// an Animator clip or a physics ragdoll — moves the splats identically (the math is proven in
// tests/fit/test_gs_skinning.cpp == ncg::fit::deform_avatar).
//
// SETUP (Unity 2022+):
//   1. Import rock_char.glb (glTFast). It yields a SkinnedMeshRenderer with the 55-bone skeleton.
//   2. Create an empty GameObject, add this component.
//   3. Assign: PlyFile = rock_char.ply (as a TextAsset — rename to rock_char.ply.bytes),
//              SkinFile = rock_char.ply.skin (rename to .bytes),
//              CharacterSMR = the imported SkinnedMeshRenderer,
//              SkinCompute = GaussianSplatSkin.compute, SplatMaterial = a material using SplatQuad.shader.
//   4. Press Play. The splats skin to the animator/ragdoll-driven skeleton.
//
// This is a compact, correct splatter (camera-facing quads with a Gaussian falloff). For maximum
// fidelity, swap the draw call for a dedicated GS plugin (Aras-P UnityGaussianSplatting) fed by the
// SkinnedPositions/SkinnedRotations buffers this component computes — same skinning, better blend/sort.
using System;
using System.IO;
using UnityEngine;

[DisallowMultipleComponent]
public class GaussianSplatRenderer : MonoBehaviour
{
    [Header("Assets")]
    public TextAsset PlyFile;             // rock_char.ply  (rename to .bytes to import as TextAsset)
    public TextAsset SkinFile;            // rock_char.ply.skin (rename to .bytes)
    public SkinnedMeshRenderer CharacterSMR;  // the rigged glb's renderer (skeleton + bindposes)
    public ComputeShader SkinCompute;     // GaussianSplatSkin.compute
    public Material SplatMaterial;        // material using SplatQuad.shader
    [Range(0.1f, 3f)] public float SizeScale = 1.0f;

    int _count;
    ComputeBuffer _restPos, _restRot, _scale, _color, _opacity, _joints, _weights;
    ComputeBuffer _outPos, _outRot, _boneMatrices;
    Matrix4x4[] _boneMtx;
    Matrix4x4[] _bindposes;
    Transform[] _bones;
    int _kernel;

    void Start()
    {
        if (PlyFile == null || SkinFile == null || CharacterSMR == null || SkinCompute == null)
        { Debug.LogError("GaussianSplatRenderer: assign all required fields."); enabled = false; return; }

        ParsePly(PlyFile.bytes);
        ParseSkin(SkinFile.bytes);

        _bones = CharacterSMR.bones;
        _bindposes = CharacterSMR.sharedMesh.bindposes;
        if (_bones.Length != _bindposes.Length)
            Debug.LogWarning($"bones {_bones.Length} != bindposes {_bindposes.Length}");
        _boneMtx = new Matrix4x4[_bones.Length];
        _boneMatrices = new ComputeBuffer(_bones.Length, 64);

        _outPos = new ComputeBuffer(_count, 12);
        _outRot = new ComputeBuffer(_count, 16);
        _kernel = SkinCompute.FindKernel("SkinSplats");
    }

    // ---- .ply (standard 3DGS) parsing: x y z, nx ny nz, f_dc0..2, opacity, scale0..2, rot0..3 ----
    void ParsePly(byte[] bytes)
    {
        int hdrEnd = IndexOf(bytes, "end_header\n");
        string header = System.Text.Encoding.ASCII.GetString(bytes, 0, hdrEnd);
        foreach (var line in header.Split('\n'))
            if (line.StartsWith("element vertex")) _count = int.Parse(line.Split(' ')[2]);
        int stride = 17 * 4;  // 17 float properties
        int off = hdrEnd + "end_header\n".Length;

        var pos = new Vector3[_count]; var rot = new Vector4[_count];
        var scl = new Vector3[_count]; var col = new Vector3[_count]; var op = new float[_count];
        const float SH_C0 = 0.28209479177387814f;
        using (var ms = new MemoryStream(bytes, off, _count * stride))
        using (var br = new BinaryReader(ms))
        {
            for (int i = 0; i < _count; i++)
            {
                float x = br.ReadSingle(), y = br.ReadSingle(), z = br.ReadSingle();
                br.ReadSingle(); br.ReadSingle(); br.ReadSingle();           // normals
                float f0 = br.ReadSingle(), f1 = br.ReadSingle(), f2 = br.ReadSingle();
                float opacity = br.ReadSingle();
                float s0 = br.ReadSingle(), s1 = br.ReadSingle(), s2 = br.ReadSingle();
                float r0 = br.ReadSingle(), r1 = br.ReadSingle(), r2 = br.ReadSingle(), r3 = br.ReadSingle();
                pos[i] = new Vector3(x, y, z);
                rot[i] = new Vector4(r1, r2, r3, r0);                        // (w,x,y,z) -> (x,y,z,w)
                scl[i] = new Vector3(Mathf.Exp(s0), Mathf.Exp(s1), Mathf.Exp(s2));
                col[i] = new Vector3(0.5f + SH_C0 * f0, 0.5f + SH_C0 * f1, 0.5f + SH_C0 * f2);
                op[i] = 1f / (1f + Mathf.Exp(-opacity));                     // sigmoid
            }
        }
        _restPos = Make(pos); _restRot = Make(rot); _scale = Make(scl); _color = Make(col); _opacity = Make(op);
    }

    void ParseSkin(byte[] bytes)
    {
        using (var ms = new MemoryStream(bytes))
        using (var br = new BinaryReader(ms))
        {
            int n = br.ReadInt32();
            if (n != _count) Debug.LogWarning($"skin N {n} != ply {_count}");
            var joints = new uint[n * 4]; var weights = new float[n * 4];
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < 4; k++) joints[i * 4 + k] = (uint)br.ReadInt32();
                for (int k = 0; k < 4; k++) weights[i * 4 + k] = br.ReadSingle();
            }
            _joints = new ComputeBuffer(n, 16); _joints.SetData(joints);
            _weights = new ComputeBuffer(n, 16); _weights.SetData(weights);
        }
    }

    void LateUpdate()
    {
        // Bone deformation in the renderer's local space (matches Unity skinning convention).
        Matrix4x4 rootInv = CharacterSMR.transform.worldToLocalMatrix;
        for (int j = 0; j < _bones.Length; j++)
            _boneMtx[j] = rootInv * _bones[j].localToWorldMatrix * _bindposes[j];
        _boneMatrices.SetData(_boneMtx);

        SkinCompute.SetInt("_Count", _count);
        SkinCompute.SetBuffer(_kernel, "_RestPos", _restPos);
        SkinCompute.SetBuffer(_kernel, "_RestRot", _restRot);
        SkinCompute.SetBuffer(_kernel, "_Joints", _joints);
        SkinCompute.SetBuffer(_kernel, "_Weights", _weights);
        SkinCompute.SetBuffer(_kernel, "_BoneMatrices", _boneMatrices);
        SkinCompute.SetBuffer(_kernel, "_OutPos", _outPos);
        SkinCompute.SetBuffer(_kernel, "_OutRot", _outRot);
        SkinCompute.Dispatch(_kernel, (_count + 255) / 256, 1, 1);
    }

    void OnRenderObject()
    {
        if (SplatMaterial == null || _outPos == null) return;
        SplatMaterial.SetBuffer("_Pos", _outPos);
        SplatMaterial.SetBuffer("_Rot", _outRot);
        SplatMaterial.SetBuffer("_Scale", _scale);
        SplatMaterial.SetBuffer("_Color", _color);
        SplatMaterial.SetBuffer("_Opacity", _opacity);
        SplatMaterial.SetFloat("_SizeScale", SizeScale);
        SplatMaterial.SetMatrix("_Local2World", CharacterSMR.transform.localToWorldMatrix);
        SplatMaterial.SetPass(0);
        // 6 verts (a quad) per splat, generated in the vertex shader.
        Graphics.DrawProceduralNow(MeshTopology.Triangles, 6, _count);
    }

    // ---- helpers ----
    static int IndexOf(byte[] hay, string needle)
    {
        byte[] n = System.Text.Encoding.ASCII.GetBytes(needle);
        for (int i = 0; i <= hay.Length - n.Length; i++)
        {
            bool ok = true;
            for (int j = 0; j < n.Length; j++) if (hay[i + j] != n[j]) { ok = false; break; }
            if (ok) return i;
        }
        throw new Exception("end_header not found in .ply");
    }
    ComputeBuffer Make(Vector3[] a){ var b=new ComputeBuffer(a.Length,12); b.SetData(a); return b; }
    ComputeBuffer Make(Vector4[] a){ var b=new ComputeBuffer(a.Length,16); b.SetData(a); return b; }
    ComputeBuffer Make(float[] a){ var b=new ComputeBuffer(a.Length,4); b.SetData(a); return b; }

    void OnDestroy()
    {
        foreach (var b in new[]{_restPos,_restRot,_scale,_color,_opacity,_joints,_weights,_outPos,_outRot,_boneMatrices})
            b?.Release();
    }
}
