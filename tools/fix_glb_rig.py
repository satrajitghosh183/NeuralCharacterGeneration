import sys, numpy as np
from pygltflib import GLTF2
CT={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}
inp,out=sys.argv[1],sys.argv[2]
g=GLTF2().load(inp); blob=bytearray(g.binary_blob())
def acc(i):
    a=g.accessors[i]; bv=g.bufferViews[a.bufferView]; o=(bv.byteOffset or 0)+(a.byteOffset or 0)
    return np.frombuffer(blob,CT[a.componentType],a.count*NC[a.type],o).reshape(a.count,NC[a.type]).copy()
p=g.meshes[0].primitives[0]
V=acc(p.attributes.POSITION).astype(np.float64)
Jo=acc(p.attributes.JOINTS_0).astype(np.int64)
W=acc(p.attributes.WEIGHTS_0).astype(np.float64)
if W.max()>1.5: W=W/255.0
skin=g.skins[0]; joints=skin.joints; J=len(joints); node2slot={n:k for k,n in enumerate(joints)}
# per-vertex per-joint weight matrix Wm [n,J]
n=V.shape[0]; Wm=np.zeros((n,J))
for c in range(Jo.shape[1]): np.add.at(Wm, (np.arange(n), Jo[:,c]), W[:,c])
parent_node=[-1]*len(g.nodes)
for i,nd in enumerate(g.nodes):
    for ch in (nd.children or []): parent_node[ch]=i
pslot=[node2slot.get(parent_node[joints[k]],-1) for k in range(J)]
def centroid(mask):
    w=mask.sum()
    return (V*mask[:,None]).sum(0)/w if w>1e-8 else None
Jpos=np.full((J,3),np.nan)
for k in range(J):
    pk=pslot[k]
    pos=None
    if pk!=-1:                                   # articulation = verts shared by k and its parent
        sh=np.minimum(Wm[:,k],Wm[:,pk])
        if sh.sum()>1e-6: pos=centroid(sh)
    if pos is None: pos=centroid(Wm[:,k])        # fallback: plain weighted centroid
    if pos is None: pos=np.array([0,0,0.0])
    Jpos[k]=pos
# fill leftover nan top-down by parent
for k in sorted(range(J),key=lambda k:(0 if pslot[k]==-1 else 1)):
    if np.isnan(Jpos[k]).any(): Jpos[k]=Jpos[pslot[k]] if pslot[k]!=-1 else np.zeros(3)
# ---- write node translations (local = world - parent_world) ----
for k in range(J):
    pk=pslot[k]; loc=Jpos[k]-(Jpos[pk] if pk!=-1 else 0.0)
    g.nodes[joints[k]].translation=[float(loc[0]),float(loc[1]),float(loc[2])]
    g.nodes[joints[k]].matrix=None; g.nodes[joints[k]].rotation=None; g.nodes[joints[k]].scale=None
# ---- rewrite inverse-bind matrices = translate(-Jpos) column-major ----
ai=skin.inverseBindMatrices; a=g.accessors[ai]; bv=g.bufferViews[a.bufferView]
off=(bv.byteOffset or 0)+(a.byteOffset or 0)
ibm=np.zeros((J,16),np.float32)
for k in range(J):
    M=np.eye(4); M[:3,3]=-Jpos[k]; ibm[k]=M.T.reshape(-1)   # column-major
blob[off:off+J*16*4]=ibm.tobytes()
g.set_binary_blob(bytes(blob)); g.save(out)
print("wrote",out," skeleton y:[%.2f,%.2f] x:[%.2f,%.2f]"%(Jpos[:,1].min(),Jpos[:,1].max(),Jpos[:,0].min(),Jpos[:,0].max()))
