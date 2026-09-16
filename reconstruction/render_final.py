import os, sys, math, numpy as np, torch, glob
sys.path.insert(0, os.path.expanduser("~/tools/SuGaR/gaussian_splatting"))
from scene import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from types import SimpleNamespace
import torchvision

HOME = os.environ.get("RECON_WORK") or os.path.expanduser("~/screw_recon")
# prefer the finer pruned model, fall back to the hi pruned model
CANDS = [f"{HOME}/gs_obj_hi/output_fine/point_cloud/iteration_98/point_cloud.ply",
         f"{HOME}/gs_obj_hi/output/point_cloud/iteration_98/point_cloud.ply"]
PLY = next(p for p in CANDS if os.path.exists(p))
POSES = f"{HOME}/gs_obj_hi/sparse/0/images.txt"
OUT = f"{HOME}/renders_final"; os.makedirs(OUT, exist_ok=True)
W, H, FX = 1800, 1200, 10969.6
FoVx = 2*math.atan(W/(2*FX)); FoVy = 2*math.atan(H/(2*FX))
N = 120                                      # frames per 360° orbit
ELEV_OFFSETS = [0, +8, +15]                  # degrees relative to capture elevation
                                             # (-15 dropped: near-horizon shows floater streaks)
                                             # +8 adds an intermediate band -> denser novel-view coverage
                                             # that the raw single-ring capture does not contain
print("model:", PLY)

def qR(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
C=[]
for ln in open(POSES):
    if ln.startswith('#') or not ln.strip(): continue
    t=ln.split()
    if len(t)>=10 and t[-1].endswith('.jpg'):
        q=np.array(list(map(float,t[1:5]))); tr=np.array(list(map(float,t[5:8]))); C.append(-qR(q).T@tr)
C=np.array(C); O=C.mean(0); U,S,Vt=np.linalg.svd(C-O); n=Vt[-1]

gauss=GaussianModel(3); gauss.load_ply(PLY)
xyz=gauss.get_xyz.detach().cpu().numpy()
lum=(0.5+0.28209479177387814*gauss._features_dc.detach().cpu().numpy()[:,0,:]).clip(0,1)@[0.299,0.587,0.114]
op=torch.sigmoid(gauss._opacity).detach().cpu().numpy()[:,0]
T=np.median(xyz[(lum>0.45)&(op>0.3)],0)
if np.dot(n, C.mean(0)-T)<0: n=-n
D=np.linalg.norm(C-T,axis=1).mean()
phi0=np.arcsin(np.clip(((C-T)@n)/np.linalg.norm(C-T,axis=1),-1,1)).mean()
e1=Vt[0]-np.dot(Vt[0],n)*n; e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
print(f"capture elevation={math.degrees(phi0):.1f}deg  -> orbits at "
      f"{[round(math.degrees(phi0))+o for o in ELEV_OFFSETS]} deg")

bg=torch.zeros(3,device="cuda")
pipe=SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
def look_at(Cp,tg,up):
    z=tg-Cp; z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.stack([x,y,z],0)
def render_to(Cp,tg,path):
    Rw2c=look_at(Cp,tg,n); Tw2c=-Rw2c@Cp
    wv=torch.tensor(getWorld2View2(Rw2c.T.astype(np.float32),Tw2c.astype(np.float32))).transpose(0,1).cuda()
    proj=getProjectionMatrix(0.01,100,FoVx,FoVy).transpose(0,1).cuda()
    cam=SimpleNamespace(FoVx=FoVx,FoVy=FoVy,image_height=H,image_width=W,world_view_transform=wv,
        full_proj_transform=(wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),camera_center=wv.inverse()[3,:3])
    torchvision.utils.save_image(render(cam,gauss,pipe,bg)["render"].clamp(0,1),path)

for off in ELEV_OFFSETS:
    tag = "elev_base" if off == 0 else (f"elev_+{off}" if off > 0 else f"elev_{off}")
    d=f"{OUT}/{tag}"; os.makedirs(d,exist_ok=True); el=phi0+math.radians(off)
    for i in range(N):
        az=2*math.pi*i/N
        Cp=T+D*(math.cos(el)*(math.cos(az)*e1+math.sin(az)*e2)+math.sin(el)*n)
        render_to(Cp,T,f"{d}/{i:04d}.png")
    print(f"rendered {tag}: {N} frames")
print("done ->", OUT)
