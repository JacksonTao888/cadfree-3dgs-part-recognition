"""Parametric GB standard parts for the FGIP set, built to nominal GB dimensions, exported as STL.
Threads are not modelled: the printed originals came from AutoCAD's simplified representation, which
draws a thread as a plain cylinder. Every other feature follows the standard's dimension table."""
import os, numpy as np, trimesh
from trimesh.creation import cylinder, box, extrude_polygon, revolve, triangulate_polygon
from shapely.geometry import Polygon
OUT=os.path.dirname(os.path.abspath(__file__)); T30=np.tan(np.radians(30))
def hexprism(s,h):
    r=s/np.sqrt(3); a=np.linspace(0,2*np.pi,7)[:-1]+np.pi/6
    m=extrude_polygon(Polygon(np.c_[r*np.cos(a),r*np.sin(a)]),h); m.apply_translation([0,0,-h/2]); return m
def cyl(d,h,z=0): 
    m=cylinder(radius=d/2,height=h,sections=128); m.apply_translation([0,0,z]); return m
def rev(pts):
    p=np.array(pts,float)
    if not np.allclose(p[0],p[-1]): p=np.vstack([p,p[0]])
    return revolve(p,sections=128)
def sub(a,*bs):
    for b in bs: a=a.difference(b)
    return a
def save(m,name,note):
    m.remove_unreferenced_vertices(); m.export(f"{OUT}/{name}.stl")
    print(f"{name:30s} {note:44s} watertight={str(m.is_watertight):5s} bbox={np.round(m.extents,1)}")

# ---------- nuts ----------
def hex_nut(s,m,d,chamfer=True):
    e=2*s/np.sqrt(3); body=hexprism(s,m)
    if chamfer:
        hc=(e/2-s/2)*T30
        body=body.intersection(rev([[0,-m/2],[s/2,-m/2],[e/2,-m/2+hc],[e/2,m/2-hc],[s/2,m/2],[0,m/2]]))
    return sub(body,cyl(d,m*3))
save(hex_nut(18,10.8,12),"nut_a__GB-T_6170_M12","1型六角螺母 s18 m10.8 d12  (m/s 0.60 vs 实测 0.63)")
save(hex_nut(16,5.0,10),"nutF__GB-T_6172.1_M10","六角薄螺母 s16 m5 d10  (m/s 0.31 vs 实测 0.30)")
save(hex_nut(18,12.2,12),"nut_e__GB-T_6175_M12","2型六角螺母 s18 m12.2 d12  (m/s 0.68 vs 实测 0.68)")
save(hex_nut(19,15.0,12),"nutH__GB-T_56_M12","六角厚螺母 s19 m15 d12  (m/s 0.79 vs 实测 0.81)")
save(hex_nut(19,18.3,12),"nut_c__thick_hex_TBD","很厚的六角螺母 m/s=0.96，超出常见国标厚度，待确认")
def castle_nut(s,m,w,n,d,slots=6):
    b=hex_nut(s,m,d,chamfer=True)
    for k in range(slots):
        sl=box([n,s*3,(m-w)*2]); sl.apply_transform(trimesh.transformations.rotation_matrix(k*np.pi/slots*2,[0,0,1]))
        sl.apply_translation([0,0,m/2]); b=b.difference(sl)
    return b
save(castle_nut(19,15.0,8.8,3.2,12),"nut_b__GB-T_6178_M12","1型六角开槽螺母 s19 m15 w8.8 n3.2 6槽  (m/s 0.79 vs 实测 0.80)")
def flange_nut(s,m,dc,c,d):
    hh=m-c; body=hexprism(s,hh); body.apply_translation([0,0,-m/2+c+hh/2])
    fl=rev([[0,-m/2],[dc/2,-m/2],[dc/2,-m/2+c],[s/2*0.99,-m/2+c+(dc-s)/2*0.5],[0,-m/2+c+(dc-s)/2*0.5]])
    return sub(body.union(fl),cyl(d,m*3))
save(flange_nut(15,12.2,22.3,1.8,10),"nutG__GB-T_6177.1_M10","六角法兰面螺母 s15 dc22.3 m12.2  (m/s 0.81 vs 实测 0.81)")
def round_slot_nut(dk,m,n,t,d,slots=4):
    b=sub(cyl(dk,m),cyl(d,m*3))
    for k in range(slots):
        sl=box([n,t*2,m*2]); sl.apply_translation([0,dk/2,0])
        sl.apply_transform(trimesh.transformations.rotation_matrix(k*2*np.pi/slots,[0,0,1])); b=b.difference(sl)
    return b
save(round_slot_nut(28,8,4,2,16),"nut_d__GB-T_812_M16x1.5","圆螺母 dk28 m8 n4 t2 4槽")
# ---------- screws ----------
def socket_cap(d,dk,k,ss,ts,L):
    soc=hexprism(ss,ts*2); soc.apply_translation([0,0,k])
    return sub(cyl(dk,k,k/2).union(cyl(d,L,-L/2)),soc)
save(socket_cap(12,18,12,10,6,45),"screw_a__GB-T_70.1_M12x45","内六角圆柱头螺钉 dk18 k12 s10 t6")
def hex_bolt(d,s,k,L):
    e=2*s/np.sqrt(3); hc=(e/2-s/2)*T30
    head=hexprism(s,k); head.apply_translation([0,0,k/2])
    head=head.intersection(rev([[0,0],[e/2,0],[e/2,k-hc],[s/2,k],[0,k]]))
    return head.union(cyl(d,L,-L/2))
save(hex_bolt(10,16,6.4,40),"screw_d__GB-T_5783_M10x40","六角头螺栓 全螺纹 s16 k6.4")
def shoulder(ds,dk,k,ls,dt,lt,ss,ts):
    soc=hexprism(ss,ts*2); soc.apply_translation([0,0,k])
    return sub(cyl(dk,k,k/2).union(cyl(ds,ls,-ls/2)).union(cyl(dt,lt,-ls-lt/2)),soc)
save(shoulder(10,15,8,20,8,13,6,3.5),"screw_f__GB-T_5281_d10-M8","内六角圆柱头轴肩螺钉 ds10 dk15 k8 轴肩20 M8x13")
def slotted_cheese(d,dk,k,n,t,L):
    sl=box([dk*1.5,n,t*2]); sl.apply_translation([0,0,k])
    return sub(cyl(dk,k,k/2).union(cyl(d,L,-L/2)),sl)
save(slotted_cheese(10,16,6,2.5,3.0,30),"screw_g__GB-T_65_M10x30","开槽圆柱头螺钉 dk16 k6 n2.5 t3")
# ---------- gears ----------
def gear_poly(z,mod,alpha=20.0,npts=12):
    a=np.radians(alpha); r=z*mod/2; rb=r*np.cos(a); ra=r+mod; rf=r-1.25*mod
    tmax=np.sqrt(max((ra/rb)**2-1,1e-9)); ts=np.linspace(0,tmax,npts)
    P=np.array([[np.cos(t)+t*np.sin(t),np.sin(t)-t*np.cos(t)] for t in ts])*rb
    th0=np.arctan2(P[0,1],P[0,0]); half=np.pi/(2*z)+(np.tan(a)-a)
    ang=np.arctan2(P[:,1],P[:,0])-th0; rr=np.linalg.norm(P,axis=1)
    up=np.c_[rr*np.cos(half-ang),rr*np.sin(half-ang)]; dn=np.c_[up[:,0],-up[:,1]]
    pts=[]
    for k in range(z):
        R=np.array([[np.cos(k*2*np.pi/z),-np.sin(k*2*np.pi/z)],[np.sin(k*2*np.pi/z),np.cos(k*2*np.pi/z)]])
        pts+=list(dn[::-1]@R.T)+list(up@R.T)
        a0=np.arctan2((up[0]@R.T)[1],(up[0]@R.T)[0])
        for t in np.linspace(a0,a0+2*np.pi/z-2*half,6)[1:-1]: pts.append([rf*np.cos(t),rf*np.sin(t)])
    return Polygon(pts).buffer(0)
def loft(poly,h,twist_deg,layers=48):
    P=np.array(poly.exterior.coords)[:-1]; N=len(P); V=[]; F=[]
    for i in range(layers+1):
        t=np.radians(twist_deg)*i/layers; R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        V.append(np.c_[P@R.T,np.full(N,-h/2+h*i/layers)])
    V=np.vstack(V)
    for i in range(layers):
        for j in range(N):
            a,b=i*N+j,i*N+(j+1)%N; c,d=a+N,b+N
            F+= [[a,b,d],[a,d,c]]
    fv,ff=triangulate_polygon(poly,engine="triangle" if False else None)
    base=len(V); V=np.vstack([V,np.c_[fv,np.full(len(fv),-h/2)],np.c_[fv,np.full(len(fv),h/2)]])
    nf=len(fv)
    for f in ff: F.append([base+f[0],base+f[2],base+f[1]])
    t=np.radians(twist_deg); R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    V[base+nf:base+2*nf,:2]=fv@R.T
    for f in ff: F.append([base+nf+f[0],base+nf+f[1],base+nf+f[2]])
    m=trimesh.Trimesh(V,np.array(F)); m.merge_vertices(); m.fix_normals(); return m
def gear(z,mod,face,bore,twist=0.0):
    return sub(loft(gear_poly(z,mod),face,twist),cyl(bore,face*3))
save(gear(20,2.0,19.0,4.0),"gear_a__spur_z20_m2","直齿圆柱齿轮 z20 m2 齿宽19 孔4")
save(gear(13,3.0,22.0,10.0,twist=22.0),"gear_b__helical_z13_m3","斜齿圆柱齿轮 z13 m3 齿宽22 孔10 螺旋22deg")
print("\n-> ", OUT)
