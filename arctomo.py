
import matplotlib.pyplot as plt
import random
import torch

from torch.multiprocessing import Pool
from torch.autograd import Variable

def chase(a,c,d,b):
    
    n=len(a)
    x=torch.ones(n)
    p=torch.ones(n)
    q=torch.ones(n-1)   
    y=torch.ones(n)

    p[0]=a[0]   
    for i in range(n-1):
       q[i]=c[i]*(1/p[i])
       p[i+1]=a[i+1]-d[i]*q[i]
    
    y[0]=b[0]/p[0]
    for i in range(1,n):
        y[i]=(b[i]-d[i-1]*y[i-1])/p[i]
    
    x[n-1]=y[n-1]
    for i in range(n-2,-1,-1):
        x[i]=y[i]-q[i]*x[i+1]
 #       x[i]=y[i]-x[i+1]*c[i]/p[i] 

    return x

def diff(x):
    n=len(x)
    y=torch.ones(n-1)
    for i in range(n-1):
        y[i]=x[i+1]-x[i]
    return y
        
def triangle(x,y0,p):
    y=torch.ones(len(y0))*-1
    for j in range(len(y)):
        if(int(y0[j])==y0[j] and j>0 and j>y0[j] and y0[j]>=0 and y0[j]<len(y0)):
            y[j]=y[int(y0[j])]
        else:
            y[j]=y0[j];
    h=diff(x)
    g=diff(y)
    n=len(h)
    c=h[1:n]
    d=h[0:n-1]
    a=2*(c+d)
    f=g/h
    b=6*diff(f)
    if p==1:
        a=torch.cat((torch.tensor([1]),a,torch.tensor([1])))
        b=torch.cat((torch.tensor([0]),b,torch.tensor([0])))
        c=torch.cat((torch.tensor([0]),c))
        d=torch.cat((d,torch.tensor([0])))
    else:
        a0=0
        b0=0
        a=torch.cat((torch.tensor([2*h[0]]),a,torch.tensor([2*h[n-1]])))  
        b=torch.cat((torch.tensor([6*(f[0]-a0)]),b,torch.tensor([6*(b0-f[n-1])])))
        c=torch.cat((torch.tensor([h[0]]),c))
        d=torch.cat((d,torch.tensor([h[n-1]]))) 
    z=chase(a,c,d,b)    
    s=z[0:n]   
    t=diff(z)
    a=y[0:n]
    b=f-h*s/2-h*t/6
    c=s/2
    d=t/(6*h)    

    return a,b,c,d

def chasetriangle(x,a,b,c,d,z):
    k=len(x)-2
    for i in range(len(x)-1):
        if z>=x[i] and z<x[i+1]:
            k=i
            break
    if z<x[0]:
        k=0
    u=z-x[k]
    t=a[k]+u*(b[k]+u*(c[k]+u*d[k])) 
    t1=b[k]+u*(c[k]*2+u*d[k]*3)
    t2=c[k]*2+u*d[k]*6        
    return t,t1,t2

def readtimes(file,dt,mul=1000):
    fp=open(file)
    i=0;j=0;m=0;
    x=[];y=[];flag=[]
    ray=[];pors=[];t0=[];t=[]
    while True:
        line=fp.readline()
        if not line: break
        line = line.rstrip()
        words=line.split()
        if words[0]=="#raypath":
            j=1
            if len(t)>0: 
                t0+=[t]
                t=[]
        if line[0]=='#': continue    
        i+=1;
        if i==1: shotx=[x1 * mul for x1 in map(float,line.split())]
        if i==2: recx=[x1 * mul for x1 in map(float,line.split())];#recx=recx[0:-1:3]
        if i==3: vp=[x1 * mul for x1 in map(float,line.split())]
        if i==4: vpflag=list(map(int,line.split()))
        if i==5: vp0=[x1 * mul for x1 in map(float,line.split())]
        if i==6: vpflag1=list(map(int,line.split()))
        if i==7: vs=[x1 * mul for x1 in map(float,line.split())]
        if i==8: vsflag=list(map(int,line.split()))
        if i==9: vs0=[x1 * mul for x1 in map(float,line.split())]
        if i==10: vsflag1=list(map(int,line.split()))
        if i>10 and j==0:
            if m%3==0: x+=[[x1 * mul for x1 in map(float,line.split())]]
            if m%3==1: y+=[[x1 * mul for x1 in map(float,line.split())]]
            if m%3==2: flag+=[list(map(int,line.split()))]
            m+=1
        if i>10 and j>=1:
            if j==1: ray+=[list(map(int,line.split()))]
            if j==2: pors+=[list(map(int,line.split()))]
            if j>2: 
                tt=list(map(float,line.split()));#tt=tt[0:-1:3]
                for k in range(len(tt)):
                    if tt[k]>0:
                        tt[k]+=(random.random()-0.5)*dt*2;
                t+=[tt]
            j+=1
        
    fp.close()
    if len(t)>0: 
        t0+=[t]
        t=[]
    return shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0

def writemodel(shotx,recx,vp0,vpflag,vp01,vpflag1,vs0,vsflag,vs01,vsflag1,x,y0,flag,ray,pors,t0,f,file):		
    lines=[];
    lines+=["# shot points\n"]
    lines+=[" ".join(str(i) for i in shotx)+"\n"]
    lines+=["# receivers\n"]
    lines+=[" ".join(str(i) for i in recx)+"\n"]
    lines+=["#P velocity model for each layer\n"]
    lines+=[" ".join("{:.6f}".format(i) for i in vp0)+"\n"]
    lines+=["#P velocity inversion flag\n"]
    lines+=[" ".join(str(i) for i in vpflag)+"\n"]
    lines+=["#P bottom model for each layer\n"]
    lines+=[" ".join("{:.6f}".format(i) for i in vp01)+"\n"]
    lines+=["#P bottom inversion flag\n"]
    lines+=[" ".join(str(i) for i in vpflag1)+"\n"]
    
    lines+=["#S velocity model for each layer\n"]
    lines+=[" ".join("{:.6f}".format(i) for i in vs0)+"\n"]
    lines+=["#S velocity inversion flag\n"]
    lines+=[" ".join(str(i) for i in vsflag)+"\n"]
    lines+=["#S bottom model for each layer\n"]
    lines+=[" ".join("{:.6f}".format(i) for i in vs01)+"\n"]
    lines+=["#S bottom inversion flag\n"]
    lines+=[" ".join(str(i) for i in vsflag1)+"\n"]
    for j in range(len(y0)):
        lines+=["# "+str(j+1)+" interface\n"]
        lines+=[" ".join(str(i) for i in x[j])+"\n"]
        lines+=[" ".join("{:.6f}".format(i) for i in y0[j])+"\n"]
        lines+=[" ".join(str(i) for i in flag[j])+"\n"]   
    for j in range(len(ray)):
        lines+=["#raypath code according to interface number\n"]
        lines+=[" ".join(str(i) for i in ray[j])+"\n"]
        lines+=["# P or S between each raypath segment\n"]
        lines+=[" ".join(str(i) for i in pors[j])+"\n"]
        for k in range(len(t0[j])):
            lines+=["# shot "+str(k+1)+" travel times\n"]
            tt=t0[j][k].copy()
            for m in range(len(tt)):                
                n=j*len(shotx)*len(recx)+k*len(recx)+m
#                print(m,n,tt[m])
                if f[n]!=0 and tt[m]!=0:
                    tt[m]+=f[n]
                else:
                    tt[m]=0
                        
            lines+=[" ".join("{:.6f}".format(i) for i in tt)+"\n"]
        
    
    fp=open(file,"w")
    fp.writelines(lines)
    fp.close()
def writetomoxyz(x,y,dx,dy,vp,vp0,vs,vs0,rho,rho0):
# ORIGIN_X ORIGIN_Z END_X END_Z
# SPACING_X SPACING_Z
# NX NZ
# VP_MIN VP_MAX VS_MIN VS_MAX RHO_MIN RHO_MAX
# x(1) z(1) vp vs rho
# x(2) z(1) vp vs rho
# ...
# x(NX) z(1) vp vs rho
# x(1) z(2) vp vs rho
# x(2) z(2) vp vs rho
# ...
# x(NX) z(2) vp vs rho
# x(1) z(3) vp vs rho
# ...
# ...
# x(NX) z(NZ) vp vs rho
    a1=[];b1=[];c1=[];d1=[];#rho=[2200,2400,2600,2800];rho0=[2300,2500,2700,2900];
    minx=1.e18;miny=1.e18;maxx=-1.e18;maxy=-1.e18;#dx=10;dy=10;
    for n in range(len(x)):
        minx=min(minx,min(x[n])); miny=min(miny,min(y[n])); maxx=max(maxx,max(x[n])); maxy=max(maxy,max(y[n]))+dy
        a0,b0,c0,d0=triangle(x[n],y[n],0)   
        a1+=[a0];b1+=[b0];c1+=[c0];d1+=[d0]
    lines=[]
    nx=int((maxx-minx)/dx);    ny=int((maxy-miny)/dy);
    lines+=["{:.3f}".format(minx)+" "+"{:.3f}".format(miny)+" "+"{:.3f}".format(maxx)+" "+"{:.3f}".format(maxy)+"\n"]
    lines+=["{:.3f}".format(dx)+" "+"{:.3f}".format(dy)+"\n"]
    lines+=["{:d}".format(nx)+" "+"{:d}".format(ny)+"\n"]
    lines+=["{:.3f}".format(min(min(vp),min(vp0)))+" "+"{:.3f}".format(max(max(vp),max(vp0)))+" "+"{:.3f}".format(min(min(vs),min(vs0)))+" "+"{:.3f}".format(max(max(vs),max(vs0)))+" "+"{:.3f}".format(min(min(rho),min(rho0)))+" "+"{:.3f}".format(max(max(rho),max(rho0)))+"\n"]
    yy0=[]

    for k in range(len(x)):
        yy=[chasetriangle(x[k],a1[k],b1[k],c1[k],d1[k],minx + i * dx)[0] for i in range(nx)] 
        yy0+=[yy]
 
    for j in range(ny):
        yy=miny+j*dy
        for i in range(nx):
            xx=minx+i*dx
            for k in range(len(x)-1,0,-1):
                if yy0[k-1][i]>=yy and yy0[k][i]<=yy:break;
            rr=(yy0[k-1][i]-yy)/(yy0[k-1][i]-yy0[k][i]);
            lines+=["{:.3f}".format(xx) +" "+"{:.3f}".format(yy)+" "+"{:.3f}".format(vp[k-1]+(vp0[k-1]-vp[k-1])*rr)+" "+"{:.3f}".format(vs[k-1]+(vs0[k-1]-vs[k-1])*rr)+" "+"{:.3f}".format(rho[k-1]+(rho0[k-1]-rho[k-1])*rr)+"\n"]
    fp=open("tomo_file.xyz","w")
    fp.writelines(lines)
    fp.close()        
def initxx(y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1):
    xx=[];ww=[]
   
    for i in range(len(vp)):
        if vpflag[i]==1 and vp[i]>0:
            xx+=[vp[i]];ww+=[100]
    for i in range(len(vs)):
        if vsflag[i]==1 and vs[i]>0:
            xx+=[vs[i]];ww+=[100]
    for i in range(len(vp0)):
        if vpflag1[i]==1 and vp0[i]>0:
            xx+=[vp0[i]];ww+=[100]
    for i in range(len(vs0)):
        if vsflag1[i]==1 and vs0[i]>0:
            xx+=[vs0[i]];ww+=[100]
    for i in range(len(y)):
        for j in range(len(y[i])):
            if flag[i][j]==1 and (j<=y[i][j] or int(y[i][j])!=y[i][j] or y[i][j]<0 or y[i][j]>=len(y[i])):
                xx+=[y[i][j]] ;  ww+=[1]   
    return xx,ww

def updatexx(xx,yy,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1):
    k=0
    vp1=vp.copy();vs1=vs.copy();vp2=vp0.copy();vs2=vs0.copy()
    y1=yy.copy();
    for i in range(len(yy)):y1[i]=yy[i].copy()

    
    for i in range(len(vp1)):
        if vpflag[i]==1 and vp1[i]>0:
            vp1[i]=xx[k];k+=1
    for i in range(len(vs1)):
        if vsflag[i]==1 and vs1[i]>0:
            vs1[i]=xx[k];k+=1
    for i in range(len(vp2)):
        if vpflag1[i]==1 and vp2[i]>0:
            vp2[i]=xx[k];k+=1
    for i in range(len(vs2)):
        if vsflag1[i]==1 and vs2[i]>0:
            vs2[i]=xx[k];k+=1
    for i in range(len(y1)):
        for j in range(len(y1[i])):
            if flag[i][j]==1 and (j<=y1[i][j] or int(y1[i][j])!=y1[i][j] or y1[i][j]<0 or y1[i][j]>=len(y1[i])):
                y1[i][j]=xx[k];k+=1
                 
            
    

    return y1,vp1,vs1,vp2,vs2

def ratios(tt,x0,xn,x,a,b,c,d,vv,vv0,ray):
    xx=torch.cat((torch.tensor([x0]),tt,torch.tensor([xn])))
    n=len(xx)
    y=torch.zeros(n)
    u=torch.zeros(n)
    g=torch.zeros(n-1)

    vv1=vv0*1
    for i in range(n):
        j=ray[i]-1        
        y[i],u[i],u2=chasetriangle(x[j],a[j],b[j],c[j],d[j],xx[i])
        if i>0:
            x2=(xx[i]+xx[i-1])/2
            k=min(ray[i-1]-1,j)
            y1,u1,u3=chasetriangle(x[k],a[k],b[k],c[k],d[k],x2)
            y2,u2,u3=chasetriangle(x[k+1],a[k+1],b[k+1],c[k+1],d[k+1],x2)

            g[i-1]=(vv1[i-1]-vv[i-1])/(y1-y2)
            if abs(g[i-1])<1.e-6 and g[i-1]!=0:
                g[i-1]=0
       
    dx=diff(xx);dy=diff(y)
    ff=torch.sqrt(dx*dx+dy*dy)

    p=dx/(ff*vv);
    nom11=torch.zeros(n-2);nom12=torch.zeros(n-2)
    nom21=torch.zeros(n-2);nom22=torch.zeros(n-2)

     
    t=2*ff/(vv+vv1)
    for k in range(n-1):
        if  g[k]==0:
            if ray[k]==ray[k+1]:
                t[k]=ff[k]/vv[k]
                dy1=torch.sign(dx[k])*dx[k]*u[k]
                dy2=torch.sign(dx[k])*dx[k]*u[k+1]
                ff1=torch.sqrt(dx[k]*dx[k]+dy1*dy1)
                ff2=torch.sqrt(dx[k]*dx[k]+dy2*dy2)
                if k>=1:
                    nom12[k-1]=dx[k]/(ff2*vv[k])            
                    nom22[k-1]=dy2/(ff2*vv[k])                    
                if k<n-2:
                    nom11[k]=dx[k]/(ff1*vv[k])
                    nom21[k]=dy1/(ff1*vv[k]) 
            elif  ray[k]<ray[k+1]:
                t[k]=ff[k]/vv[k]*0.4+ff[k]/vv1[k]*0.6
                if k>=1:
                    nom12[k-1]=dx[k]/(ff[k]*vv[k])            
                    nom22[k-1]=dy[k]/(ff[k]*vv[k])
                if k<n-2:
                    nom11[k]=dx[k]/(ff[k]*vv1[k])
                    nom21[k]=dy[k]/(ff[k]*vv1[k])
            elif  ray[k]>ray[k+1]:
                t[k]=ff[k]/(0.6*vv[k]+0.4*vv1[k])
                if k>=1:
                    nom12[k-1]=dx[k]/(ff[k]*vv1[k])            
                    nom22[k-1]=dy[k]/(ff[k]*vv1[k])
                if k<n-2:
                    nom11[k]=dx[k]/(ff[k]*vv[k])
                    nom21[k]=dy[k]/(ff[k]*vv[k])
 
        else:
            if ray[k]==ray[k+1]:
                p1=2/(dx[k]**2*g[k]**2 + 4*vv[k]**2)**(1/2)
                if p1*max(vv[k],vv1[k])<1:p1=p1*2
                if dx[k]==0:
                    dx[k]
                else:
                    p1=torch.sign(dx[k])*p1
                p[k]=p1
                
                pp=torch.sign(g[k])*torch.abs(1/vv[k]**2 - p1**2)**(1/2)
                t[k]=torch.sign(g[k])*2*torch.log((torch.abs(pp) + 1/torch.abs(vv[k]))/torch.abs(p1))/g[k]
                if k>=1:
                    nom12[k-1]=p1
                    nom22[k-1]=-pp
                if k<n-2:
                    nom11[k]=p1;
                    nom21[k]=pp;
            else:  
                pmax=1/max(vv[k],vv1[k])
                maxdx=((1 - pmax**2*vv[k]**2)**(1/2)-(1 - pmax**2*vv1[k]**2)**(1/2))/(g[k]*pmax)                     
                p2=(2*dx[k])/torch.abs((vv[k]**2 - vv1[k]**2)**2/g[k]**2 + dx[k]**4*g[k]**2 + 2*dx[k]**2*(vv[k]**2 + vv1[k]**2))**(1/2)
                # xarc,yarc=drawarc([xx[k],y[k]],[xx[k+1],y[k+1]] ,g[k],p2) 
                # up=min(ray[k]-1,ray[k+1]-1);down=max(ray[k]-1,ray[k+1]-1);
                # yup=xarc*1;ydown=xarc*1;
                # for j in range(len(xarc)):                
                #     yup[j],ll,kk=chasetriangle(x[up],a[up],b[up],c[up],d[up],xarc[j])
                #     ydown[j],ll,kk=chasetriangle(x[down],a[down],b[down],c[down],d[down],xarc[j])
                # if maxdx<abs(dx[k]) and any(yup[j]+0.02<yarc[j] or ydown[j]-0.02>yarc[j] for j in range(len(xarc))):
                #     p2=torch.sign(dx[k])*(pmax*2-torch.sign(dx[k])*p2)
                if maxdx<abs(dx[k]):
                    p2=torch.sign(dx[k])*(pmax*2-torch.sign(dx[k])*p2)
                p[k]=p2; 
                pp1=torch.abs(1/vv1[k]**2 - p2**2)**(1/2)
                pp=torch.abs(1/vv[k]**2  - p2**2)**(1/2)
                t[k]=torch.log((torch.abs(pp) + 1/torch.abs(vv[k]))/(torch.abs(pp1) + 1/torch.abs(vv1[k])))/g[k]
                if ray[k]<ray[k+1]:
                    if k>=1:                    
                        nom12[k-1]=p2
                        nom22[k-1]=-pp
                    if k<n-2:
                        nom11[k]=p2;
                        nom21[k]=-pp1;                        
                else:
                    if k>=1:
                        nom12[k-1]=p2
                        nom22[k-1]=pp1 
                    if k<n-2:  
                        nom11[k]=p2;
                        nom21[k]=pp;
                 
                
    nom1=nom11*1.e6-nom12*1.e6;
    nom2=nom21*1.e6-nom22*1.e6          
    f=-(nom1+nom2*u[1:n-1])
    return f,t,y,g,p
    
def velo(vp,vs,vp1,vs1,ray,pors):
    nn=len(ray);vv=torch.zeros(nn-1);vv1=torch.zeros(nn-1);nv=[]
    for i in range(1,nn):
        nv+=[min(ray[i]-1,ray[i-1]-1)]
    for i in range(len(pors)):
        if pors[i]==1:
            vv[i]=vp[nv[i]];
            j=nv[i]-1;
            while vv[i]<0:
                vv[i]=vp1[j];
                if vv[i]<0:vv[i]=vp[j];
                j=j-1;
            vv1[i]=vp1[nv[i]];
            j=nv[i]
            while vv1[i]<0:
                vv1[i]=vp[j];
                if vv1[i]<0:vv1[i]=vp1[j-1];
                j=j-1;
        if pors[i]==-1:
            vv[i]=vs[nv[i]];
            j=nv[i]-1;
            while vv[i]<0:
                vv[i]=vs1[j];
                if vv[i]<0:vv[i]=vs[j];
                j=j-1;
            vv1[i]=vs1[nv[i]];
            j=nv[i]
            while vv1[i]<0:
                vv1[i]=vs[j];
                if vv1[i]<0:vv1[i]=vs1[j-1];
                j=j-1;
    return vv,vv1

def myopt(xx,x0,xn,x,a,b,c,d,vv,vv1,ray):
    II=torch.eye(len(xx));damp=1.e-8
    gg=torch.eye(len(xx))
    uu=torch.eye(len(xx))
    xx.requires_grad_(True)    
    niter=0;
    f,t,y,g,p=ratios(xx,x0,xn,x,a,b,c,d,vv,vv1,ray)
    for i in range(len(f)):
        gg[i] = torch.autograd.grad(f[i], xx,retain_graph=True)[0]
    while True:
        gt=torch.transpose(gg, 0, 1)
        dx=torch.linalg.solve(torch.mm(gt,gg)+damp*II,torch.mv(gt,f))
        x1=xx-dx
        f0=f; 
        x1.retain_grad()
        f,t,y,g,p=ratios(x1,x0,xn,x,a,b,c,d,vv,vv1,ray)

        if torch.norm(f)<torch.norm(f0) and all(element > min(x[0]) for element in x1) and all(element <  max(x[0]) for element in x1):
            badgrad=False
            for i in range(len(f)):
                uu[i] = torch.autograd.grad(f[i], x1,retain_graph=True)[0]
                if any(torch.isnan(uu[i])):
                    badgrad=True
#            print(uu,niter)
            if badgrad:break
            
            gg=uu
            damp/=1.2
            xx=x1 
        else:
            damp*=2.5            
            f=f0
        niter+=1
        if torch.norm(dx)<1.e-7 or damp>1.e20 or niter>1000:break
#        if torch.norm(dx)<1.:break
    return xx,f,t,y,g,p,uu

def drawarc(A, B, g, p):  
    nn=100;
    if g * p == 0:  
        return  torch.linspace(A[0], B[0], nn),torch.linspace(A[1], B[1], nn)
      
    # R  
    R = 1 / abs(g * p).detach()
    B=torch.Tensor(B);A=torch.Tensor(A)
    vector_AB = B - A
    distance_AB = torch.norm(vector_AB)  
  
      
    if distance_AB > 2 * R:  
#        print("nonexist arc")  
        return [0],[0] 
  
    
    mid_point = (A+B) / 2  
      
  
    perp_vector = torch.Tensor([-vector_AB[1], vector_AB[0]])
  
    perp_vector = perp_vector/torch.norm(perp_vector)  
   
    offset = -torch.sqrt(R**2 - (distance_AB / 2)**2)  
    # if B[1] > A[1]:  
    #     offset = -offset  
    if B[0] > A[0]:  
         offset = -offset  
    if g < 0:  
        offset = -offset  
      
    
    center = mid_point + offset * perp_vector  
  
    
    vector_A = A - center  
    vector_B = B - center  
    theta1 = torch.arctan2(vector_A[1], vector_A[0])  
    theta2 = torch.arctan2(vector_B[1], vector_B[0])
  
    if theta1 > theta2:  
            theta1, theta2 = theta2, theta1  

    if theta2 - theta1 > torch.pi:  
            theta1, theta2 = theta2, theta1  
            theta1 -= 2*torch.pi    
  
    dtheta=(theta2-theta1)/nn/10;
    theta = torch.linspace(theta1+dtheta, theta2-dtheta, nn)
    x = center[0] + R * torch.cos(theta)  
    y = center[1] + R * torch.sin(theta) 
    
    return x,y

            
def raytrace(shotx,recx,ray,x,a1,b1,c1,d1,a2,b2,c2,d2,xx0,vv,vv1,vv0,vv01):

    yy=torch.linspace(shotx, recx,len(ray))
    x1=yy[1:len(ray)-1]
    x0,f,t,y,g,p,uu=myopt(x1,shotx,recx,x,a1,b1,c1,d1,vv,vv1,ray)    
    if  any(t<=0) or (len(f)>0 and min(abs(x) for x in f) > 1.e0): return 0.,torch.zeros(len(xx0)),f,uu,t,g,p,0,0,0,0 

    f,t0,y,g,p=ratios(x0,shotx,recx,x,a2,b2,c2,d2,vv0,vv01,ray)
    gg=torch.zeros(len(f),len(x0))
    hh=torch.zeros(len(f),len(xx0))
    g0=torch.zeros(len(t),len(x0))
    h0=torch.zeros(len(t),len(xx0))
    xx0.retain_grad()
    x0.retain_grad()
   
    for i in range(len(f)):
        gg[i] = torch.autograd.grad(f[i], x0, retain_graph=True)[0]    
        np= torch.autograd.grad(f[i], xx0, retain_graph=True,allow_unused=True)[0]
        if np is None:
            hh[i]=torch.zeros(len(xx0))
        else:
            hh[i]=np
    for i in range(len(t)):    
        g0[i] = torch.autograd.grad(t0[i], x0, retain_graph=True)[0]
        np= torch.autograd.grad(t0[i], xx0, retain_graph=True,allow_unused=True)[0]    
        if np is None:
            h0[i]=torch.zeros(len(xx0))
        else:
            h0[i]=np
    
    gt=gg.transpose(0,1)@gg;II=torch.eye(len(x0));
    if len(gt)==0:
        damp=1.e-8;
    else: 
        damp=torch.max(abs(gt));
    dgdh=-torch.linalg.solve(gt+damp*1.e-8*II,gg.transpose(0,1)@hh)
    #gdh=-torch.linalg.solve(gg,hh)
    dtdh=torch.mm(g0,dgdh)+h0;
    t=t0.sum().detach()
    dtdh=dtdh.sum(0).detach()  
    xx=torch.cat((torch.tensor([shotx]),x0,torch.tensor([recx]))).tolist()
    yy=y.tolist()
    plt.scatter(shotx , yy[0])
    bady=False
    for i in range(len(ray)-1):
        xarc,yarc=drawarc([xx[i],yy[i]],[xx[i+1],yy[i+1]] ,g[i],p[i]) 
        yup=xarc*1;ydown=xarc*1;
        if ray[i+1]==ray[i]:
            down=ray[i];up=ray[i]-1;
            if up<0:up=0
        else:
            up=min(ray[i]-1,ray[i+1]-1);down=max(ray[i]-1,ray[i+1]-1);
        for j in range(len(xarc)):
            if up<0:
                yup[j]=1.e28
            else:
                yup[j],ll,kk=chasetriangle(x[up],a1[up],b1[up],c1[up],d1[up],xarc[j])
            ydown[j],ll,kk=chasetriangle(x[down],a1[down],b1[down],c1[down],d1[down],xarc[j])
        if any(yup[j]+0.02<yarc[j] or ydown[j]-0.02>yarc[j] for j in range(len(xarc))):
            if ray[i]!=-3 or ray[i+1]==2:
                indexes = [j for j in range(len(xarc)) if yup[j] +0.02 < yarc[j] or ydown[j] - 0.02 > yarc[j]]
                plt.plot(xarc,yup,linewidth=0.25, linestyle='--', color='red')
                plt.plot(xarc,ydown,linewidth=0.25, linestyle='--', color='red')
                plt.plot(xarc,yarc,linewidth=0.5, linestyle='--', color='red')            
            bady=True;break;
        plt.plot(xarc,yup,linewidth=0.25)
        plt.plot(xarc,ydown,linewidth=0.25)
        plt.plot(xarc,yarc,linewidth=0.5)
                
    if  bady:t=0

    return t,dtdh,f,uu,t0,g,p,gg,hh,g0,h0

def multi(xx,shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0):
    
    yy,vvp,vvs,vp1,vs1=updatexx(xx,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
    a1=[];b1=[];c1=[];d1=[]
    for n in range(len(x)):
        a0,b0,c0,d0=triangle(x[n],yy[n],0)   
        a1+=[a0];b1+=[b0];c1+=[c0];d1+=[d0]
        
    
    xx0=Variable(torch.tensor(xx),requires_grad=True)
    y0,vpp,vss,vp01,vs01=updatexx(xx0,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
    
    a2=[];b2=[];c2=[];d2=[]
    for n in range(len(x)):
        a0,b0,c0,d0=triangle(x[n],y0[n],0)   
        a2+=[a0];b2+=[b0];c2+=[c0];d2+=[d0]
    vv,vv1=velo(vvp,vvs,vp1,vs1,ray,pors)
    vv0,vv01=velo(vpp,vss,vp01,vs01,ray,pors)
    
    t=[];dtdh=[];
    for k in range(len(t0)):
        if t0[k]==0:
            t1=0;dtdh1=[0]*len(xx)
        else:
            t1,dtdh1,f,uu,tt,g,p0,gg,hh,g0,h0=raytrace(shotx,recx[k],ray,x,a1,b1,c1,d1,a2,b2,c2,d2,xx0,vv,vv1,vv0,vv01)
        
        t+=[t1];dtdh+=[dtdh1]
    
    return t,dtdh
    
def alltimes(xx,shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0,nproc):
    
    yy,vvp,vvs,vp1,vs1=updatexx(xx,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
    a1=[];b1=[];c1=[];d1=[]
    for n in range(len(x)):
        a0,b0,c0,d0=triangle(x[n],yy[n],0)   
        a1+=[a0];b1+=[b0];c1+=[c0];d1+=[d0]
    xx0=Variable(torch.tensor(xx),requires_grad=True)
    y0,vpp,vss,vp01,vs01=updatexx(xx0,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
    
    a2=[];b2=[];c2=[];d2=[]
    for n in range(len(x)):
        a0,b0,c0,d0=triangle(x[n],y0[n],0)   
        a2+=[a0];b2+=[b0];c2+=[c0];d2+=[d0]

    n=len(ray)*len(shotx)*len(recx)
    dt=torch.zeros(n)
    dtdh0=torch.zeros(n,len(xx))
    
    if __name__=="__main__" and nproc==1:
        fig, ax=plt.subplots()
        ax.set_aspect('equal')
    for i in range(len(ray)):
        
        vv,vv1=velo(vvp,vvs,vp1,vs1,ray[i],pors[i])
        vv0,vv01=velo(vpp,vss,vp01,vs01,ray[i],pors[i])
        print(ray[i])
        if __name__=="__main__" and nproc>1:
             p=Pool(nproc)
             result=[]
             for j in range(len(shotx)):
                 result.append(p.apply_async(multi,args=(xx,shotx[j],recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray[i],pors[i],t0[i][j],)))
             p.close()
             p.join()
        if __name__=="__main__" and nproc>1:
             t=[];dtdh=[]
             for j in range(len(shotx)):               
                    t1,dtdh1=result[j].get()
                    t+=[t1];dtdh+=[dtdh1]
        elif __name__=="__main__":
             t=[];dtdh=[]
             for j in range(len(shotx)):
                    t1=[];dtdh1=[]       
                    for k in range(len(recx)):

                        if t0[i][j][k]==0:t2=0;dtdh2=[0]*len(xx0) 
                        else:
                            t2,dtdh2,f,uu,tt,g,p0,gg,hh,g0,h0=raytrace(shotx[j],recx[k],ray[i],x,a1,b1,c1,d1,a2,b2,c2,d2,xx0,vv,vv1,vv0,vv01)
                        t1+=[t2];dtdh1+=[dtdh2]
                    
                    # t1,dtdh1=multi(xx,shotx[j],recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray[i],pors[i],t0[i][j])
                    t+=[t1];dtdh+=[dtdh1]
                    
        for j in range(len(shotx)):                        
            for k in range(len(recx)):
                m=i*len(shotx)*len(recx)+j*len(recx)+k                   
                if t[j][k]==0:dt[m]=0 
                else:dt[m]=t[j][k]-t0[i][j][k]
                dtdh0[m]=torch.Tensor(dtdh[j][k]);
                # if(abs(dt[m])>10):
                #     dt[m]=0
                #     dtdh0[m]=torch.zeros(len(xx));
                    
    if __name__=="__main__" and nproc==1:
        plt.draw()  
        plt.pause(0.1) 
        plt.savefig('last.ps')
                
                  
    return dt,dtdh0
                
torch.autograd.set_detect_anomaly(False)
torch.set_default_tensor_type(torch.DoubleTensor)

shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0=readtimes("good123.txt",0.0,1)
#writetomoxyz(x[1:],y[1:],150,150,vp[1:],vp0[1:],vs[1:],vs0[1:],[1000,2400,2900,3382][1:],[1000,2700,3000,3387][1:])
#writetomoxyz(x[1:],y[1:],50,50,vp[1:],vp0[1:],vs[1:],vs0[1:],[1000,2900,3382][1:],[1000,3000,3387][1:])
# for i in range(len(vp)):
#     if vp0[i]==vp[i]:vp0[i]+=vp0[i]*1.e-3
# for i in range(len(vs)):
#     if vs0[i]==vs[i]:vs0[i]+=vs0[i]*1.e-3
xx,ww=initxx(y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
nproc=1
if __name__=="__main__":
    
    II=torch.eye(len(xx));damp=1.e-8;#II=torch.diag(torch.Tensor(ww))
    f,g=alltimes(xx,shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0,nproc)
    writemodel(shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0,f.tolist(),"temp.txt")
    print(f.norm())
    while True:    
        gt=torch.transpose(g, 0, 1)
#        print(torch.mm(gt,g))
        dx=torch.linalg.solve(torch.mm(gt,g)+damp*II,torch.mv(gt,f))
        x1=(torch.tensor(xx)-dx).tolist()
        y0,vp00,vs00,vp001,vs001=updatexx(x1,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
        # if any(x > 10000 for x in vp00) or any(x > 10000 for x in vp001) or any(x > 10000 for x in vs00) or any(x > 10000 for x in vs001): 
        #     damp*=2.5
        #     continue
        
        # if vp001[1]-vp00[1]<200: 
        #     damp*=2.5
        #     continue
        
        # if any(min(y0[i])<max(y0[i+1]) for i in range(len(y0)-1)): 
        #     damp*=2.5
        #     continue 
        f0=f
        f,g=alltimes(x1,shotx,recx,vp,vpflag,vp0,vpflag1,vs,vsflag,vs0,vsflag1,x,y,flag,ray,pors,t0,nproc)    
        if torch.norm(f)<torch.norm(f0):# and all(element > 0 for element in x1):
            print(vp00,vp001,y0)
            damp/=1.2
            xx=x1            
        else:
            damp*=2.5
            print(f.norm())
            f=f0

        print(dx.norm(),f.norm())
        if dx.norm()<1.e-3:break
    y0,vp0,vs0,vp01,vs01=updatexx(xx,y,flag,vp,vs,vp0,vs0,vpflag,vsflag,vpflag1,vsflag1)
    writemodel(shotx,recx,vp0,vpflag,vp01,vpflag1,vs0,vsflag,vs01,vsflag1,x,y0,flag,ray,pors,t0,f.tolist(),"temp.txt")

 

