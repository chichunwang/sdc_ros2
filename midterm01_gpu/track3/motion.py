"""Exact planar constant-turn integration with world-frame velocity state."""
import numpy as np
from .geometry import wrap


def turn_motion(state, gyro_rate, dt, gyro_available=True):
    x=np.asarray(state,dtype=float); y=x.copy(); F=np.eye(8); G=np.zeros(8)
    F[:3,3:6]=np.eye(3)*dt
    if not gyro_available:
        y[:3]+=x[3:6]*dt; G[6]=1.
        return y,F,G
    angle=(gyro_rate-x[7])*dt
    if abs(angle)<1e-4:
        sinc=1-angle**2/6+angle**4/120
        cosc=angle/2-angle**3/24+angle**5/720
        ds=-angle/3+angle**3/30-angle**5/840
        dc=.5-angle**2/8+angle**4/144
    else:
        sinc=np.sin(angle)/angle; cosc=(1-np.cos(angle))/angle
        ds=(angle*np.cos(angle)-np.sin(angle))/angle**2
        dc=(angle*np.sin(angle)-1+np.cos(angle))/angle**2
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    B=dt*np.array([[sinc,-cosc],[cosc,sinc]])
    dB=dt**2*np.array([[ds,-dc],[dc,ds]])
    dR=dt*np.array([[0.,-1.],[1.,0.]])@R
    v=x[3:5]
    y[:2]+=B@v; y[2]+=x[5]*dt; y[3:5]=R@v; y[6]=wrap(x[6]+angle)
    F[:2,3:5]=B; F[3:5,3:5]=R
    F[:2,7]=-dB@v; F[3:5,7]=-dR@v; F[6,7]=-dt
    G[:2]=dB@v/dt; G[3:5]=dR@v/dt; G[6]=1.
    return y,F,G
