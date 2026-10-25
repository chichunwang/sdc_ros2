"""Small pose/solver arithmetic on CUDA; host transfers are output/control only."""
import cupy as cp


def create_rng(seed):
    """Load the pinned NVIDIA wheel explicitly, including existing containers.

    The current container may predate the Compose LD_LIBRARY_PATH addition.
    Missing dependency/load errors propagate; this never falls back to CPU.
    """
    import ctypes
    from importlib.metadata import distribution
    library=distribution('nvidia-curand-cu12').locate_file('nvidia/curand/lib/libcurand.so.10')
    ctypes.CDLL(str(library),mode=ctypes.RTLD_GLOBAL)
    return cp.random.RandomState(seed)


def rotation_exp(v):
    """Batched SO(3) exponential; small-angle limits stay finite on CUDA."""
    v=cp.asarray(v,dtype=cp.float64)
    theta2=cp.sum(v*v,axis=-1)
    theta=cp.sqrt(theta2)
    a=cp.sinc(theta/cp.pi)
    b=0.5*cp.sinc(theta/(2*cp.pi))**2
    W=cp.zeros(v.shape[:-1]+(3,3),dtype=cp.float64)
    W[...,0,1]=-v[...,2];W[...,0,2]=v[...,1]
    W[...,1,0]=v[...,2];W[...,1,2]=-v[...,0]
    W[...,2,0]=-v[...,1];W[...,2,1]=v[...,0]
    return cp.eye(3)+a[...,None,None]*W+b[...,None,None]*(W@W)



def apply_delta(T,scale,delta):
    out=cp.asarray(T).copy()
    out[:3,3]+=delta[:3]
    out[:3,:3]=rotation_exp(delta[3:6])@out[:3,:3]
    return cp.asnumpy(out),float(cp.asarray(scale)*cp.exp(delta[6]))
