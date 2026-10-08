"""Post-pilot diagnostics for unresolved approximation and informativeness issues."""
from experiments import *

def audit():
    fam=make_family(); out=[]; quadrature=[]
    for n in (64,256):
        for m in (0,4,16,64):
            for rep in (0,1):
                seed=SEED+100000+n*1000+m*30+rep; rng=np.random.default_rng(seed)
                truth=rng.uniform(-1,1,3); Y=sample_data(fam.Q(truth),n,m,rng); like=MaskLikelihood(Y)
                anchors=np.array(list(product([-1.,0.,1.],repeat=3)))
                threshold=logsumexp([like(fam.Q(x)) for x in anchors])-np.log(27)+np.log(.05)
                grid=np.array(list(product(np.linspace(-1,1,9),repeat=3)))
                ll=np.array([like(fam.Q(x)) for x in grid]); x,diag=affine_mle(fam,like)
                q5=model_average(fam,like,5); q7=model_average(fam,like,7)
                w5=solve_qp(q5,constraints={'upper':.2})['w']; w7=solve_qp(q7,constraints={'upper':.2})['w']
                out.append(dict(n=n,m=m,rep=rep,grid_points=len(grid),grid_retained_fraction=float(np.mean(ll>=threshold)),
                    grid_rejected=int(np.sum(ll<threshold)),mle_minus_fine_grid=diag['loglik']-float(ll.max()),
                    quadrature_weight_distance=float(np.linalg.norm(w5-w7)),
                    quadrature_objective_difference=float(.5*w5@fam.Q(truth)@w5-.5*w7@fam.Q(truth)@w7)))
    write_json('post_pilot_audit.json',{'scope':'16 fixed S2 instances, reps 0 and 1 per cell; diagnostic only; no retuning',
        'results':out,'note':'Grid retention is not a volume certificate or a valid replacement for an outer cover.'})
    print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__': audit()
