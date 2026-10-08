"""Independent continuous-posterior check on fixed high-information boundary cases."""
import json,time
import numpy as np
from core import MaskLikelihood,solve_qp
from .experiment_v2 import OUT,CFG,write_json,append_row
from .new_generators import family,truth_points,data_instance,importance_scenarios,seed_for
from .scenario_inference import posterior

def chain(like,fam,start,scale,seed):
    rng=np.random.default_rng(seed);x=start.copy();ll=like(fam.Q(x));draw=[];accepted=0;block=0;factor=1.
    for i in range(12000):
        z=x+rng.normal(size=len(x))*scale*factor
        z=1-np.abs((z+1)%4-2);new=like(fam.Q(z))
        if np.log(rng.uniform())<new-ll:x=z;ll=new;accepted+=1;block+=1
        if i<4000 and (i+1)%100==0:
            factor*=np.exp(.4*(block/100-.3));factor=np.clip(factor,.05,4);block=0
        if i>=4000 and i%4==0:draw.append(x.copy())
    return np.array(draw),accepted/12000

def diagnostics(draws):
    split=np.concatenate([draws[:,:draws.shape[1]//2],draws[:,draws.shape[1]//2:]],axis=0)
    n=split.shape[1];W=np.var(split,axis=1,ddof=1).mean(0);B=n*np.var(split.mean(1),axis=0,ddof=1)
    rhat=np.sqrt(((n-1)*W/n+B/n)/W);ess=[]
    for j in range(draws.shape[2]):
        cor=[]
        for x in draws[:,:,j]:
            x=x-x.mean();n0=len(x);f=np.fft.rfft(x,n=2*n0);a=np.fft.irfft(abs(f)**2)[:n0]
            cor.append(a/a[0])
        ac=np.mean(cor,axis=0);pairs=ac[1:-1:2]+ac[2::2];pos=[]
        for v in pairs:
            if v<0:break
            pos.append(v if not pos else min(pos[-1],v))
        ess.append(draws.shape[0]*draws.shape[1]/max(1,1+2*sum(pos)))
    return rhat,np.array(ess)

def run():
    start=time.perf_counter();cpu=time.process_time();records=[]
    manifest=json.loads((CFG/'G2_manifest.json').read_text())
    chosen=[r for r in manifest if r['family_id'] in (0,1,2) and r['point_id']==1 and r['rep']==0 and r['n']==256 and r['m']==64]
    write_json(CFG/'posterior_audit_manifest.json',chosen)
    for r in chosen:
        fam=family('G2',r['family_seed']);theta=truth_points(r['point_seed'])[1]
        Y=data_instance(fam,theta,r['n'],r['m'],r['data_seed']);like=MaskLikelihood(Y)
        sc=importance_scenarios(fam,Y,1024,seed_for('posterior-audit',r['family_seed']));p=posterior(sc,Y)
        proposal=sc.proposal_diagnostics;mode=np.r_[proposal['outer_mean'],proposal['inner_fits'][len(proposal['inner_fits'])//2]['eta_mode']]
        scale=np.r_[proposal['outer_sd'],proposal['inner_fits'][len(proposal['inner_fits'])//2]['eta_scale']]
        chains=[];rates=[]
        for k,initial in enumerate((np.zeros(3),mode)):
            d,a=chain(like,fam,initial,scale,seed_for('mcmc',r['family_seed'],k));chains.append(d);rates.append(a)
        draws=np.array(chains);rh,ess=diagnostics(draws);flat=draws.reshape(-1,3)
        qmean=np.mean([fam.Q(x) for x in flat],axis=0);wref=solve_qp(qmean,constraints={'upper':.2})['w']
        details=[]
        for size in (256,512,1024):
            for scramble in range(3):
                ss=importance_scenarios(fam,Y,size,seed_for('mcmc-comparison',r['family_seed'],scramble));pp=posterior(ss,Y)
                q=np.einsum('m,mij->ij',pp.p,ss.Q);w=solve_qp(q,constraints={'upper':.2})['w']
                details.append({'size':size,'scramble':scramble,'posterior_mean':pp.p@ss.nodes,
                    'weight_distance':np.linalg.norm(w-wref),'mc_reference_excess':.5*(w@qmean@w-wref@qmean@wref),
                    'reference_integral':ss.proposal_diagnostics['reference_integral_estimate']})
        np.savez_compressed(OUT/f"posterior_chains_{r['instance_id']}.npz",draws=draws)
        records.append(r|{'posterior_mean':flat.mean(0),'rhat':rh,'ess':ess,'acceptance':rates,
            'passed_chain_diagnostic':bool(np.max(rh)<1.05 and np.min(ess)>100),'comparison':details,
            'target':'uniform continuous prior times full-mask likelihood; no true parameter in sampler',
            'scope':'classical split Rhat and autocorrelation ESS; two chains are diagnostics, not proof of global mixing'})
        print('posterior audit',r['family_id'],'Rhat',rh,'ESS',ess,flush=True)
    write_json(OUT/'posterior_audit.json',records)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'posterior_audit','wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})

if __name__=='__main__':run()
