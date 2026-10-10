"""Supplementary equal-error-budget safe reference, declared before E4 outcome inspection."""
import json,time
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from .common import OUT,CFG,write,append,rows,timed_stage
from .compat_models import Family
from .baselines import evaluate

def select(fam,baseline,candidates,pilot,gate,alpha=.005):
    logp=pilot-logsumexp(pilot);logq=logsumexp(logp+gate)
    retained=np.flatnonzero(gate>=logq+np.log(alpha))
    approved=[]
    for c,w in enumerate(candidates):
        difference=fam.costs(w)-fam.costs(baseline);pad=1e-11*(1+np.max(abs(fam.costs(w))))
        if np.all(difference[retained]<=-pad):approved.append(c)
    W=np.vstack([baseline,candidates[approved]]);p=fam.posterior(pilot+gate);Q=np.einsum('m,mij->ij',p,fam.Q)
    if len(W)==1:a=np.ones(1);gap=0.
    else:
        H=W@Q@W.T;scale=max(float(np.max(abs(H))),1e-8)
        fit=minimize(lambda x:(.5*x@H@x/scale,H@x/scale),np.ones(len(W))/len(W),jac=True,method='SLSQP',
          bounds=[(0,1)]*len(W),constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(len(W))},
          options={'ftol':1e-13,'maxiter':200})
        a=np.maximum(fit.x,0);a/=a.sum();g=H@a;gap=max(0,float(g@a-g.min()))
    return dict(w=a@W,approved=approved,retained=retained,optimization_gap=gap,convex_weights=a,logq=float(logq),alpha=alpha)

def run(budget):
    conf=json.loads((CFG/'protocol.json').read_text());counts={}
    for split in ('development','confirmation'):
        path=OUT/f'E4_{split}_safe_reference.jsonl';done={r['dataset'] for r in rows(path)};families={};count=0
        for evidence in rows(OUT/f'E4_{split}_evidence.jsonl'):
            budget.check();key=evidence['dataset']
            if key in done:continue
            name=evidence['family']
            if name not in families:
                spec=json.loads((OUT/'geometry'/f'{name}.json').read_text());families[name]=Family(np.array(spec['Q']),upper=.2)
            f=families[name];baseline=np.array(evidence['baseline']);candidates=np.array(evidence['candidate_weights']).reshape(-1,f.d)
            pilot=np.array(evidence['pilot_loglik']);gate=np.array(evidence['gate_loglik']);truth=evidence['truth']
            a=select(f,baseline,candidates,pilot,gate);change=float(f.costs(a['w'])[truth]-f.costs(baseline)[truth]);pad=1e-11*(1+abs(f.costs(baseline)[truth]))
            if np.any((f.costs(a['w'])-f.costs(baseline))[a['retained']]>1e-10):raise RuntimeError('retained-model safety check failed')
            k={field:evidence[field] for field in ['dataset','experiment','split','family','d','M','nats','joint_rows','pilot_joint_rows','actual_information','rep','truth','epsilon','epsilon_ratio']}
            harmful=any(f.costs(candidates[j])[truth]-f.costs(baseline)[truth]>pad for j in a['approved'])
            append(path,{**k,'method':'CS-Safe',**evaluate(f,a['w'],f.posterior(pilot+gate),evidence['epsilon'],truth),
              'true_change_from_pilot':change,'harmful_change':bool(change>pad),'beneficial_change':bool(change < -pad),
              'harmful_approval':bool(harmful),'approved_count':len(a['approved']),'candidate_count':len(candidates),
              'target_error':.005,'retained_models':a['retained'],'optimization_gap':a['optimization_gap'],'logq':a['logq']})
            count+=1
        counts[split]=count
    write(OUT/'safety_reference_manifest.json',{'additional_rows':counts,'alpha':.005,
      'status':'supplementary comparator; original methods/configurations unchanged; not a retuned candidate'})
    print(f'Equal-budget safety reference complete: {counts}',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
