from .risk_envelopes import RiskSpec


def all_specs():
    a=[RiskSpec('A-Mean','mean',rho=1),RiskSpec('B-Mean','mean',rho=1,conditional=True),
       RiskSpec('A-Max','max'),RiskSpec('B-Max','max',conditional=True)]
    for rho in (.25,.5):
        a += [RiskSpec(f'A-ATK:{rho}','count',rho=rho),RiskSpec(f'B-ATK:{rho}','count',rho=rho,conditional=True)]
        for name,cond,ref,absolute in [('A-Tail-P',False,False,False),('A-Tail-U',False,True,False),
            ('B-Tail-P',True,False,False),('B-Tail-U',True,True,False),('A-Absolute',False,False,True)]:
            a.append(RiskSpec(f'{name}:{rho}',rho=rho,conditional=cond,reference=ref,absolute=absolute))
    for value in (.01,.1,1.):
        a += [RiskSpec(f'C-LP-Max:{value}','max',penalty=value),
              RiskSpec(f'C-LP-Tail:{value}',reference=True,penalty=value),
              RiskSpec(f'C-Soft:{value}','soft',temperature=value),
              RiskSpec(f'C-Soft-Cond:{value}','soft',conditional=True,temperature=value),
              RiskSpec(f'C-LR:{value}','lr',reference=True,radius=value)]
    for c in (.25,.5,1.): a.append(RiskSpec(f'C-Budget:{c}','budget',reference=True,budget_fraction=c))
    for name,kind in [('RankThenP','naive'),('pR','pr'),('FilterThenK','filter'),('PostSampleK','sample')]:
        a.append(RiskSpec('H-'+name,kind))
        a.append(RiskSpec('H-Conditional-'+name,kind,conditional=True))
    return a


def core_specs():
    names=['A-Mean','A-Max','B-Max','A-Tail-P:0.25','A-Tail-P:0.5','B-Tail-P:0.25','B-Tail-P:0.5',
           'A-Tail-U:0.5','B-Tail-U:0.5']
    lookup={s.name:s for s in all_specs()}
    return [lookup[n] for n in names]


def blend_specs():
    return [RiskSpec('A-Blend',blend=.5),RiskSpec('B-Blend',blend=.5,conditional=True)]
