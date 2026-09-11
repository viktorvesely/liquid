"""Reproduce scaling analysis from the four notebook-selected launches; no training.
Run from repository root: MPLCONFIGDIR=/tmp/mpl .venv/bin/python liquid_jax/scaling_analysis/analyze.py
"""
from pathlib import Path
import os,json,itertools,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/liquid-scaling-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent/'runs'
NAMES=['exp_scaling_742fde19a2_Cifar10_20260907_145449','exp_scaling_4f767e649d_Bikes_20260907_172915','exp_scaling_1b88e205dd_Svhn_20260907_182603','exp_scaling_2f7815bfc8_Energy_20260907_214458']
TASKS=['Cifar10','Svhn','Bikes','Energy']; LABELS=['CIFAR-10','SVHN','Bikes','Energy']
KEY=['task','predictors','pwidth','dwidth']; LS=[0,1,2,4,8,16,32]

def extract():
 rows=[]; curves=[]; manifest=[]; audits=[]
 for name in NAMES:
  folder=RUNS/name; task=name.split('_')[3]
  files=sorted(folder.glob('*_metrics.json')); assert len(files)==315
  for fi,path in enumerate(files):
   parts=path.stem.removesuffix('_metrics').split('_'); info=dict(zip(parts[::2],parts[1::2])); info={k:int(v) if v.isdigit() else v for k,v in info.items()}; info['task']=task
   assert info['mixing']=='sum' and info['ambiguity']=='both'
   raw=path.read_bytes(); d={k:np.asarray(v,dtype=float) for k,v in json.loads(raw).items()}
   shape=d['validation_performance_loss'].shape
   assert shape==((100 if task in TASKS[:2] else 2000),5)
   assert all(v.shape==shape and np.isfinite(v).all() for v in d.values())
   manifest.append(dict(file=str(path.relative_to(RUNS)),sha256=hashlib.sha256(raw).hexdigest()))
   zpath=path.with_name(path.name.replace('_metrics.json','_eval_metrics.npz'))
   # Hash archives without retaining their bytes.
   h=hashlib.sha256()
   with open(zpath,'rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
   manifest.append(dict(file=str(zpath.relative_to(RUNS)),sha256=h.hexdigest()))
   vals={}
   with np.load(zpath) as z:
    if info['delegators']:
     for k in ['loss','loss_under_oracle','delegator_regret_loss','metric','metric_under_oracle','from_epoch']: vals[k]=z[k].astype(float)
     w=z['predictors_weight_per_model'].astype(float); p=z['predictors_perfomance_per_model'].astype(float); a=z['predictors_ambiguity_per_model'].astype(float)
     vals['P']=(w*p).sum(-1).mean(-1); vals['A']=(w*a).sum(-1).mean(-1)
     vals['P_unweighted']=p.mean((1,2)); vals['A_unweighted']=a.mean((1,2))
     vals['effective_predictors']=(1/(w*w).sum(-1)).mean(-1)
     vals['C']=z['delegators_perfomance_per_model'].astype(float).mean((1,2))
     vals['D']=z['delegators_ambiguity_per_model'].astype(float).mean((1,2))
     vals['Q']=vals['C']-vals['D']
     vals['reference_loss']=np.minimum(vals['loss'],vals['loss_under_oracle'])
     vals['decomp_residual']=vals['P']-vals['A']-vals['reference_loss']
     vals['positive_gap']=vals['loss']-vals['reference_loss']
     assert all(np.isfinite(v).all() for v in vals.values())
     vals['json_npz_gap']=vals['loss']-d['validation_performance_loss'][-1]
     audits.append(dict(task=task,run=info['run'],residual_max=float(np.max(np.abs(vals['decomp_residual']))),weight_sum_error=float(np.max(np.abs(w.sum(-1)-1)))))
    else: assert not z.files
   metric='accuracy_metric' if task in TASKS[:2] else 'r2_metric'
   for s in range(5):
    row=info|dict(seed_id=s,final_loss=d['validation_performance_loss'][-1,s],train_loss=d['performance_loss'][-1,s],final_metric=d['validation_'+metric][-1,s],tail_loss=d['validation_performance_loss'][-max(1,shape[0]//10):,s].mean(),best_loss=d['validation_performance_loss'][:,s].min(),balance=d['validation_load_balancing_loss'][-1,s])
    row.update({k:float(v[s]) for k,v in vals.items()}); rows.append(row)
   # All epochs are retained after averaging seeds, as in notebook.
   for e in range(shape[0]): curves.append(info|dict(epoch=e+1,loss=d['validation_performance_loss'][e].mean(),train_loss=d['performance_loss'][e].mean()))
  print('Loaded',task,flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'seed_metrics.csv',index=False)
 pd.DataFrame(curves).to_csv(ROOT/'learning_curves.csv.gz',index=False)
 pd.DataFrame(audits).to_csv(ROOT/'decomposition_audit.csv',index=False)
 (ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))

import sys
if '--rebuild' in sys.argv or not (ROOT/'seed_metrics.csv').exists(): extract()
df=pd.read_csv(ROOT/'seed_metrics.csv')
assert len(df)==6300
assert not df.duplicated(KEY+['delegators','seed_id']).any()
means=df.groupby(KEY+['delegators'],as_index=False).mean(numeric_only=True)
means.to_csv(ROOT/'condition_means.csv',index=False)
# Seed IDs share keys throughout the factorial grid; bootstrap seed blocks jointly.
COUNTS=np.array([np.bincount(x,minlength=5) for x in itertools.product(range(5),repeat=5)])/5

def effects(data, metric='final_loss', baseline=1):
 b=data[data.delegators==baseline][KEY+['seed_id',metric]].rename(columns={metric:'baseline'})
 return data.merge(b,on=KEY+['seed_id'],validate='many_to_one')

def summarize(g,metric='final_loss'):
 a=g.pivot(index=KEY,columns='seed_id',values=metric).to_numpy()
 b=g.pivot(index=KEY,columns='seed_id',values='baseline').to_numpy()
 point=(100*(1-a.mean(1)/b.mean(1))).mean()
 boot=(100*(1-(a@COUNTS.T)/(b@COUNTS.T))).mean(0)
 lo,hi=np.quantile(boot,[.025,.975])
 gains=100*(1-a.mean(1)/b.mean(1))
 return dict(gain=point,lo=lo,hi=hi,wins=int((gains>0).sum()),n=len(gains),median=np.median(gains),q25=np.quantile(gains,.25),q75=np.quantile(gains,.75))

# Every relative effect retains L=1 as baseline. Direct loss is missing at L=0.
logged=effects(df)
direct=effects(df[df.delegators>0],'loss')
logged_summ=pd.DataFrame([dict(task=t,L=l,**summarize(g)) for (t,l),g in logged.groupby(['task','delegators'])])
summ=pd.DataFrame([dict(task=t,L=l,**summarize(g,'loss')) for (t,l),g in direct.groupby(['task','delegators'])])
logged_summ.to_csv(ROOT/'logged_scaling_summary.csv',index=False);summ.to_csv(ROOT/'scaling_summary.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
COL=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
figs=ROOT/'figures';figs.mkdir(exist_ok=True)
def save(fig,name):
 fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');fig.savefig(figs/(name+'.png'),bbox_inches='tight');plt.close(fig)
def setup(ax):ax.axhline(0,color='0.55',lw=.7,zorder=0);ax.grid(axis='y',alpha=.15)
def xaxis(ax,zero=True):
 ax.set_xticks(range(7) if zero else range(1,7))
 ax.set_xticklabels(LS if zero else LS[1:])
 ax.tick_params(axis='x',labelsize=7)
 ax.set_xlim(-.25 if zero else .75,6.25)
 ax.set_xlabel('Delegators $L$')
# Main comparison: uniform routing is explicitly displayed, without relabeling its surrogate as CE.
fig,axs=plt.subplots(1,4,figsize=(7.2,2.7),sharey=True)
for ax,t,title in zip(axs,TASKS,LABELS):
 g=logged_summ[logged_summ.task==t].set_index('L').loc[LS];setup(ax)
 ax.plot(range(7),g.gain,'o-',color=COL[0],ms=3,label='Logged loss (all $L$)')
 ax.fill_between(range(7),g.lo,g.hi,color=COL[0],alpha=.16)
 ax.set_title(title);xaxis(ax)
axs[0].set_ylabel('Validation loss reduction (%)');fig.tight_layout();save(fig,'01_scaling')
# Full scaling curves for every factor: the same comparison is used for all three rows.
factors=['pwidth','predictors','dwidth'];flabels=['Predictor width $b_M$','Predictor count $M$','Delegator width $b_L$']
conditional=[]
for endpoint,data in [('logged',logged),('direct',direct)]:
 for factor in factors:
  for (t,l,v),g in data.groupby(['task','delegators',factor]):
   conditional.append(dict(endpoint=endpoint,task=t,L=l,factor=factor,value=v,**summarize(g,'final_loss' if endpoint=='logged' else 'loss')))
cond=pd.DataFrame(conditional);cond.to_csv(ROOT/'capacity_scaling.csv',index=False)
fig,axs=plt.subplots(3,4,figsize=(7.2,7.1),sharey=True)
for i,(factor,label) in enumerate(zip(factors,flabels)):
 for j,(t,title) in enumerate(zip(TASKS,LABELS)):
  ax=axs[i,j];setup(ax)
  for color,v in zip(COL,sorted(df[factor].unique())):
   g=cond[(cond.endpoint=='logged')&(cond.task==t)&(cond.factor==factor)&(cond.value==v)].set_index('L').loc[LS]
   ax.plot(range(7),g.gain,'o-',color=color,ms=2,lw=1,label=str(v))
  xaxis(ax)
  if i==0:ax.set_title(title)
  if j==0:ax.set_ylabel(label+'\nLoss reduction (%)')
  if j==3:ax.legend(title=label,fontsize=6,title_fontsize=6,loc='lower left',ncol=2)
fig.tight_layout();save(fig,'02_capacity_scaling')
# Balanced functional ANOVA: exact descriptive partition of capacity variation at each L.
base=means[means.delegators==1].set_index(KEY)
allcells=means.merge(base[['loss','final_loss']].reset_index(),on=KEY,suffixes=('','_1'))
allcells['gain']=100*(1-allcells.loss/allcells.loss_1)
allcells['logged_gain']=100*(1-allcells.final_loss/allcells.final_loss_1)
allcells.to_csv(ROOT/'all_condition_contrasts.csv',index=False)
terms=[]
for endpoint,column in [('direct','gain'),('logged','logged_gain')]:
 for (t,l),g in allcells[allcells.delegators>1].groupby(['task','delegators']):
  components={():np.repeat(g[column].mean(),len(g))};total=((g[column]-g[column].mean())**2).sum()
  for k in range(1,4):
   for fs in itertools.combinations(factors,k):
    a=g.groupby(list(fs))[column].transform('mean').to_numpy()
    for sub,v in components.items():
     if set(sub)<set(fs):a=a-v
    components[fs]=a
    terms.append(dict(endpoint=endpoint,task=t,L=l,term=':'.join(fs),ss=float((a*a).sum()),total=total))
  assert np.allclose(sum((v*v).sum() for k,v in components.items() if k),total)
terms=pd.DataFrame(terms);terms.to_csv(ROOT/'factorial_variation_by_L.csv',index=False)
importance=terms.groupby(['endpoint','task','term'],as_index=False)[['ss','total']].sum();importance['share']=100*importance.ss/importance.total
importance.to_csv(ROOT/'factorial_importance.csv',index=False)
fig,axs=plt.subplots(1,4,figsize=(7.2,2.9),sharey=True)
for ax,t,title in zip(axs,TASKS,LABELS):
 g=importance[(importance.endpoint=='direct')&(importance.task==t)].set_index('term')
 vals=[g.loc[f,'share'] for f in factors]+[g[g.index.str.contains(':')]['share'].sum()]
 ax.bar(range(4),vals,color=COL[:3]+['.6']);ax.set(title=title,xticks=range(4),xticklabels=['$b_M$','$M$','$b_L$','Interactions'],ylim=(0,65))
 ax.tick_params(axis='x',rotation=45)
 for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.0f}%',ha='center',fontsize=7)
axs[0].set_ylabel('Share of capacity variation (%)');fig.tight_layout();save(fig,'06_factor_importance')
# Predictor AND delegator performance-ambiguity decompositions in the main analysis.
# Each component is normalized by its own family's L=1 individual loss, avoiding
# comparisons of raw CE-to-oracle nats with predictive MSE/CE.
components=[];fault=[]
for t in TASKS:
 b=base.loc[[t]]
 for l in LS[1:]:
  z=means[(means.task==t)&(means.delegators==l)].set_index(KEY).loc[b.index]
  components.append(dict(task=t,L=l,p_error=((z.P-b.P)/b.P*100).mean(),p_ambiguity=((z.A-b.A)/b.P*100).mean(),p_net=(((z.P-z.A)-(b.P-b.A))/b.P*100).mean(),d_error=((z.C-b.C)/b.C*100).mean(),d_ambiguity=((z.D-b.D)/b.C*100).mean(),d_net=((z.Q-b.Q)/b.C*100).mean()))
  fault.append(dict(task=t,L=l,total_change=((z.loss-b.loss)/b.loss*100).mean(),predictor_change=((z.loss_under_oracle-b.loss_under_oracle)/b.loss*100).mean(),delegator_change=((z.delegator_regret_loss-b.delegator_regret_loss)/b.loss*100).mean(),selected_predictor_change=((z.reference_loss-b.reference_loss)/b.loss*100).mean(),selected_delegator_change=((z.positive_gap-b.positive_gap)/b.loss*100).mean(),total_level=(z.loss/b.loss*100).mean(),predictor_level=(z.loss_under_oracle/b.loss*100).mean(),delegator_level=(z.delegator_regret_loss/b.loss*100).mean()))
components=pd.DataFrame(components);components.to_csv(ROOT/'both_ambiguity_decompositions.csv',index=False)
means.assign(predictor_difference=means.P-means.A,delegator_difference=means.C-means.D)[KEY+['delegators','P','A','predictor_difference','C','D','delegator_difference']].to_csv(ROOT/'raw_decomposition_means.csv',index=False)
fault=pd.DataFrame(fault);fault.to_csv(ROOT/'fault_attribution.csv',index=False)
assert np.allclose(fault.predictor_change+fault.delegator_change,fault.total_change,atol=1e-5)
fig,axs=plt.subplots(2,4,figsize=(7.2,5.5))
for j,(t,title) in enumerate(zip(TASKS,LABELS)):
 g=components[components.task==t]
 for i,prefix in enumerate(['p','d']):
  ax=axs[i,j];setup(ax)
  for col,suffix,style,label in zip(COL,['error','ambiguity','net'],['o-','s--','^-'],['Δ individual loss','Δ ambiguity','Δ(loss − ambiguity)']):ax.plot(range(1,7),g[prefix+'_'+suffix],style,color=col,ms=2.5,label=label)
  xaxis(ax,zero=False)
  if i==0:ax.set_xlabel('')
  if i==0:ax.set_title(title)
  if j==0:ax.set_ylabel(('Predictors' if i==0 else 'Delegators')+'\nChange from $L=1$ (%)')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'03_loss_accounting')
# The main attribution figure shows only the changes needed to explain scaling.
fig,axs=plt.subplots(1,4,figsize=(7.2,2.8),sharey=True)
for ax,t,title in zip(axs,TASKS,LABELS):
 g=fault[fault.task==t];setup(ax)
 for color,key,style,label in zip(COL,['predictor','delegator','total'],['o-','s-','--'],['Predictor/reference','Delegation','Total']):
  ax.plot(range(1,7),-g[key+'_change'],style,color=color if key!='total' else 'black',ms=3,label=label)
 xaxis(ax,zero=False);ax.set_title(title)
axs[0].set_ylabel('Contribution to loss reduction (%)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=7,frameon=False)
fig.tight_layout(rect=(0,0,1,.90));save(fig,'04_fault_attribution')
# Paired comparisons of failure onset / late saturation, each still normalized by L=1.
contrasts=[]
for t in TASKS:
 g=fault[fault.task==t].set_index('L')
 for lo,hi in [(1,2),(1,4),(1,32),(8,32)]:
  contrasts.append(dict(task=t,from_L=lo,to_L=hi,**{k:g.loc[hi,k]-g.loc[lo,k] for k in ['total_change','predictor_change','delegator_change','selected_predictor_change','selected_delegator_change']}))
pd.DataFrame(contrasts).to_csv(ROOT/'failure_contrasts.csv',index=False)
# All-factor attribution: connect capacity response to predictor/reference and delegation terms.
conditional_fault=[]
for factor in factors:
 for (t,l,v),g in allcells[allcells.delegators>0].groupby(['task','delegators',factor]):
  b=base.loc[g.set_index(KEY).index].reset_index();g=g.reset_index(drop=True)
  conditional_fault.append(dict(task=t,L=l,factor=factor,value=v,total_change=((g.loss-b.loss)/b.loss*100).mean(),predictor_change=((g.loss_under_oracle-b.loss_under_oracle)/b.loss*100).mean(),delegator_change=((g.delegator_regret_loss-b.delegator_regret_loss)/b.loss*100).mean()))
pd.DataFrame(conditional_fault).to_csv(ROOT/'conditional_fault_attribution.csv',index=False)
# Native performance and endpoint sensitivity include L=0; baseline always L=1.
rob=[]
for metric in ['final_loss','tail_loss','best_loss']:
 for (t,l),g in effects(df,metric).groupby(['task','delegators']):rob.append(dict(task=t,L=l,endpoint=metric,**summarize(g,metric)))
rob=pd.DataFrame(rob);rob.to_csv(ROOT/'endpoint_sensitivity.csv',index=False)
fig,axs=plt.subplots(2,4,figsize=(7.2,5.4))
for j,(t,title) in enumerate(zip(TASKS,LABELS)):
 z=means[means.task==t].groupby('delegators').mean(numeric_only=True).loc[LS]
 axs[0,j].plot(range(7),z.final_metric,'o-',color=COL[0],ms=3)
 axs[0,j].set(title=title,ylabel='Accuracy' if j<2 else '$R^2$');xaxis(axs[0,j])
 for c,metric,label in zip(COL,['final_loss','tail_loss','best_loss'],['Final epoch','Last 10%','Min. (optimistic)']):
  g=rob[(rob.task==t)&(rob.endpoint==metric)].set_index('L').loc[LS]
  axs[1,j].plot(range(7),g.gain,'o-',color=c,label=label,ms=3)
 setup(axs[1,j]);xaxis(axs[1,j])
axs[1,0].set_ylabel('Loss reduction vs. $L=1$ (%)');axs[1,0].legend(fontsize=6,loc='lower right');fig.tight_layout();save(fig,'05_endpoint_sensitivity')
curves=pd.read_csv(ROOT/'learning_curves.csv.gz')
fig,axs=plt.subplots(1,4,figsize=(7.2,2.9))
for ax,t,title in zip(axs,TASKS,LABELS):
 for color,l in zip(['.45']+COL,[0,1,4,32]):
  g=curves[(curves.task==t)&(curves.delegators==l)].groupby('epoch')[['loss','train_loss']].mean()
  g=g.rolling(max(1,len(g)//100),min_periods=1).mean();g=g[g.index>g.index.max()*.1]
  ax.plot(g.index,g.loss,color=color,label=f'$L={l}$');ax.plot(g.index,g.train_loss,color=color,ls='--',alpha=.8)
 ax.set(title=title,xlabel='Epoch')
 if t=='Cifar10':ax.set_ylabel('Logged performance loss');ax.legend(fontsize=6)
fig.tight_layout();save(fig,'07_learning_curves')
# Preserve the complete 45-cell endpoint map as a supplementary table/figure.
cell=allcells[allcells.delegators==32];cell.to_csv(ROOT/'endpoint_contrasts.csv',index=False)
fig,axs=plt.subplots(1,4,figsize=(7.2,4.8),sharey=True);lim=np.ceil(cell.gain.abs().max()/5)*5
for ax,t,title in zip(axs,TASKS,LABELS):
 g=cell[cell.task==t].set_index(['predictors','pwidth','dwidth'])
 grid=np.array([[g.loc[(m,p,d),'gain'] for d in [4,8,16]] for m in [2,4,8,16,32] for p in [4,8,16]])
 im=ax.imshow(grid,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
 for i in range(15):
  for j in range(3):ax.text(j,i,'0' if abs(grid[i,j])<.5 else f'{grid[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if abs(grid[i,j])>lim*.55 else 'black')
 for y in [2.5,5.5,8.5,11.5]:ax.axhline(y,color='white',lw=1.5)
 ax.set(title=title,xticks=range(3),xticklabels=[4,8,16],xlabel='Delegator width $b_L$',yticks=range(15),yticklabels=[f'{m:2d} / {p:2d}' for m in [2,4,8,16,32] for p in [4,8,16]])
axs[0].set_ylabel('Predictor count $M$ / width $b_M$');fig.subplots_adjust(left=.085,right=.90,bottom=.12,top=.92,wspace=.15)
fig.colorbar(im,cax=fig.add_axes([.92,.17,.014,.65]),label='Loss reduction, $L=1$ to $32$ (%)');save(fig,'08_full_grid')
print('IMPORTANCE',importance[importance.endpoint=='direct'].pivot(index='task',columns='term',values='share').round(2).to_string())
print('FAILURE CONTRASTS',pd.DataFrame(contrasts).round(3).to_string(index=False))
import platform
source_files=['paper.tex','experiment.py','evaluation.py','math_utils.py','train.py','architectures.py','atomic_networks.py','experiment_scaling.ipynb']
(ROOT/'provenance.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,source_hashes={f:hashlib.sha256((ROOT.parent/f).read_bytes()).hexdigest() for f in source_files}),indent=2))
# Uncertainty for responsibility contrasts: resample the same five seed blocks,
# including the shared L=1 denominator, for both components jointly.
uncertainty=[]
for t in TASKS:
 for lo,hi in [(1,2),(1,4),(1,32),(8,32)]:
  arrays={}
  for k in ['loss','loss_under_oracle','delegator_regret_loss']:
   aa=df[(df.task==t)&(df.delegators==hi)].pivot(index=KEY,columns='seed_id',values=k).to_numpy()
   bb=df[(df.task==t)&(df.delegators==lo)].pivot(index=KEY,columns='seed_id',values=k).to_numpy()
   denom=df[(df.task==t)&(df.delegators==1)].pivot(index=KEY,columns='seed_id',values='loss').to_numpy()
   draws=100*(((aa-bb)@COUNTS.T)/(denom@COUNTS.T)).mean(0)
   low,high=np.quantile(draws,[.025,.975])
   uncertainty.append(dict(task=t,from_L=lo,to_L=hi,component=k,change=(100*(aa.mean(1)-bb.mean(1))/denom.mean(1)).mean(),lo=low,hi=high))
pd.DataFrame(uncertainty).to_csv(ROOT/'fault_uncertainty.csv',index=False)
