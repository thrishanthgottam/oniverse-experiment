import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=json.load(open('results.json'))
main=R['main']
CL=['exact','near_2','near_5','near_10','near_20']
def ms(f): 
    a=np.array([f(r) for r in main]); return a.mean(), a.std()
print(R['env']); print('docs',R['n_docs'])
print('N',main[0]['N'],'new used',[r['n_new_used'] for r in main],'removed',[r['new_removed_as_near_copies'] for r in main])
print('infospheres',ms(lambda r:r['n_infospheres']),'mean size',ms(lambda r:r['mean_infosphere_size']))
print('bloom bytes',ms(lambda r:r['bloom_bytes']),'k',main[0]['bloom_k'],'embed ms',ms(lambda r:r['embed_ms_mean']))
names=['A','B','C','D1@0.95','D2@0.85','D2@0.9','D2@0.95','E@0.85','E@0.9','E@0.95']
S={}
for n in names:
    S[n]={c:ms(lambda r:r['metrics'][n][c]) for c in CL+['new','precision','recall_all_dups']}
    print(n,' '.join(f"{c}={S[n][c][0]:.3f}±{S[n][c][1]:.3f}" for c in CL+['new','precision','recall_all_dups']))
print('LATENCY (N=5000)')
for k in ['A','B','C','D1','D2','E']:
    print(k, ms(lambda r:r['latency_ms'][k]['mean']), ms(lambda r:r['latency_ms'][k]['p95']))
print('DIAG')
for c in CL:
    print(c, ms(lambda r:r['diag'][c]['median_cos_to_own_centroid']), ms(lambda r:r['diag'][c]['frac_centroid_ge_0.95']),
      {k:round(ms(lambda r:r['diag'][c]['routing_recall@k'][k])[0],3) for k in ['1','3','5','10']})
# scaling
sc={}
for r in R['scaling']: sc.setdefault(r['N'],[]).append(r)
print('SCALING')
Ns=sorted(sc)
for N in Ns:
    print(N,{k:round(np.mean([r['latency_ms'][k]['mean'] for r in sc[N]]),4) for k in ['A','B','C','D1','D2','E']}, 'infos',np.mean([r['n_infospheres'] for r in sc[N]]),
      'D2@0.95 near recall', round(np.mean([np.mean([r['metrics']['D2@0.95'][c] for c in CL[1:]]) for r in sc[N]]),3),
      'bloomKB',round(np.mean([r['bloom_bytes'] for r in sc[N]])/1024,1))
# Figure 1: recall by class
labels=['Exact','2%','5%','10%','20%']
cfg=[('A','A  Bloom only','tab:gray','o'),('B','B  Bloom→MinHash→cosine','tab:blue','s'),('C','C  MinHash LSH (all items)','tab:green','^'),
     ('D1@0.95','D1  ONIVERSE, centroid gate 0.95','tab:red','v'),('D2@0.95','D2  ONIVERSE, top-5 routing','tab:orange','D'),('E@0.95','E  Brute-force cosine','black','x')]
fig,ax=plt.subplots(figsize=(7.2,4.2))
for n,l,c,m in cfg:
    y=[S[n][k][0] for k in CL]; e=[S[n][k][1] for k in CL]
    ax.errorbar(range(5),y,yerr=e,label=l,color=c,marker=m,capsize=3,lw=1.4,ms=6)
ax.set_xticks(range(5)); ax.set_xticklabels(labels); ax.set_xlabel('Duplicate type (fraction of words edited)')
ax.set_ylabel('Recall (detected as duplicate)'); ax.set_ylim(-0.03,1.05); ax.grid(alpha=.3); ax.legend(fontsize=7.5,loc='center right')
plt.tight_layout(); plt.savefig('fig_recall.png',dpi=200); plt.close()
# Figure 2: latency scaling
fig,ax=plt.subplots(figsize=(7.2,3.8))
for k,l,c,m in [('A','A  Bloom only','tab:gray','o'),('B','B  Bloom→MinHash→cosine','tab:blue','s'),('C','C  MinHash LSH','tab:green','^'),('D2','D2  ONIVERSE top-5 routing','tab:orange','D'),('E','E  Brute-force cosine','black','x')]:
    y=[np.mean([r['latency_ms'][k]['mean'] for r in sc[N]]) for N in Ns]
    ax.plot(Ns,y,label=l,color=c,marker=m,lw=1.4)
ax.set_xlabel('Stored chunks (N)'); ax.set_ylabel('Mean detection latency per chunk (ms)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_latency.png',dpi=200); plt.close()
json.dump({k:{c:list(v) for c,v in d.items()} for k,d in S.items()},open('summary.json','w'),indent=1)
