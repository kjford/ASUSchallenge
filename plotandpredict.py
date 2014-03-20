'''
Plot data from ASUS Kaggle challenge
Assumes data already in MySQL database

Plots:
1. Repairs vs time averaged for each module group
2. Repairs vs time averaged for each component group
3. Module - Component interaction matrix as image (total repairs in last year)
4. Repair vs sale of modules over time
5. Repair life-time (repair_date - sale_date) for each module-component pair

'''
import numpy as np
import MySQLdb as mdb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import relativedelta
import csv

predfile='pred_lifetime_longertail.csv'

authent = {'host':'localhost',
           'user':'pyuser',
           'passwd':'testpass',
           'db':'asuschal'}


con = mdb.connect(**authent)
plotcycle=('k','r','b','g','m','c','y','k--','r--','b--','g--','m--','c--','y--')
xlims=('2005-01','2010-01')
with con:
    cur = con.cursor()
    # unique module and components
    cur.execute("""SELECT DISTINCT module
                FROM repairs
                ORDER BY module""")
    allmod=[x[0] for x in cur.fetchall()]
    nmod=len(allmod)
    cur.execute("""SELECT DISTINCT component
                FROM repairs
                ORDER BY component""")
    allcomp=[x[0] for x in cur.fetchall()]
    ncomp=len(allcomp)
    '''
    FIG 1
    repairs per module vs time
    '''
    
    df=pd.io.sql.read_frame("""SELECT module, repair_date, COUNT(*) as nrep
                    FROM repairs
                    GROUP BY repair_date, module
                    ORDER BY repair_date
                    """,con)
    df1=df.set_index('repair_date')
    plt.figure()
    c=0
    for k,g in df1.groupby('module'):
        g.nrep.plot(style=plotcycle[c],label=k,xlim=xlims)
        c+=1
    plt.legend(loc='best')
    plt.title('Repairs per Module vs Time')
    plt.show()
    
    
    
    '''
    FIG 2
    repairs per component vs time
    '''
    
    df2=pd.io.sql.read_frame("""SELECT component, repair_date, COUNT(*) as nrep
                    FROM repairs
                    GROUP BY repair_date, component
                    ORDER BY repair_date
                    """,con)
    df2=df2.set_index('repair_date')
    plt.figure()
    count=0
    sp=1
    for k,g in df2.groupby('component'):
        plt.subplot(2,3,sp)
        g.nrep.plot(label=k,xlim=xlims)
        if count%6==0:
            sp+=1
            plt.legend(loc='best')
        count+=1
    plt.title('Repairs per Module vs Time')
    plt.show()
    
    '''
    FIG 3
    repairs per module-component in last year
    '''
    df3=pd.io.sql.read_frame("""SELECT module, component, COUNT(*) as nrep
                    FROM repairs
                    WHERE repair_date > DATE('2009-01-01')
                    GROUP BY module, component
                    ORDER BY module, component
                    """,con)
    
    interactmat = np.zeros((nmod,ncomp))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for ind,row in df3.iterrows():
        i=allmod.index(row['module'])
        j=allcomp.index(row['component'])
        interactmat[i,j]=row['nrep']
    ax.imshow(interactmat,interpolation='None')
    ax.set_xticks(range(ncomp))
    ax.set_yticks(range(nmod))
    ax.set_xticklabels(allcomp)
    ax.set_yticklabels(allmod)
    ax.set_title('Module-component repairs in last year')
    fig.show()
    
    '''
    Fig 4
    sales per module
    note: modules are sold as a unit, redundant info in components
    '''
    dfsale=pd.io.sql.read_frame("""SELECT module, sale_date, sum(sale_num) as nrep
                    FROM sales
                    GROUP BY sale_date, module
                    ORDER BY sale_date, module
                    """,con)
    df4=dfsale.set_index('sale_date')
    plt.figure()
    '''
    for k,g in df1.groupby('module'): 
        g.nrep.plot(label=k)
    '''
    plt.legend(loc='best')
    c=0
    for k,g in df4.groupby('module'): 
        if k!='M0':
            g.nrep.cumsum().plot(style=plotcycle[c],label=k,xlim=xlims)
            c+=1
    plt.legend(loc='best')
    plt.title('Cumulative Sales per Module vs Time')
    plt.show()
    
    '''
    Fig 5
    repair life-time per module
    '''
    df5=pd.io.sql.read_frame("""SELECT module, component, AVG(datediff(repair_date,sale_date)) AS avlt
                    FROM repairs
                    GROUP BY module, component
                    ORDER BY module, component""",con)
    ltmat = np.zeros((nmod,ncomp))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for ind,row in df5.iterrows():
        i=allmod.index(row['module'])
        j=allcomp.index(row['component'])
        ltmat[i,j]=row['avlt']
    ax.imshow(ltmat,interpolation='None')
    ax.set_xticks(range(ncomp))
    ax.set_yticks(range(nmod))
    ax.set_xticklabels(allcomp)
    ax.set_yticklabels(allmod)
    ax.set_title('Average life-time of module/component')
    fig.show()
    
    '''
    Fig 6
    Ditribution of ages of each module a repair time
    '''
    
    df6=pd.io.sql.read_frame("""SELECT module, Round(datediff(repair_date,sale_date)/30) AS dd, count(*) AS nrep
                    FROM repairs
                    GROUP BY dd, module
                    ORDER BY module, dd""",con)
    
    df6=df6.set_index('dd')
    plt.figure()
    c=0
    for k,g in df6.groupby('module'):
        g.nrep.plot(style=plotcycle[c],label=k,xlim=(-5,60))
        c+=1
    plt.legend(loc='best')
    plt.title('Age at repair (months)')
    plt.show()


'''
Fit data:
compute age distribution for each module over time
use to get p(repair|age)
'''

# agemat is a nmonths x nmonths array
# each column is a month in time and each row is a month in age
agemats={}
dstart='2005-01-01'
dend='2011-07-01'

#function to generate list of dates in month increments
def monthinclist(datestart,dateend):
    fmt='%Y-%m-%d'
    d1=datetime.strptime(dateend,fmt)
    d2=datetime.strptime(datestart,fmt)
    r=relativedelta.relativedelta(d1,d2)
    nmonths=r.months + 12*r.years
    molist=[d2.strftime(fmt)]
    y=d2.year
    m=d2.month
    d=d2.day
    for i in range(nmonths):
        if m+1>12:
            m=1
            y+=1
        else:
            m+=1
        molist.append(datetime(y,m,d).strftime(fmt))
    return molist


dlist = monthinclist(dstart,dend)
nmo=len(dlist)
shiftmat=np.diag(np.ones(nmo-1),k=-1)
fig=plt.figure()
c=0
# for each module:
# get age of each sold unit over time
for k,g in dfsale.groupby('module'):
    agem=np.zeros((nmo,nmo))
    for i in range(nmo):
        shfted=shiftmat.dot(agem).dot(shiftmat.T)
        try:
            testind=datetime.date(datetime.strptime(dlist[i],'%Y-%m-%d'))
            toadd=float(g.nrep[g.sale_date==testind])
        except:
            toadd=0.0
        agem[:,i:]=shfted[:,i:]
        agem[0,i]=toadd
        
    agemats[k]=agem.copy()
    ax=fig.add_subplot(3,3,c)
    ax.imshow(agemats[k],interpolation='None')
    ax.set_xlabel(k)
    ax.set_ylabel('Age (mo)')
    c+=1
fig.show()

# normalize repairs|age by age sum over ages to last repair date
lastrepairind = dlist.index('2009-12-01')+1
repgiveage={}
c=0
plt.figure()
# make function to fit exponential decay to data
# make linear with first value as y intercept
def expfit(x,y,newx):
    yl=np.log(y)
    a, b = np.polyfit(x, yl, 1)
    ynew=np.exp(b+a*newx)
    return ynew
    
for k,g in df6.groupby('module'):
    ages=np.array(g.index,dtype='int')
    counts=np.array(g.nrep)
    # drop last two as these look unreliable
    ages=ages[:-2]
    counts=counts[:-2]
    # get cumulative age distribution
    agesum=agemats[k][:,:lastrepairind].sum(axis=1)
    r=np.zeros(len(agesum))
    r[ages]=counts
    # divide repair counts|age by age distribution
    r=r/agesum
    r[~np.isfinite(r)]=0
    #fit decay from 24 months looks good to 32 or first 0
    newx=np.arange(len(r[24:]))
    fitto=list(r).index(0)
    if k=='M2':
        fitto=32
    elif k=='M7':
        fitto=40
    elif fitto>36:
        fitto=36
    y=expfit(range(fitto-24),r[24:fitto],newx)
    r[24:]=y
    repgiveage[k]=r
    plt.plot(range(len(agesum)),r,plotcycle[c],label=k)
    c+=1

plt.title('Normalized Repair|age')
plt.legend(loc='best')
plt.show()

# use this to predict number of model repairs (in training set and test)
modupred={}
c=0
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for modu,vals in repgiveage.iteritems():
    modupred[modu]=agemats[modu].T.dot(vals)
    ax.plot(range(len(dlist)),modupred[modu],plotcycle[c],label=modu)
    c+=1
ax.set_ylim((0,9000))
ax.set_xticks(range(len(dlist))[::4])
ax.set_xticklabels(dlist[::4])
plt.title('Predictions')
plt.legend(loc='best')
plt.show()

# partition model repairs by component given weights in interaction matrix
normpart=interactmat/(1.0*interactmat.sum(axis=1).reshape(9,1))

# write out predictions using target mapping
outmap=pd.read_csv('Output_TargetID_Mapping.csv')
id=[]
pred=[]
idcnt=1
for index,row in outmap.iterrows():
    id.append(idcnt)
    idcnt+=1
    modkey=row[0]
    compkey=row[1]
    yr=row[2]
    mo=row[3]
    # make the date string to search by
    preddate=datetime(yr,mo,1).strftime('%Y-%m-%d')
    predind = dlist.index(preddate)
    # get the predicted number of repairs for that model
    nmodrep=modupred[modkey][predind]
    # get the proportion due to the specific component
    compind=allcomp.index(compkey)
    modind=allmod.index(modkey)
    predi=(np.round(nmodrep*(1.0*normpart[modind,compind])))
    if ~np.isfinite(predi):
        predi=0
    pred.append(predi)
    
print('Average number of repairs: %d'%(np.mean(pred)))
header=['id','target']    
f = open(predfile,'wb')
csvf=csv.writer(f)
csvf.writerow(header)
csvf.writerows(zip(id,pred))
f.close()