import fastf1
import matplotlib.pyplot as plt
import pandas as pd
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()
########PREDICTING_PIT_STOP############
#print(dir(session))
b = session.laps
print(b.info())
#print(b[["PitOutTime","PitInTime"]])
b["PitOutTime"]=b["PitOutTime"].fillna(0)
b["PitInTime"]=b["PitInTime"].fillna(0)

for i in range(len(b)-1):
    if(b["TyreLife"].iloc[i+1]==1):
        b.loc[b.index[i],"idk"]=1
    else:
        b.loc[b.index[i],"idk"]=0
b["Compound"]=b["Compound"].map({
    "SOFT":0,"MEDIUM":1,"HARD":2
})
b["LapStartTime"]=b["LapStartTime"].dt.total_seconds()
b["PitOutTime"] = (
    pd.to_timedelta(
        b["PitOutTime"]
    )
    .dt.total_seconds()
)
b["PitInTime"] = (
    pd.to_timedelta(
        b["PitInTime"]
    )
    .dt.total_seconds()
)
b["idk"] = (
    b["TyreLife"]
    .shift(-1)
    == 1
).fillna(False).astype(int)

for i in range(len(b)-1):
    b.loc[b.index[i],"lt"]=b["LapStartTime"].iloc[i+1]-b["LapStartTime"].iloc[i]
    b.loc[b.index[i],"pt"]=b["PitOutTime"].iloc[i+1]-b["PitInTime"].iloc[i]
for i in range(len(b)-1):
    if(b["idk"].iloc[i]==1):
        b.loc[b.index[i],"lt"]=b["lt"].iloc[i]-b["pt"].iloc[i]
print(b[["LapNumber","PitOutTime","PitInTime","TyreLife","idk","Compound","LapStartTime","lt","pt"]].head())
from sklearn.ensemble import RandomForestClassifier
l=RandomForestClassifier()
X=b[["LapNumber","TyreLife","Compound","lt"]]
y=b["idk"]
from sklearn.model_selection import train_test_split
f,g,h,i=train_test_split(X,y,stratify=y,train_size=0.5)
l.fit(f,h)
m=l.predict(g)
print(m)
print(i)
from sklearn.metrics import confusion_matrix
print(
   confusion_matrix(i, m)
)
from sklearn.metrics import accuracy_score
acc=accuracy_score(m,i)
print(acc)



#######TYRE_DEGRADATION########
print(b[["TyreLife","FreshTyre"]])
b = b.pick_quicklaps()
b["LapTimeSeconds"] = (
    b["LapTime"]
    .dt.total_seconds()
)
X = b[[
    "TyreLife",
    "Compound",
    "LapNumber",
    
]]
y = b["LapTimeSeconds"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = GradientBoostingRegressor()
model.fit(
    X_train,
    y_train
)
pred = model.predict(X_test)
print(
    "MAE:",
    mean_absolute_error(
        y_test,
        pred
    )
)
print(
    pd.DataFrame({
        "Actual": y_test,
        "Predicted": pred
    })
)

#plt.scatter(
#    b["TyreLife"],
#    b["LapTimeSeconds"]
#)
#plt.xlabel("Tyre Life")
#plt.ylabel("Lap Time (seconds)")
#plt.title("Tyre Degradation")
#plt.show()
from sklearn.metrics import r2_score
print(
    r2_score(
        y_test,
        pred
    )
)
for name, imp in zip(
    X.columns,
    model.feature_importances_
):
    print(name, imp)
plt.plot(b["LapNumber"],b["LapTimeSeconds"])
#plt.show()
colors={0:"red",1:"green",2:"blue"}
for compound in b["Compound"].unique():
    temp=b[b["Compound"]==compound]
    plt.scatter(temp["TyreLife"],temp["LapTimeSeconds"],c=colors[compound],label=compound)
plt.legend()
#plt.show()


#######PIT WINDOW SIMULATOR#######
#1. we predict the lap time of next 5-10 laps with and without pitstops if the time
#   without pitstops is less then no pitstop or else the code should advise to take the pitstop
#2. we take avg pitstop time and now add it to time of the laps with pistop and then make the desicion
#3. we need avg pitstop time and we already predicted lap time seconds but now we need to do it separately for 
#.  before and after new tyre is made
#now will have time when tyre is changed i.e pitstop and never will have other values
#pit_loss=[]
#for i in range(len(b)-1):
    #if(b["TyreLife"].iloc[i]>b["TyreLife"].iloc[i+1]):
     #   b.loc[b.index[i],"now"]=b["LapTimeSeconds"].iloc[i]
    #else:
     #   b.loc[b.index[i],"never"]=b["LapTimeSeconds"].iloc[i]
  #      pit_loss.append(b["LapStartTime"].iloc[i+1]-b["LapStartTime"].iloc[i])
#print(pit_loss)
#print(sum(pit_loss)/len(pit_loss))
print(b.info())
time_to_predict_new=[]
time_to_predict_old=[]
for i in range(len(b)-1):
    if b.loc[b.index[i],"TyreLife"]<7:
        time_to_predict_new.append(b["LapTimeSeconds"].iloc[i])
    else:
        time_to_predict_old.append(b["LapTimeSeconds"].iloc[i])

print(time_to_predict_new)
print(time_to_predict_old)
from sklearn.ensemble import GradientBoostingRegressor
fresh = b[b["TyreLife"] < 7]
ntfresh=b[b["TyreLife"]>=7]
h=GradientBoostingRegressor()
X=fresh[["LapNumber","TyreLife","Compound"]]
y=time_to_predict_new
X = X.iloc[:-1]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)
h.fit(X_train,y_train)
f=h.predict(X_test)
print(f)
from sklearn.metrics import r2_score
print(
    r2_score(
        y_test,
        f
    )
)
u=GradientBoostingRegressor()
X=ntfresh[["LapNumber","TyreLife","Compound","LapTimeSeconds"]]
y=time_to_predict_old

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42
)
u.fit(X_train,y_train)
p=u.predict(X_test)
print(p)
from sklearn.metrics import r2_score
print(
    r2_score(
        y_test,
        p
    )
)
h=[]
for x in p:
    h.append(x)
print(h)
j=[]
for x in f:
    j.append(x)
print(j)
print(len(h),len(j))
for i in range(14,len(h)):
    j.append(sum(j)/len(j))
print(j)
print(len(h),len(j))
lap_nums = ntfresh.loc[X_test.index]["LapNumber"].tolist()

for i in range(len(h)-5):
    if sum(h[i:i+5]) < sum(j[i:i+5]):
        print("Pit around lap", lap_nums[i])