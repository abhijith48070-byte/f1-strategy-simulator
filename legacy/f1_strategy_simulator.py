import fastf1
import matplotlib.pyplot as plt
import pandas as pd
session = fastf1.get_session(2024, "Monaco", "R")
session.load()
##################################################
# PIT STOP PREDICTION
##################################################
laps = session.laps
print(laps.info())
print(laps[["PitOutTime", "PitInTime"]])
laps["PitOutTime"] = laps["PitOutTime"].fillna(0)
laps["PitInTime"] = laps["PitInTime"].fillna(0)
laps["Compound"] = laps["Compound"].map({"SOFT": 0,"MEDIUM": 1,"HARD": 2})
laps["LapStartTime"] = laps["LapStartTime"].dt.total_seconds()
laps["PitOutTime"] = (pd.to_timedelta(laps["PitOutTime"]).dt.total_seconds())
laps["PitInTime"] = (pd.to_timedelta(laps["PitInTime"]).dt.total_seconds())
laps["PitStopNextLap"] = (laps["TyreLife"].shift(-1) .eq(1).fillna(False).astype(int))
for i in range(len(laps) - 1):
    laps.loc[laps.index[i], "AdjustedLapTime"] = (laps["LapStartTime"].iloc[i + 1]- laps["LapStartTime"].iloc[i])
    laps.loc[laps.index[i], "PitTimeLoss"] = (laps["PitOutTime"].iloc[i + 1]- laps["PitInTime"].iloc[i])
for i in range(len(laps) - 1):
    if laps["PitStopNextLap"].iloc[i] == 1:
        laps.loc[laps.index[i], "AdjustedLapTime"] = (laps["AdjustedLapTime"].iloc[i]- laps["PitTimeLoss"].iloc[i])
print( laps[
        [
            "LapNumber",
            "PitOutTime",
            "PitInTime",
            "TyreLife",
            "PitStopNextLap",
            "Compound",
            "LapStartTime",
            "AdjustedLapTime",
            "PitTimeLoss",
        ]
    ].head()
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
pit_stop_model = RandomForestClassifier()
features_pitstop = laps[[ "LapNumber","TyreLife","Compound","AdjustedLapTime"]]
target_pitstop = laps["PitStopNextLap"]
X_train, X_test, y_train, y_test = train_test_split(
    features_pitstop,
    target_pitstop,
    stratify=target_pitstop,
    train_size=0.5
)
pit_stop_model.fit(X_train, y_train)
pit_predictions = pit_stop_model.predict(X_test)
print(pit_predictions)
print(y_test)
print(confusion_matrix( y_test,pit_predictions))
pit_accuracy = accuracy_score(y_test,pit_predictions)
print(pit_accuracy)

##################################################
# TYRE DEGRADATION MODEL
##################################################
print(laps[["TyreLife","FreshTyre"]])
laps = laps.pick_quicklaps()
laps["LapTimeSeconds"] = (laps["LapTime"].dt.total_seconds())
features_tyred = laps[["TyreLife","Compound","LapNumber"]]
target_tyred = laps["LapTimeSeconds"]
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
x_train, x_test, y_train, y_test = train_test_split(features_tyred,target_tyred,test_size=0.3,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
tyre_degradation_model = GradientBoostingRegressor()
tyre_degradation_model.fit(x_train,y_train)
predicted_lap_times = tyre_degradation_model.predict(x_test)
print("MAE:",mean_absolute_error(y_test,predicted_lap_times))
print(pd.DataFrame(  {"Actual": y_test,"Predicted": predicted_lap_times}))
plt.scatter(laps["TyreLife"],laps["LapTimeSeconds"])
plt.xlabel("Tyre Life")
plt.ylabel("Lap Time (seconds)")
plt.title("Tyre Degradation")
plt.show()
print(r2_score(  y_test,predicted_lap_times))
for feature_name, importance in zip(features_tyred.columns,tyre_degradation_model.feature_importances_):
    print(feature_name,importance)
plt.plot(laps["LapNumber"],laps["LapTimeSeconds"])
plt.show()
colors = {0: "red",1: "green",2: "blue"}
for compound in laps["Compound"].unique():
    compound_data = laps[laps["Compound"] == compound]
    plt.scatter(compound_data["TyreLife"],compound_data["LapTimeSeconds"],c=colors[compound],label=compound)
plt.legend()
plt.show()

##############################################
# PIT WINDOW SIMULATOR
###############################################

print(laps.info())
fresh_tyre_lap_times = []
worn_tyre_lap_times = []
for i in range(len(laps) - 1):
    if laps.loc[laps.index[i], "TyreLife"] < 7:
        fresh_tyre_lap_times.append(laps["LapTimeSeconds"].iloc[i])
    else:
        worn_tyre_lap_times.append(laps["LapTimeSeconds"].iloc[i])
print(fresh_tyre_lap_times)
print(worn_tyre_lap_times)
fresh_tyre_stints = laps[laps["TyreLife"] < 7]
worn_tyre_stints = laps[laps["TyreLife"] >= 7]
worn_tyre_stints = worn_tyre_stints.reset_index(drop=True)
fresh_tyre_model = GradientBoostingRegressor()
features_pitw= fresh_tyre_stints[["LapNumber", "TyreLife","Compound"]]
target_pitw = fresh_tyre_lap_times
features_pitw=features_pitw[:len(target_pitw)]
features = features_pitw.iloc[:-1]
Xk_train, Xk_test, y_train, y_test = train_test_split(features_pitw,target_pitw,test_size=0.5,random_state=42)
fresh_tyre_model.fit(Xk_train,y_train)
fresh_predictions = fresh_tyre_model.predict(Xk_test)
print(fresh_predictions)
print(r2_score(y_test,fresh_predictions))
worn_tyre_model = GradientBoostingRegressor()

worn_features = worn_tyre_stints[["LapNumber", "TyreLife", "Compound"]]
worn_target = worn_tyre_lap_times

worn_features = worn_features[:len(worn_target)]
worn_target = worn_target[:len(worn_features)]

Xl_train, Xl_test, y_train, y_test = train_test_split(
    worn_features,
    worn_target,
    test_size=0.5,
    random_state=42
)

worn_tyre_model.fit(Xl_train, y_train)
worn_predictions = worn_tyre_model.predict(Xl_test)
print(worn_predictions)
print(r2_score( y_test,worn_predictions))
worn_prediction_list = []
for prediction in worn_predictions:
    worn_prediction_list.append(prediction)
print(worn_prediction_list)
fresh_prediction_list = []
for prediction in fresh_predictions:
    fresh_prediction_list.append(prediction)
print(fresh_prediction_list)
print(len(worn_prediction_list),len(fresh_prediction_list))
for i in range(14,len(worn_prediction_list)):
    fresh_prediction_list.append(sum(fresh_prediction_list)/ len(fresh_prediction_list))
print(fresh_prediction_list)
print(len(worn_prediction_list),len(fresh_prediction_list))
print("max worn index:", max(worn_tyre_stints.index))
print("len worn:", len(worn_tyre_stints))

print("worn_X_test index:")
print(Xl_test.index.tolist())

print("max worn_X_test index:", max(Xl_test.index))
lap_numbers = worn_tyre_stints.loc[Xl_test.index]["LapNumber"].tolist()
for i in range(len(worn_prediction_list) - 5):
    if (sum(worn_prediction_list[i:i + 5])<sum(fresh_prediction_list[i:i + 5])):
        print("Pit around lap",lap_numbers[i])
