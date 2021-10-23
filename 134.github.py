import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture



def sign(data2):
    f4=data2
    f4 = data2.sort_values(by=['VIN','Time'],ascending=(True,True))
    f5=f4
    f5['shift_Ign'] = f4.groupby('VIN')['Ignition_state'].apply(lambda i: i.shift(1))
    f5['diff_Ign'] = f5['Ignition_state'] - f5['shift_Ign']

    f5['shift_C_D'] = f4.groupby('VIN')['Charge_state'].apply(lambda i: i.shift(1))
    f5['diff_C_D'] = f5['Charge_state'] - f5['shift_C_D']
    return f5

def charge(data3):
    f6 = data3[data3["diff_C_D"] != 0]
    f6 = f6.dropna(subset=["diff_C_D"])
    f7 = f6.sort_values(by=['VIN', 'Time'], ascending=(True, True))
    f7 = f7.drop(columns=["shift_C_D", "shift_Ign","Ignition_state", "Charge_state"])
    concat_group = pd.DataFrame()
    concat_group_1 = pd.DataFrame()
    f8 = pd.DataFrame()
    f9 = pd.DataFrame()
    for name, group in f7.groupby(f7['VIN']):
        group = group.reset_index()
        new_group = group.drop(group[(group['diff_C_D'] == 1) & (group.index == 0)].index)
        new_group_1 = new_group.drop(
            new_group[(new_group['diff_C_D'] == -1) & (new_group.index == len(new_group))].index)
        new_group_1 = new_group_1.reset_index()
        new_group_1["pair_1"] = range(0, len(new_group_1))
        new_group_1["pair_2"] = new_group_1["pair_1"] // 2
        new_group_1["pair_3"] = new_group_1["pair_1"] % 2
        f8 = new_group_1[new_group_1["pair_3"] == 0]
        f8 = f8.sort_values(by=["pair_2"], ascending=(True))
        f8.set_index(["pair_2"], inplace=True)
        f8 = f8.rename(
            columns={'VIN': 'CS_VIN', 'Time': 'CS_Time', 'Mileage': 'CS_Mileage',
                     'SOC': 'CS_SOC', 'Charging_voltage': 'CS_Charging_voltage',
                     'Charging_current': 'CS_Charging_current'})
        f9 = new_group_1[new_group_1["pair_3"] == 1]
        f9 = f9.sort_values(by=['pair_2'], ascending=(True))
        f9.set_index(["pair_2"], inplace=True)
        f9 = f9.rename(
            columns={'VIN': 'CE_VIN', 'Time': 'CE_Time', 'Mileage': 'CE_Mileage',
                     'SOC': 'CE_SOC', 'Charging_voltage': 'CE_Charging_voltage',
                     'Charging_current': 'CE_Charging_current'})
        f10 = pd.concat([f8, f9], axis=1)
        concat_group_1 = concat_group_1.append(f10, ignore_index=True)

    f11 = concat_group_1.drop(columns=["level_0", "index","diff_Ign", "diff_C_D", "diff_Ign", "pair_1", "pair_3",
                                       "level_0", "index", "diff_Ign", "diff_C_D", "pair_1", "pair_3", "CE_VIN"])
    f11["CS_Day"] = f11["CS_Time"].dt.day
    f11["CS_Hour"] = f11["CS_Time"].dt.hour
    f11["CS_Minute"] = f11["CS_Time"].dt.minute
    f11["CE_Day"] = f11["CE_Time"].dt.day
    f11["CE_Hour"] = f11["CE_Time"].dt.hour
    f11["CE_Minute"] = f11["CE_Time"].dt.minute
    f11["SC"] = f11["CS_Hour"] + round(f11["CS_Minute"] / 60.0, 1)  #SC
    f11["EC"] = f11["CE_Hour"] + round(f11["CE_Minute"] / 60.0, 1)  # EC
    f11["C_time_1"] = f11["EC"] - f11["SC"]
    f11["C_time_2"] = (f11["CE_Day"] - f11["CS_Day"]) * 24
    f11["CD"] = f11["C_time_1"] + f11["C_time_2"]  # CD
    f11["CS_Power"] = f11["CS_Charging_voltage"]*f11["CS_Charging_current"]
    f11["CE_Power"] = f11["CE_Charging_voltage"]* f11["CE_Charging_current"]
    f11["power"] = f11["CS_Power"]
    f11["power"].loc[f11["CS_Power"] < f11["CE_Power"]] = f11["CE_Power"]
    return f11

def park(data4):
    f12=data4[data4["diff_Ign"]!=0]
    f12=f12.dropna(subset=["diff_Ign"])
    f13=f12.sort_values(by=['VIN','Time'],ascending=(True,True))
    f13=f13.drop(columns=["shift_C_D", "shift_Ign","Ignition_state", "Charge_state"])
    #,"index_sign"
    f14=f13.loc[f13["diff_Ign"]==1]
    f15=f13.loc[f13["diff_Ign"]==-1]
    f14 = f14.rename(
        columns={'VIN': 'CS_VIN', 'Time': 'PS_Time', 'Mileage': 'CS_Mileage',
                 'SOC': 'PS_SOC', 'Charging_voltage': 'CS_Charging_voltage',
                 'Charging_current': 'CS_Charging_current'})
    f15 = f15.rename(
        columns={'VIN': 'CS_VIN', 'Time': 'PE_Time', 'Mileage': 'CE_Mileage',
                 'SOC': 'PE_SOC', 'Charging_voltage': 'CE_Charging_voltage',
                 'Charging_current': 'CE_Charging_current'})
    f14["PS_Day"] = f14["PS_Time"].dt.day
    f14["PS_Hour"] = f14["PS_Time"].dt.hour
    f14["PS_Minute"] = f14["PS_Time"].dt.minute
    f15["PE_Day"] = f15["PE_Time"].dt.day
    f15["PE_Hour"] = f15["PE_Time"].dt.hour
    f15["PE_Minute"] = f15["PE_Time"].dt.minute
    f14["SP"] = f14["PS_Hour"] + round(f14["PS_Minute"] / 60.0, 1)
    f15["EP"] = f15["PE_Hour"] + round(f15["PE_Minute"] / 60.0, 1)
    return  f14,f15

def day(date1):
    f25=date1
    f25['week']=f25['CS_Time'].dt.dayofweek
    f25['Day'] = f25['week']
    f25['Day'].loc[f25['week'] == 5] = 1
    f25['Day'].loc[f25['week'] == 6] = 1
    f25['Day'].loc[f25['week'] == 0] = 0
    f25['Day'].loc[f25['week'] == 1] = 0
    f25['Day'].loc[f25['week'] == 2] = 0
    f25['Day'].loc[f25['week'] == 3] = 0
    f25['Day'].loc[f25['week'] == 4] = 0
    return f25

def price(price_data):
    price_data["Price"]=price_data["SC"]
    f28=price_data[["SC","EC","Price"]]
    f29=f28.values
    for i in range(0,len(f29)):
        a2=f29[i,0]
        a3=f29[i,1]
        if a2<=a3:
            if a2<8:
                if a3<8:
                    f29[i,2]=33
                elif a3<22:
                    f29[i,2]=int((33*(8-int(a2))+64*(int(a3)-8))/(int(a3)-int(a2)))
                else:
                    f29[i,2]=int(((33*((8-int(a2))+(int(a3)-22)))+64*(22-8))/(int(a3)-int(a2)))
            elif a2<22:
                if a3<22:
                    f29[i,2]=64
                else:
                    f29[i,2]=int((64*(22-int(a2))+33*(int(a3)-22))/(int(a3)-int(a2)))
            else:
                f29[i,2]=33
        else:
            if a2<8:
                f29[i,2]=int((33*(int(a3)+(8-int(a2))+24-22)+64*(22-8))/(int(a3)+24-int(a2)))
            elif a2<22:
                if a3<8:
                    f29[i,2]=int((64*(22-int(a2))+33*(int(a3)+24-22))/(int(a3)+24-int(a2)))
                else:
                    f29[i,2]=int((33*(24+8-22)+64*(22-int(a2)+int(a3)-8))/(int(a3)+24-int(a2)))
            else:
                if a3<8:
                    f29[i,2]=64
                elif a3<22:
                    f29[i,2]=int((33*(24-int(a2)+8)+64*(int(a3)-8)))
                else:
                    f29[i,2]=int((33*(24-int(a2)+int(a3)-22+8)+64*(22-8))/(int(a3)+24-int(a2)))
    f30=f29.reshape(len(f29),3)
    f31=pd.DataFrame(f30,columns=['SC','EC','Price'])
    f32=pd.merge(price_data,f31,left_index=True,right_index=True)
    return f32

def Mileage(data5):
    f33 = data5.sort_values(by=['CS_VIN', 'CS_Time'], ascending=(True, True))
    concat_group_2 = pd.DataFrame()
    for name, group in f33.groupby(f33['CS_VIN']):
        group['shift_CE_Mileage'] = group.groupby('CS_VIN')['CS_Mileage'].apply(lambda i: i.shift(1))
        group['Trip'] = group['CS_Mileage'] - group['shift_CE_Mileage']
        group['shift_CS_Mileage'] = group.groupby('CS_VIN')['CS_Mileage'].apply(lambda i: i.shift(-1))
        group['Tra_next'] = group['shift_CS_Mileage'] - group['CS_Mileage']
        concat_group_2 = concat_group_2.append(group, ignore_index=True)
    f34=concat_group_2.dropna()
    return f34

def merge(chargesession, startpark, endpark,type):
    f16 = pd.merge(chargesession,startpark, how='left', on=['CS_VIN','CS_Mileage'])
    f16["C_P_Time"] = f16["SC"] - f16["SP"]
    f16["C_Time_2"] = (f16["CS_Day"] - f16["PS_Day"]) * 24
    f16["DCI"] = f16["C_P_Time"] + f16["C_Time_2"]  # DCI
    f17 = pd.merge(f16, endpark, how='left', on=['CS_VIN', 'CE_Mileage'])
    f17["C_P_E_Time"] = f17["EP"] - f17["EC"]
    f17["C_E_Time_2"] = (f17["PE_Day"] - f17["CE_Day"]) * 24
    f17["DTI"] = f17["C_P_E_Time"] + f17["C_E_Time_2"]  # DTI
    f17["Time"]=f17["EP"]-f17["SP"]+(f17["PE_Day"]-f17["PS_Day"])*24
    f18=day(f17)
    f31=price(f18)
    f19 = f31[["CS_VIN","SC_x" ,"DCI", "CD", "DTI"]]
    f20 = pd.merge(f19,type,how="left",on="CS_VIN")
    f21 = Mileage(f31)
    f22= f21[["CS_VIN","SC_x" ,"DCI", "CD", "DTI", "CS_SOC", "power", "Price_x", "Time", "Trip", "Tra_next", "Day"]]
    f36 = pd.merge(f22, type, how="left", on="CS_VIN")
    return f20,f36

def GMM(clusterdata,factors):
    f21=pd.DataFrame(clusterdata,columns=["SC_x","DCI","CD","DTI"])
    a1=np.zeros([2])
    a1=a1.astype(np.int)
    a=0
    for i in range(2,21):
        gmm = GaussianMixture(n_components=i)
        gmm.fit(f21.values)
        BIC=gmm.bic(f21.values)
        arr_1=[i,BIC]
        a1=np.append(a1,arr_1,axis=0)
        if i==2:
           arr_2=BIC
        elif BIC < arr_2:
            arr_2=BIC
            a=i
    a1=np.delete(a1,[0,1],axis=0)
    a2=a1.reshape((19,2))
    f22=pd.DataFrame(a2,columns=["components","BIC"])
    gmm = GaussianMixture(n_components=a)
    gmm.fit(f21.values)
    labels = gmm.predict(f21.values)
    f23=pd.DataFrame(labels,columns=["Patterns"])
    f24=pd.merge(clusterdata,f23,left_index=True,right_index=True)
    f38=f24[["SC_x","DCI","CD","DTI","Type","Patterns"]]
    f38=f38.rename(columns={"SC_x":"SC"})
    f37=pd.merge(factors,f24,how="left", on=["CS_VIN","SC_x","DCI","CD","DTI"])
    f37=f37.rename(columns={"CS_SOC":"SOC","power":"Power","Price_x":"Price",
                            "Type_x":"Type"})
    f39=f37[["SOC","Power","Price","Time","Trip","Tra_next","Type","Patterns"]]
    f26=f39[(f39["Type"]==0)]
    f27=f39[(f39["Type"]==1)]
    return f22,f38,f26,f27


if __name__=="__main__":
    df1=pd.read_csv("01.data.csv")
    df2=df1
    df2["Time"] = pd.to_datetime(df1["Time"], format="%Y-%m-%d %H:%M:%S")
    df3=pd.read_csv("02.type.csv")
    df4=sign(df2)
    df5=charge(df4)
    df6,df7=park(df4)
    df8,df9=merge(df5,df6,df7,df3)
    df10,df11,df12,df13= GMM(df8,df9)
    df10.to_csv("BIC.csv")
    df11.to_csv("GMM_result.csv")
    df12.to_csv("mlogit_PEVs.csv")
    df13.to_csv("mlogit_CSEVs.csv")
