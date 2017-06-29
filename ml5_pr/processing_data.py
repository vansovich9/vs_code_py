import pandas as pd
import numpy as np

def LoadFile(file_name ):
    data = pd.read_csv(file_name, header=0,
                    sep=';', na_values=['None'])

    data['ap_hi'] = abs(data['ap_hi'])
    data['ap_lo'] = abs(data['ap_lo'])
    data['ap_lo'] = data['ap_lo'].fillna(0)
    data['ap_hi'] = data['ap_hi'].fillna(0)

    data['age'] = round(data['age']/365, 2)

    data['gender'] = data['gender'] - 1
    
    data.at[(data['ap_lo'] == 0),'ap_lo'] = data['ap_lo'].mean()
    data.at[(data['ap_hi'] == 0),'ap_hi'] = data['ap_hi'].mean()

    while (data[data.ap_lo > 200]['id'].count()>0):
        data.at[(data['ap_lo'] > 200),'ap_lo'] = data[data.ap_lo > 200]['ap_lo'] // 10

    while (data[data.ap_hi > 250]['id'].count()>0):
        data.at[(data['ap_hi'] > 250),'ap_hi'] = data[data.ap_lo > 250]['ap_hi'] // 10

    while (data[data.ap_hi <= 20]['id'].count()>0):
        data.at[(data['ap_hi'] < 50),'ap_hi'] = data[data.ap_lo < 50]['ap_hi'] * 10

    while (data[data.ap_lo <= 20]['id'].count()>0):
        data.at[(data['ap_lo'] <= 20),'ap_lo'] = data[data.ap_lo <= 20]['ap_lo'] * 10

    data.at[(data.ap_lo > data.ap_hi),['ap_hi','ap_lo']] = data[data.ap_lo > data.ap_hi][['ap_lo','ap_hi']]

    data_bmi = pd.DataFrame(
        {'bmi': round(((data['weight']) / ((data['height'] / 100)**2)), 2)})

    #data_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
    #    109 + (0.5 * round(data['age'] / 365, 2)) + (0.1 * data['weight']), 0)})
    #data_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
        #64 + (0.1 * round(data['age'] / 365, 2)) + (0.15 * data['weight']), 0)})
    data_ap_hi_n = pd.DataFrame({'ap_hi_n': round(
        109 + (0.5 * data['age']) + (0.1 * data['weight']), 0)})
    data_ap_lo_n = pd.DataFrame({'ap_lo_n': round(
        64 + (0.1 * data['age']) + (0.15 * data['weight']), 0)})

    data = pd.concat((data, data_bmi, data_ap_hi_n, data_ap_lo_n), axis=1)

    data['ap_hi_n'] = data['ap_hi_n'].fillna(0)
    data['ap_lo_n'] = data['ap_lo_n'].fillna(0)

    data['ap_hi_n'] = np.sqrt((data['ap_hi'] - data['ap_hi_n'])**2)
    data['ap_lo_n'] = np.sqrt((data['ap_lo'] - data['ap_lo_n'])**2)
    data.at[(data['ap_hi_n'] >= 20 ), 'ap_hi_c'] = 1.0
    data.at[(data['ap_lo_n'] >= 10 ), 'ap_lo_c'] = 1.0

    data['ap_hi_c'] = data['ap_hi_c'].fillna(0)
    data['ap_lo_c'] = data['ap_lo_c'].fillna(0)

    data.at[(data['bmi'] <= 16 ), 'bmi_n_1'] = 1 	#Выраженный дефицит массы тела
    data.at[(data['bmi'] <= 18.5 ), 'bmi_n_2'] = 1 	#Недостаточная (дефицит) масса тела
    data.at[(data['bmi'] >= 25 ), 'bmi_n_4'] = 1 	#Избыточная масса тела (предожирение)
    data.at[(data['bmi'] >= 30 ), 'bmi_n_5'] = 1 	#Ожирение первой степени
    data.at[(data['bmi'] >= 35 ), 'bmi_n_6'] = 1 	#Ожирение второй степени
    data.at[(data['bmi'] >= 40 ), 'bmi_n_7'] = 1 	#Ожирение третьей степени (морбидное)

    data['bmi_n_1'] = data['bmi_n_1'].fillna(0)
    data['bmi_n_2'] = data['bmi_n_2'].fillna(0)
    data['bmi_n_4'] = data['bmi_n_4'].fillna(0)
    data['bmi_n_5'] = data['bmi_n_5'].fillna(0)
    data['bmi_n_6'] = data['bmi_n_6'].fillna(0)
    data['bmi_n_7'] = data['bmi_n_7'].fillna(0)
    #bmi
    data.at[(data['bmi'] < 18.5 ), 'bmi_r_1'] = 1 	#Risk of developing problems such as nutritional deficiency and osteoporosis	under 18.5
    data.at[(data['bmi'] >= 23 ), 'bmi_r_3'] = 1 	    #Moderate risk of developing heart disease, high blood pressure, stroke, diabetes	23 to 27.5
    data.at[(data['bmi'] >= 27.5 ), 'bmi_r_4'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5

    data['bmi_r_1'] = data['bmi_r_1'].fillna(0)
    data['bmi_r_3'] = data['bmi_r_3'].fillna(0)
    data['bmi_r_4'] = data['bmi_r_4'].fillna(0)
    data['bmi'] = data['bmi'].fillna(0)

    #cholesterol
    data.at[(data['cholesterol'] == 1 ), 'cholesterol_1'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data.at[(data['cholesterol'] == 2 ), 'cholesterol_2'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data.at[(data['cholesterol'] == 3 ), 'cholesterol_3'] = 1 	#High risk of developing heart disease, high blood pressure, stroke, diabetes	over 27.5
    data['cholesterol_1'] = data['cholesterol_1'].fillna(0)
    data['cholesterol_2'] = data['cholesterol_2'].fillna(0)
    data['cholesterol_3'] = data['cholesterol_3'].fillna(0)
    #gluc
    data.at[(data['gluc'] == 1 ), 'gluc_1'] = 1
    data.at[(data['gluc'] == 2 ), 'gluc_2'] = 1
    data.at[(data['gluc'] == 3 ), 'gluc_3'] = 1
    data['gluc_1'] = data['gluc_1'].fillna(0)
    data['gluc_2'] = data['gluc_2'].fillna(0)
    data['gluc_3'] = data['gluc_3'].fillna(0)

    return data