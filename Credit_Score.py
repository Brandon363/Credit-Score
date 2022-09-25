import streamlit as st
import pandas as pd
import numpy as np
import joblib


import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


from streamlit_option_menu import option_menu

selected_menu = option_menu(
        menu_title = None,
        options = ["Home", "About The Developer", "Contact"],
        icons = ["house", "person-workspace","envelope"],
        menu_icon = "cast",
        orientation = "horizontal"
)


#Load data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

#Get the keys from the dictionary
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# Find the key from the dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


data = load_data("data/german_credit_data_3.csv")

model = joblib.load('files/My_model')
  

Sex_label     = {'male': 1, 'female': 0}

Purpose_label = {'radio/TV': 5, 
                    'education': 3, 
                    'furniture/equipment': 4, 
                    'car': 1, 'business': 0, 
                    'domestic appliances': 2, 
                    'repairs': 6, 
                    'vacation/others': 7} 

Housing_label          = {'own': 1, 'free': 0, 'rent': 2}

Saving_accounts_label  = {'little': 0, 'quite rich': 2, 'rich': 3, 'moderate': 1}


Checking_account_label = {'little': 0, 'moderate': 1, 'rich': 2}

class_label            = {'Good' : 0, 'Bad' : 1}




#-------------------------------------PAGE 1 ----------------------------------------------------
nav = st.sidebar.radio("Navigation",["Client", "Admin"])
if selected_menu == "Home":

        #-------------------------------------Client page ----------------------------------------------------
        if nav == "Client":
            st.subheader("Prediction")
            #defining variables
            sex                 = st.selectbox("Select your Gender", tuple(Sex_label.keys()))
            housing             = st.selectbox("Select your Housing condition", tuple(Housing_label.keys()))
            saving_accounts     = st.selectbox("Select your Saving accounts state", tuple(Saving_accounts_label.keys()))
            checking_account    = st.selectbox("Select your current account state", tuple(Checking_account_label.keys()))
            age                 = st.number_input("How old are you", 18, 100)
            duration            = st.number_input("Select the duration", 0, 100)
            credit_amount       = st.number_input("Select your credit amount", 0, 100000)

            #Encoded values
            c_sex             = get_value(sex, Sex_label)
            c_housing         = get_value(housing, Housing_label)
            c_saving_accounts = get_value(saving_accounts, Saving_accounts_label)
            c_checking_account  = get_value(checking_account, Checking_account_label)


            pretty_data = {
                "Age"               : age,
                "Sex"               : sex,
                "Housing"           : housing, 
                "Saving accounts"   : saving_accounts,
                "Checking accounts" : checking_account,
                "Credit amount"     : credit_amount,  
                "Duration"          : duration       
            }

            sample_data = [(age), 
                            (c_sex), 
                            (c_housing), 
                            (c_saving_accounts), 
                            (c_checking_account), 
                            (credit_amount), 
                            (duration)]
            #st.write(sample_data)

            shaped_data = np.array(sample_data).reshape(1, -1)



            if st.button("Predict"):
                predictor = model
                prediction = predictor.predict(shaped_data)       

                result = prediction[0]
                predicted_result = get_key(result, class_label)

                if predicted_result == "Good":
                    st.success(predicted_result)
                else:
                    st.error(predicted_result)


            if st.button("Save_information"):
                predictor = model
                prediction = predictor.predict(shaped_data)       

                result = prediction[0]
                predicted_result = get_key(result, class_label)





                to_add = {
                            "Age"               : [age],
                            "Sex"               : [sex],
                            "Housing"           : [housing], 
                            "Saving accounts"   : [saving_accounts],
                            "Checking accounts" : [checking_account],
                            "Credit amount"     : [checking_account],  
                            "Duration"          : [duration],
                            "Decision"          : [predicted_result]
                        }
                to_add = pd.DataFrame(to_add)
                to_add.to_csv("data/New_entry_Data.csv",mode='a',header = False,index= False)
                st.success("Saved")

        #-------------------------------------Admin page ----------------------------------------------------
        if nav == "Admin":

            dfx_og =  pd.read_csv(r"data/data.csv", index_col=0)

            #for EDA
            column_obj = [dt for dt in data.columns if data[dt].dtype == "O" ]






            dfx = dfx_og.sort_values(by ="prob_Good" , ascending = False)


            #turning deciles
            dfx["Deciles"] = pd.qcut(dfx["prob_Bad"], 10, labels = np.arange(1, 11, 1))
            dfx["Count"] = 1

            dfx = dfx.sort_values(by = "Deciles" , ascending = False)

            dfx["prob_Bad"] = dfx["prob_Bad"]*100
            dfx["prob_Good"] = dfx["prob_Good"]*100

            #rounding the percentages
            dfx = dfx.round({"prob_Bad": 2, "prob_Good": 2})

            pivot_table = pd.pivot_table(dfx, index = "Deciles", values = ["predicted", "prob_Good", "Count"], 
                                     aggfunc = { "predicted"               : sum,
                                                 "prob_Good"               : min, 
                                                 "Count"                   : pd.Series.count})

            pivot_table.rename(columns = {"predicted": "Bad", "Count": "Total"}, inplace = True)



            pivot_table["Good"]                  = pivot_table["Total"] - pivot_table["Bad"]
            pivot_table["Cumm_Good"]             = pivot_table["Good"].cumsum()
            pivot_table["Cumm_Bad"]              = pivot_table["Bad"].cumsum()
            pivot_table["Cumm_Bad %"]            = 100 * (pivot_table["Bad"].cumsum()/pivot_table["Bad"].sum())
            pivot_table["Cumm_Good %"]           = 100 * (pivot_table["Good"].cumsum()/pivot_table["Good"].sum())
            pivot_table["Cumm_Bad_Avoided %"]    = 100 - pivot_table["Cumm_Bad %"]





            if st.checkbox("Data Destribution"):
                dd_choice = st.selectbox("Destribute by", ["None","Sex", "Housing", "Saving accounts", "Checking account", "Risk", "Duration"])

                if dd_choice == "None":
                    st.write(" ")

                if dd_choice == "Sex":         
                    plt.figure(figsize = (6,3))
                    plt.bar(data["Sex"].value_counts().index, data["Sex"].value_counts())
                    st.pyplot()

                elif dd_choice =="Housing":         
                    plt.figure(figsize = (6,3))
                    plt.bar(data["Housing"].value_counts().index, data["Housing"].value_counts())
                    st.pyplot()

                elif dd_choice =="Saving accounts":         
                    plt.figure(figsize = (6,3))
                    plt.bar(data["Saving accounts"].value_counts().index, data["Saving accounts"].value_counts())
                    st.pyplot()

                elif dd_choice == "Checking account	":         
                    plt.figure(figsize = (6,3))
                    plt.bar(data["Checking account"].value_counts().index, data["Checking account"].value_counts())
                    st.pyplot()

                elif dd_choice == "Risk":         
                    plt.figure(figsize = (6,3))
                    plt.bar(data["Risk"].value_counts().index, data["Risk"].value_counts())
                    st.pyplot()

                elif dd_choice == "Duration":         
                    plt.figure(figsize = (9,3))
                    plt.plot(data["Duration"], linewidth = 1)
                    st.pyplot()

            if st.checkbox("Description of Distribuition Risk by"):
                rd_choice = st.selectbox("Destribute by", ["None","Sex", "Housing", "Saving accounts", "Checking account"])

                if rd_choice == "None":
                        st.write(" ")

                if rd_choice == "Sex":         
                    plt.figure(figsize = (6,3))
                    g = sns.countplot(x="Sex", data=data, palette="husl",hue="Risk")
                    g.set_title("Sex Count", fontsize=15)
                    g.set_xlabel("Sex type", fontsize=12)
                    g.set_ylabel("Count", fontsize=12)
                    st.pyplot()

                if rd_choice == "Saving accounts":         
                    plt.figure(figsize = (6,3))
                    g = sns.countplot(x="Saving accounts", data=data, palette="husl",hue="Risk")
                    g.set_title("Saving Accounts Count", fontsize=15)
                    g.set_xlabel("Saving Accounts type", fontsize=12)
                    g.set_ylabel("Count", fontsize=12)
                    st.pyplot()

                if rd_choice == "Housing":         
                    plt.figure(figsize = (6,3))
                    g = sns.countplot(x="Housing", data=data, palette="husl",hue="Risk")
                    g.set_title("Housing Count", fontsize=15)
                    g.set_xlabel("Housing type", fontsize=12)
                    g.set_ylabel("Count", fontsize=12)
                    st.pyplot()

                if rd_choice == "Checking account":         
                    plt.figure(figsize = (6,3))
                    g = sns.countplot(x="Checking account", data=data, palette="husl",hue="Risk")
                    g.set_title("Checking account Count", fontsize=15)
                    g.set_xlabel("Checking account type", fontsize=12)
                    g.set_ylabel("Count", fontsize=12)
                    st.pyplot()


            if st.checkbox("Show Current Credit Status"):
                st.dataframe(pivot_table)
            
            if st.checkbox("Show Added data"):
                new_entries = pd.read_csv("data/New_entry_Data.csv")
                st.dataframe(new_entries)


#-------------------------------------PAGE 2 ----------------------------------------------------

elif selected_menu == "About The Developer":
    #st.title("Who is he?")
    #st.write("The best there is")
    col1, col2 = st.columns(2)


    with col1:
        st.header("Brandon Mutombwa")
        st.image("my_pic/brandon_pic.jpg", caption = "Tonderai Brandon Mutombwa")
        
    with col2:
        st.write(" ")

    st.write("Brandon is a creative, Data Science student at the University of Zimbabwe who is enthusiastic about executing data driven solutions to increase efficiency, accuracy and utility of internal data processing. He is driven by a strong PASSION AND PURPOSE for solving data problems.")

    
    
 #-------------------------------------PAGE 3 ----------------------------------------------------
elif selected_menu == "Contact":
    st.write("Calls     : +263 776 464 136/ +263 77 586 0625         \nWhatsApp : +263 776 464 136        \nEmail : brandonmutombwa@gmail.com")

    
