import streamlit as st 
import joblib
import pandas as pd
import numpy as np

st.title("German Apartment Rent Prediction App")
st.markdown("Fill in the apartment details below to get the estimated total rent")
st.divider()

df=pd.read_csv("train_data.csv")
model = joblib.load('best_model_compressed.joblib')


df_city_and_plz=df[['city','geo_plz']]

df_city_and_plz=df_city_and_plz.groupby('city')['geo_plz'].apply(list)
city_and_plz_dict=df_city_and_plz.to_dict()


heating_type_list=df['heatingType'].unique().tolist()
newly_const_list=df['newlyConst'].unique()
balcony_list=df['balcony'].unique()

price_trend_list=df['pricetrend'].sort_values(ascending=True).unique()
year_constructed_list=df['yearConstructed'].sort_values(ascending=True).unique()
has_kitchen_list=df['hasKitchen'].unique()

has_cellar_list=df['cellar'].unique()
living_space_list=df['livingSpace'].unique()
condition_list=df['condition'].unique().tolist()

interior_qual_list=df['interiorQual'].unique().tolist()
pets_allowed_list=df['petsAllowed'].unique().tolist()
lift_list=df['lift'].unique()

geo_plz_list=df['geo_plz'].unique()
number_of_rooms_list=df['noRooms'].unique()
thermal_char_list=df['thermalChar'].unique()

floor_list=df['floor'].unique()
size_per_room_list=df['SizePerRoom'].unique()

type_of_flat_list=df['typeOfFlat'].unique().tolist()
garden_list=df['garden'].unique()
city_list=df['city'].unique()


def remove_value(cols,value):
       cols.remove(value)

remove_value(heating_type_list,'unknown')
remove_value(condition_list,'unknown')
remove_value(interior_qual_list,'unknown')
remove_value(pets_allowed_list,'unknown')
remove_value(type_of_flat_list,'unknown')


city_input=st.selectbox("In which city is the flat in? (Please select a city)",city_list)
geo_plz_input=st.selectbox("What is the flats Postal code (Postleitzahl)?",options=list(set(city_and_plz_dict[city_input])),index=None,placeholder="Select the plz")
heatingtype_input=st.selectbox("What is the Heating Type?",heating_type_list,index=None,placeholder="Select the heating type")

newly_const_input=st.selectbox("The flats building is newly constructed?",newly_const_list)
balcony_input=st.selectbox("has a balcony?",balcony_list)
has_kitchen_input=st.selectbox("has a built-in kitchen?",has_kitchen_list)

has_cellar_input=st.selectbox("has a cellar?",has_cellar_list)
condition_input=st.selectbox("What is the flats condition?",condition_list)
interior_qual_input=st.selectbox("What is the flats interior qaulity?",interior_qual_list)

pets_allowed_input=st.selectbox("Are pets allowed?",pets_allowed_list)
lift_input=st.selectbox("has a lift?",lift_list)
garden_input=st.selectbox("has a garden?",garden_list)

type_of_flat_input=st.selectbox("What is the flat type?",type_of_flat_list)
living_space_input=st.slider("How much is the living space (in sqm)?",min_value=min(living_space_list),
                                    max_value=max(living_space_list),key='slider_1')
price_trend_slider_input = st.slider("Price Trend",min_value=min(price_trend_list),max_value=max(price_trend_list),key='slider_2')
number_of_rooms_input=st.slider("How many rooms are there?",min_value=min(number_of_rooms_list),max_value=max(number_of_rooms_list),
                                step=0.5,key='slider_3')

year_constructed_slider_input=st.number_input("In which year was the flat constructed?",min_value=min(year_constructed_list),
                                              max_value=max(year_constructed_list),step=1.0,value=None, placeholder="Type a number")
      
thermal_char_input=st.number_input("What is the Energy Consumption value (kwh/m2a)?",min_value=min(thermal_char_list),max_value=max(thermal_char_list),
                                   value=None, placeholder="Type a number")
floor_input=st.number_input("Which floor is the flat in?",min_value=min(floor_list),max_value=max(floor_list),step=1.0,
                            value=None, placeholder="Type a number")
size_per_room_result=living_space_input/number_of_rooms_input


if st.button("Predict total rent"): 
       user_input_data= pd.DataFrame ({'heatingType':[heatingtype_input], 'newlyConst':[newly_const_input], 'balcony':[balcony_input],
        'pricetrend':[price_trend_slider_input], 'yearConstructed':[year_constructed_slider_input],'hasKitchen':[has_kitchen_input], 
        'cellar':[has_cellar_input], 'livingSpace':[living_space_input], 'condition':[condition_input], 'interiorQual':[interior_qual_input],
       'petsAllowed':[pets_allowed_input], 'lift':[lift_input], 'typeOfFlat':[type_of_flat_input], 'geo_plz':[geo_plz_input], 
       'noRooms':[number_of_rooms_input],'thermalChar':[thermal_char_input], 'floor':[floor_input], 'garden':[garden_input],
       'SizePerRoom':[size_per_room_result],'city':[city_input]})

       col_list=[heatingtype_input,newly_const_input,balcony_input,price_trend_slider_input,year_constructed_slider_input,
                 has_cellar_input,living_space_input,condition_input,interior_qual_input,pets_allowed_input,lift_input,type_of_flat_input,
                 geo_plz_input,number_of_rooms_input,thermal_char_input,floor_input,garden_input,size_per_room_result,city_input]
       
       if None in col_list:
              st.error("Please fill all the boxes, because there is at least one box that has been left unfilled!")
       else:
              pred = model.predict(user_input_data)
              pred=pred[0]
              st.success(f'The predicted total rent: {np.round(pred,decimals=0)} euros')