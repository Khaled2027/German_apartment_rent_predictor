import streamlit as st 
import joblib
import pandas as pd
import numpy as np


st.title("German Apartment Rent Prediction App")
st.markdown("Fill in the apartment details below to get the estimated total rent")
st.divider()

# Loads the data
@st.cache_data
def load_data(data):
       df=pd.read_csv(data)
       return df
df=load_data("train_data.csv")

# Loads the best model
@st.cache_resource
def load_model(model):
       model=joblib.load(model)
       return model
model = load_model('best_model_compressed.joblib')

df_city_and_plz=df[['city','geo_plz']]

# Returns a dataframe grouped by 'city' with a list of 'geo_plz' values
df_city_and_plz=df_city_and_plz.groupby('city')['geo_plz'].apply(list)
# Converts a Dataframe to a dictionary and returns a dicitonary 
city_and_plz_dict=df_city_and_plz.to_dict()

# Takes a dataframe column and returns an array with unique values
newly_const_list=df['newlyConst'].unique()
balcony_list=df['balcony'].unique()
price_trend_list=df['pricetrend'].sort_values(ascending=True).unique()

year_constructed_list=df['yearConstructed'].sort_values(ascending=True).unique()
has_kitchen_list=df['hasKitchen'].unique()
has_cellar_list=df['cellar'].unique()

living_space_list=df['livingSpace'].unique()
lift_list=df['lift'].unique()
geo_plz_list=df['geo_plz'].unique()

number_of_rooms_list=df['noRooms'].unique()
thermal_char_list=df['thermalChar'].unique()
floor_list=df['floor'].unique()

size_per_room_list=df['SizePerRoom'].unique()
garden_list=df['garden'].unique()
city_list=df['city'].unique()

# Takes a dataframe column and returns an list with unique values
condition_list=df['condition'].unique().tolist()
type_of_flat_list=df['typeOfFlat'].unique().tolist()
heating_type_list=df['heatingType'].unique().tolist()
interior_qual_list=df['interiorQual'].unique().tolist()
pets_allowed_list=df['petsAllowed'].unique().tolist()

# Takes one categorical column and removes given value 
def remove_value(cols,value):
       cols.remove(value)

remove_value(heating_type_list,'unknown')
remove_value(condition_list,'unknown')
remove_value(interior_qual_list,'unknown')
remove_value(pets_allowed_list,'unknown')
remove_value(type_of_flat_list,'unknown')

# Creates a selectbox with a header on top and options from a given list
city_input=st.selectbox("In which city is the flat in? (Please select a city)",city_list)
geo_plz_input=st.selectbox("What is the flats Postal code (Postleitzahl)?",options=list(set(city_and_plz_dict[city_input])),index=None,placeholder="Select one of the options")
heatingtype_input=st.selectbox("What is the Heating Type?",heating_type_list,index=None,placeholder="Select one of the options")

newly_const_input=st.selectbox("The flats building is newly constructed?",newly_const_list,index=None,placeholder="Select one of the options")
balcony_input=st.selectbox("has a balcony?",balcony_list,index=None,placeholder="Select one of the options")
has_kitchen_input=st.selectbox("has a built-in kitchen?",has_kitchen_list,index=None,placeholder="Select one of the options")

has_cellar_input=st.selectbox("has a cellar?",has_cellar_list,index=None,placeholder="Select one of the options")
condition_input=st.selectbox("What is the flats condition?",condition_list,index=None,placeholder="Select one of the options")
interior_qual_input=st.selectbox("What is the flats interior quality?",interior_qual_list,index=None,placeholder="Select one of the options")

pets_allowed_input=st.selectbox("Are pets allowed?",pets_allowed_list,index=None,placeholder="Select one of the options")
lift_input=st.selectbox("has a lift?",lift_list,index=None,placeholder="Select one of the options")
garden_input=st.selectbox("has a garden?",garden_list,index=None,placeholder="Select one of the options")

type_of_flat_input=st.selectbox("What is the flat type?",type_of_flat_list,index=None,placeholder="Select one of the options")

# Creates a slider with a header on top and a range of values to choose from
living_space_input=st.slider("How much is the living space (in sqm)?",min_value=min(living_space_list),
                                    max_value=max(living_space_list),key='slider_1')
price_trend_slider_input = st.slider("Price Trend",min_value=min(price_trend_list),max_value=max(price_trend_list),key='slider_2')
number_of_rooms_input=st.slider("How many rooms are there?",min_value=min(number_of_rooms_list),max_value=max(number_of_rooms_list),
                                step=0.5,key='slider_3')
# Creates a field where the user can enter a number within a specfic range
year_constructed_input=st.number_input("In which year was the flat constructed?",min_value=min(year_constructed_list),
                                              max_value=max(year_constructed_list),step=1.0,value=None, placeholder="Type a number")

thermal_char_input=st.number_input("What is the Energy Consumption value (kwh/m2a)?",min_value=min(thermal_char_list),max_value=max(thermal_char_list),
                                   value=None, placeholder="Type a number")
floor_input=st.number_input("Which floor is the flat in?",min_value=min(floor_list),max_value=max(floor_list),step=1.0,
                            value=None, placeholder="Type a number")

size_per_room_result=living_space_input/number_of_rooms_input

# Executes if the user clicks on the button 
if st.button("Predict total rent"): 
       # stores the user input in a a dictonary covnerted to a dataframe 
       user_input_data= pd.DataFrame ({'heatingType':[heatingtype_input], 'newlyConst':[newly_const_input], 'balcony':[balcony_input],
        'pricetrend':[price_trend_slider_input], 'yearConstructed':[year_constructed_input],'hasKitchen':[has_kitchen_input], 
        'cellar':[has_cellar_input], 'livingSpace':[living_space_input], 'condition':[condition_input], 'interiorQual':[interior_qual_input],
       'petsAllowed':[pets_allowed_input], 'lift':[lift_input], 'typeOfFlat':[type_of_flat_input], 'geo_plz':[geo_plz_input], 
       'noRooms':[number_of_rooms_input],'thermalChar':[thermal_char_input], 'floor':[floor_input], 'garden':[garden_input],
       'SizePerRoom':[size_per_room_result],'city':[city_input]})

       col_list=[heatingtype_input,newly_const_input,balcony_input,price_trend_slider_input,year_constructed_input,has_kitchen_input,
                 has_cellar_input,living_space_input,condition_input,interior_qual_input,pets_allowed_input,lift_input,type_of_flat_input,
                 geo_plz_input,number_of_rooms_input,thermal_char_input,floor_input,garden_input,size_per_room_result,city_input]

       # if the user leaves a field empty, they will recive an error
       if None in col_list:
              st.error("Please fill in all the fields.")
       # if the user has filled all the fields, they will get the estimate total rent
       else:
              pred = model.predict(user_input_data)
              pred=pred[0]
              st.success(f'The predicted total rent: {np.round(pred,decimals=0)} euros')