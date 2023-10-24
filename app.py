import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
import random
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report



data_url = "https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time" 

# setting up the page streamlit

st.set_page_config(
    page_title="Streamlit Food Delivery Time App", layout="wide", page_icon="./images/linear-regression.png"
)


def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

    image_nyu = Image.open('images/nyu.png')
    st.image(image_nyu, width=100)
    
    st.title("Food Delivery Time 🍜")
    
    # navigation dropdown
    
    st.sidebar.header("Dashboard")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])
    select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Delivery Time"])
    df = pd.read_csv("deliverytime.csv")
    
    
    list_variables = df.columns
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)
    # page 1 
    if app_mode == 'Introduction':
        image_header = Image.open('./images/dataset-cover.jpg')
        st.image(image_header, width=600)
        
        
        st.markdown("### 00 - Show  Dataset")
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
        
        # Descriptions for each variable in the dataset
        # ... [The code here provides descriptions for each wine quality variable]
        
        # Allow users to view either the top or bottom rows of the dataset
        num = st.number_input('No. of Rows', 5, 10)
        head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
        if head == 'Head':
            st.dataframe(df.head(num))
        else:
            st.dataframe(df.tail(num))
        
        # Display the shape (number of rows and columns) of the dataset
        st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
        st.text('(Rows,Columns)')
        st.write(df.shape)
        
        st.markdown("### 01 - Description")
        st.dataframe(df.describe())
        
        
        
        st.markdown("### 02 - Missing Values")
        st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
        dfnull = df.isnull().sum()/len(df)*100
        totalmiss = dfnull.sum().round(2)
        st.write("Percentage of total missing values:",totalmiss)
        st.write(dfnull)
        if totalmiss <= 30:
            st.success("Looks good! as we have less then 30 percent of missing values.")
        else:
            st.warning("Poor data quality due to greater than 30 percent of missing value.")
            st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")
        
        st.markdown("### 03 - Completeness")
        st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.")
        # st.write("Total data length:", len(df))
        nonmissing = (df.notnull().sum().round(2))
        completeness= round(sum(nonmissing)/len(df),2)
        st.write("Completeness ratio:",completeness)
        st.write(nonmissing)
        if completeness >= 0.80:
            st.success("Looks good! as we have completeness ratio greater than 0.85.")
        
        else:
            st.success("Poor data quality due to low completeness ratio( less than 0.85).")
        
        st.markdown("### 04 - Complete Report")
        if st.button("Generate Report"):
        
            pr = df.profile_report()
            st_profile_report(pr)
    
    
    if app_mode == 'Visualization':
        st.markdown("## Visualization")
        symbols = st.multiselect("Select two variables",list_variables,["Delivery_person_Age", "Type_of_vehicle", "Time_taken(min)"])
        width1 = st.sidebar.slider("plot width", 1, 25, 10)
        #symbols = st.multiselect("", list_variables, list_variables[:5])
        tab1, tab2= st.tabs(["Line Chart","📈 Correlation"])    
        
        tab1.subheader("Line Chart")
        st.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)
        st.write(" ")
        st.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)
        
        labels = 'motorcycle', 'scooter', 'electric_scooter', 'bicycle'
        sizes = df["Type_of_vehicle"].value_counts()
        explode = (0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        st.pyplot(fig1)

        st.map(data=df, latitude='Restaurant_latitude', longitude='Restaurant_longitude')
        
        
        #tab2.subheader("Correlation Tab 📉")
        #fig,ax = plt.subplots(figsize=(width1, width1))
        #sns.heatmap(df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
        #tab2.write(fig)
        
        
        st.write(" ")
        st.write(" ")
        st.markdown("### Pairplot")
        
        df2 = df[[list_variables[0],list_variables[1],list_variables[2],list_variables[3],list_variables[4]]]
        fig3 = sns.pairplot(df2)
        st.pyplot(fig3)
    
    
    
    
    if app_mode == 'Prediction':
        st.markdown("## Prediction")
        train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
        new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
        list_var = df["Time_taken(min)"]
        X_train, X_test, y_train, y_test, predictions,x,y= predict(select_variable,train_size,new_df,list_var)
    
        st.subheader('🎯 Results')


        st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
        st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
        st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
        st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))
    
    st.markdown(" ")
    st.markdown("### 👨🏼‍💻 **App Contributors:** ")
    st.markdown("Emmanuella Abankwah, Yinyi Feng, Sayuri Hadge")
    st.markdown(f"####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone/Linear-Regression-App'}) 🚀 ")


def predict(target_choice,train_size,new_df,output_multi):
    #independent variables / explanatory variables
    #choosing column for target
    #new_df2 = new_df[["Delivery_person_Age", "Delivery_person_Ratings"]]
    x = new_df[["Delivery_person_Age", "Delivery_person_Ratings"]]
    y = new_df["Time_taken(min)"]
    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(x.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    
    return X_train, X_test, y_train, y_test, predictions,x,y
       




if __name__=='__main__':
    main()


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        "👨🏼‍💻 Made by ",
        link("https://github.com/NYU-DS-4-Everyone", "NYU - Professor Gaëtan Brison"),
        "🚀"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()
