from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)


## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            lead_time = int(request.form.get('lead_time')),
            arrival_date_year = int(request.form.get('arrival_date_year')),
            # arrival_date_week_number = int(request.form.get('arrival_date_week_number')),
            arrival_date_day_of_month = int(request.form.get('arrival_date_day_of_month')),
            stays_in_weekend_nights = int(request.form.get('stays_in_weekend_nights')),
            adults = int(request.form.get('adults')),
            total_kids = (request.form.get('total_kids')),
            # babies = int(request.form.get('babies')),
            is_repeated_guest = int(request.form.get('is_repeated_guest')),
            previous_cancellations = int(request.form.get('previous_cancellations')),
            # booking_changes = int(request.form.get('booking_changes')),
            agent = request.form.get('agent'),
            days_in_waiting_list = int(request.form.get('days_in_waiting_list')),
            adr = float(request.form.get('adr')),
            required_car_parking_spaces = int(request.form.get('required_car_parking_spaces')),
            # total_of_special_requests = int(request.form.get('total_of_special_requests')),
            hotel = request.form.get('hotel'),
            arrival_date_month = request.form.get('arrival_date_month'),
            meal = request.form.get('meal'),
            country = request.form.get('country'),
            market_segment = request.form.get('market_segment'),
            # distribution_channel = request.form.get('distribution_channel'),
            reserved_room_type = request.form.get('reserved_room_type'),
            assigned_room_type = request.form.get('assigned_room_type'),
            deposit_type = request.form.get('deposit_type'),
            customer_type = request.form.get('customer_type'),
            # reservation_status = request.form.get('reservation_status'),


        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


