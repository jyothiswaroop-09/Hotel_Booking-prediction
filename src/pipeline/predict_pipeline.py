import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
        

    def predict(self,features):
        try:
            
            model_path=r"D:\DATA SCIENCE\PYTHON\Git and Github\Hotel_Booking prediction\src\components\artifacts\model.pkl"
            preprocessor_path=r'D:\DATA SCIENCE\PYTHON\Git and Github\Hotel_Booking prediction\src\components\artifacts\preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        # try:
        #     #model_path=os.path.join("src","components","artifacts", "model.pkl")
        #     model_path="src\components\artifacts\model.pkl"
        #     preprocessor_path=os.path.join("src","components","artifacts","preprocessor.pkl")
        #     self.model=load_object(model_path)
        #     self.preprocessor=load_object(preprocessor_path)
        #     print(f" Model path: {model_path}")
        #     print(f" Preprocessor path: {preprocessor_path}")
        
        # data_scaled=self.preprocessor.transform(features)
        # preds=self.model.predict(data_scaled)
        # print("Data transformed and prediction made.")
        # return preds
            
        # except Exception as e:
        #     raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        lead_time: int,
        arrival_date_year: int,
        # arrival_date_week_number: int,
        arrival_date_day_of_month: int,
        stays_in_weekend_nights: int,
        adults: int,
        total_kids: int,
        # babies: int,
        is_repeated_guest: int,
        previous_cancellations: int,
        # booking_changes: int,
        agent: str,
        days_in_waiting_list: int,
        adr: float,
        required_car_parking_spaces: int,
        # total_of_special_requests: int,
        hotel: str,
        arrival_date_month: str,
        meal: str,
        country: str,
        market_segment: str,
        distribution_channel: str,
        reserved_room_type: str,
        assigned_room_type: str,
        deposit_type: str,
        customer_type: str,
        reservation_status: str
    ):

        self.lead_time = lead_time
        self.arrival_date_year = arrival_date_year
        # self.arrival_date_week_number = arrival_date_week_number
        self.arrival_date_day_of_month = arrival_date_day_of_month
        self.stays_in_weekend_nights = stays_in_weekend_nights
        self.adults = adults
        self.total_kids = total_kids
        # self.babies = babies
        self.is_repeated_guest = is_repeated_guest
        self.previous_cancellations = previous_cancellations
        # self.booking_changes = booking_changes
        self.agent = agent
        self.days_in_waiting_list = days_in_waiting_list
        self.adr = adr
        self.required_car_parking_spaces = required_car_parking_spaces
        # self.total_of_special_requests = total_of_special_requests
        self.hotel = hotel
        self.arrival_date_month = arrival_date_month
        self.meal = meal
        self.country = country
        self.market_segment = market_segment
        self.distribution_channel = distribution_channel
        self.reserved_room_type = reserved_room_type
        self.assigned_room_type = assigned_room_type
        self.deposit_type = deposit_type
        self.customer_type = customer_type
        self.reservation_status = reservation_status


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "lead_time": [self.lead_time],
                "arrival_date_year": [self.arrival_date_year],
                # "arrival_date_week_number": [self.arrival_date_week_number],
                "arrival_date_day_of_month": [self.arrival_date_day_of_month],
                "stays_in_weekend_nights": [self.stays_in_weekend_nights],
                "adults": [self.adults],
                "total_kids": [self.total_kids],
                # "babies": [self.babies],
                "is_repeated_guest": [self.is_repeated_guest],
                "previous_cancellations": [self.previous_cancellations],
                # "booking_changes": [self.booking_changes],
                "agent": [self.agent],
                "days_in_waiting_list": [self.days_in_waiting_list],
                "adr": [self.adr],
                "required_car_parking_spaces": [self.required_car_parking_spaces],
                # "total_of_special_requests": [self.total_of_special_requests],
                "hotel": [self.hotel],
                "arrival_date_month": [self.arrival_date_month],
                "meal": [self.meal],
                "country": [self.country],
                "market_segment": [self.market_segment],
                "distribution_channel": [self.distribution_channel],
                "reserved_room_type": [self.reserved_room_type],
                "assigned_room_type": [self.assigned_room_type],
                "deposit_type": [self.deposit_type],
                "customer_type": [self.customer_type],
                "reservation_status": [self.reservation_status],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

