# DCDP Unit 3 Project : Advanced Supervised Modelling

## __Guideline__

In this project, you will apply supervised learning techniques (Classification) to predict one of the following. 
1. **Loan Approval Outcomes** : The dataset includes demographic, financial, and credit history features of loan applicants.
2. **Airline Passenger Satisfaction** : Customer satisfaction scores from 120,000+ airline passengers, including additional information about each passenger, their flight, and type of travel, as well as ther evaluation of different factors like cleanliness, comfort, service, and overall experience.
3. **Hotel Booking Cancellations** : booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things

- Your task : 
  - Explore the data through EDA 
  - Clean and preprocess it as needed, create new columns (Feature Engineering)
  - Build classification models using **Logistic Regression** and **KNN**. 
- After training all models, evaluate their performance using relevant metrics such as accuracy, precision, recall, F1-score, and confusion matrix. 
- Improve all models by tuning the parameters using `GridsearchCV`,applying `Regularization` ,and streamline the process using `Pipelines`. 
- In total you should have built atleast **4 models** ; 2 Basic models, and 2 of hyperparameter tuned models. 
- Conclude with a comparative analysis of the models, discussing which performed better and why. 
- Your Jupyter notebook should have **clear comments** and **markdown explanations of each of your findings, decision and approach taken**.
- In addition to Jupyter notebook, You should also create a presentation slide deck, including insights from your EDA, preprocessing decisions, and modeling rationale. The end user of this report are **Non Technical Audience**. Hence, it should not be technical. 
- This lab emphasizes technical execution and clear documentation of your process and findings to non technical audience.


## __Data Dictionary__
### Loan Approval Dataset
| Column                           | Description                                             | 
|----------------------------------|---------------------------------------------------------|
| person_age                       | Age of the person                                       |
| person_gender                    | Gender of the person                                    |
| person_education                 | Highest education level                                 |
| person_income                    | Annual income                                           | 
| person_emp_exp                   | Years of employment experience                          | 
| person_home_ownership            | Home ownership status (e.g., rent, own, mortgage)       | 
| loan_amnt                        | Loan amount requested                                   |
| loan_intent                      | Purpose of the loan                                     | 
| loan_int_rate                    | Loan interest rate                                      | 
| loan_percent_income              | Loan amount as a percentage of annual income            | 
| cb_person_cred_hist_length       | Length of credit history in years                       |
| credit_score                     | Credit score of the person                              |
| previous_loan_defaults_on_file   | Indicator of previous loan defaults                     |
| loan_status (target variable)    | Loan approval status: 1 = approved; 0 = rejected        |

---

### Airline Passenger Satisfaction

Customer satisfaction scores from 120,000+ airline passengers, including additional information about each passenger, their flight, and type of travel, as well as ther evaluation of different factors like cleanliness, comfort, service, and overall experience.

| Column | Description |
|----------|-------------|
| ID | Unique passenger identifier |
| Gender | Gender of the passenger (Female/Male) |
| Age | Age of the passenger |
| Customer Type | Type of airline customer (First-time/Returning) |
| Type of Travel | Purpose of the flight (Business/Personal) |
| Class | Travel class of the passenger seat |
| Flight Distance | Flight distance in miles |
| Departure Delay | Flight departure delay in minutes |
| Arrival Delay | Flight arrival delay in minutes |
| Departure and Arrival Time Convenience | Satisfaction level with flight departure and arrival time convenience from 1 (lowest) to 5 (highest); 0 = not applicable |
| Ease of Online Booking | Satisfaction level with online booking experience from 1 (lowest) to 5 (highest); 0 = not applicable |
| Check-in Service | Satisfaction level with check-in service from 1 (lowest) to 5 (highest); 0 = not applicable |
| Online Boarding | Satisfaction level with online boarding experience from 1 (lowest) to 5 (highest); 0 = not applicable |
| Gate Location | Satisfaction level with airport gate location from 1 (lowest) to 5 (highest); 0 = not applicable |
| On-board Service | Satisfaction level with on-board service from 1 (lowest) to 5 (highest); 0 = not applicable |
| Seat Comfort | Satisfaction level with seat comfort from 1 (lowest) to 5 (highest); 0 = not applicable |
| Leg Room Service | Satisfaction level with seat leg room from 1 (lowest) to 5 (highest); 0 = not applicable |
| Cleanliness | Satisfaction level with airplane cleanliness from 1 (lowest) to 5 (highest); 0 = not applicable |
| Food and Drink | Satisfaction level with food and drinks from 1 (lowest) to 5 (highest); 0 = not applicable |
| In-flight Service | Satisfaction level with in-flight service from 1 (lowest) to 5 (highest); 0 = not applicable |
| In-flight Wifi Service | Satisfaction level with in-flight WiFi service from 1 (lowest) to 5 (highest); 0 = not applicable |
| In-flight Entertainment | Satisfaction level with in-flight entertainment from 1 (lowest) to 5 (highest); 0 = not applicable |
| Baggage Handling | Satisfaction level with baggage handling from 1 (lowest) to 5 (highest); 0 = not applicable |
| Satisfaction | Overall satisfaction level with the airline (Satisfied / Neutral or Unsatisfied) |

### Hotel Booking

This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

The data is originally from the article [Hotel Booking Demand Datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191), written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019.

| Column | Description |
|----------|-------------|
| hotel | Resort Hotel or City Hotel |
| is_canceled (target variable) | Value indicating if the booking was canceled (1) or not (0) |
| lead_time | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date |
| arrival_date_year | Year of arrival date |
| arrival_date_month | Month of arrival date |
| arrival_date_week_number | Week number of year for arrival date |
| arrival_date_day_of_month | Day of arrival date |
| stays_in_weekend_nights | Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel |
| stays_in_week_nights | Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel |
| adults | Number of adults |
| children | Number of children |
| babies | Number of babies |
| meal | Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner) |
| country | Country of origin. Categories are represented in the ISO 3155–3:2013 format |
| market_segment | Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators” |
| distribution_channel | Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators” |
| is_repeated_guest | Value indicating if the booking name was from a repeated guest (1) or not (0) |
| previous_cancellations | Number of previous bookings that were cancelled by the customer prior to the current booking |
| previous_bookings_not_canceled | Number of previous bookings not cancelled by the customer prior to the current booking |
| reserved_room_type | Code of room type reserved. Code is presented instead of designation for anonymity reasons. |
| assigned_room_type | Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons. |
| booking_changes | Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation |
| deposit_type | Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay. |
| agent | ID of the travel agency that made the booking |
| company | ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons |
| days_in_waiting_list | Number of days the booking was in the waiting list before it was confirmed to the customer |
| customer_type | Type of booking, assuming one of four categories: Contract - when the booking has an allotment or other type of contract associated to it; Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking |
| adr | Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights |
| required_car_parking_spaces | Number of car parking spaces required by the customer |
| total_of_special_requests | Number of special requests made by the customer (e.g. twin bed or high floor) |
| reservation_status | Reservation last status, assuming one of three categories: Canceled – booking was canceled by the customer; Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why |
| reservation_status_date | Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel |

---

## Deliverables:
-  **Submission Deadline: 7th June 2026 (Sunday @ 06:00 PM)**.
-  A complete Jupyter notebook that is your own work and is well commented at every steps. **Please provide explanations of your feature enginnering, feature selection and modelling approach.** This could be as technical as possible. 
-  A Presentation Slide Deck that has the following components :
   - **Problem Statement** : Clearly define the machine learning task (Supervised Binary Classification), identify the target and features, and determine the mathematical success criteria (e.g., optimizing for Precision/Recall/F1 over Accuracy).
   - **EDA findings** - Screenshots of tables and charts and explanation, feature engineering and feature selection. This should be non technical. Use simple plain words for the readers to understand. 
     - **Do not add screenshots of your codes** as that is considered to be technical. 
   - **Base Modelling approach** - The models built and brief description of the model
   - **Model performance** - Accuracy, Precision, Recall and F1-Score for every model built. Explain what your metrics mean in simple language.
   - **Model Improvement** - Hyperparamater Tuning using `GridSearchCV` results approach and improved model performance.
   - **Model Performance Comparison** - Comparison between performance of different models.  
   - **Conclusion & Recommendations** - Actionable business advice based on your findings.



**Note:** The goal of this project is to see how you interpret and apply these concepts in a real-world context. To demonstrate your understanding, all answers must be original and written in your own words. While AI and external research can be used for brainstorming or improving phrasing your answers, **direct plagiarism or generating entire responses using AI is not permitted and will significantly impact your final evaluation.**