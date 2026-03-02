#Project Description
This system is designed for smart campuses. The problem with large campuses is that there are several classrooms unused. Large buildings require large air conditioning systems. Usually, air conditioning systems are either manually controlled, or the system is the same for the whole building. This causes the air conditioning to spend extra energy even in empty rooms. On the other hand, if a room becomes suddenly occupied, without good ventilation, the room might feel like it’s too hot and lacking air quality.  By estimating the occupancy of the rooms, the air conditioning system could be optimized by focusing it on rooms that are most likely to be occupied.

```
AI component workflow:
Dataset
   |
   v
Data ingestion
   |
   v
Data preprocessing
   ├── Data cleaning
   └── Data validation
   |
   v
Training the AI model  <--- Decision Tree Classifier
   |
   v
Hyperparameter tuning (loop)
   |
   v
Evaluation
   |
   v
Deployment
   ^
   |
Monitoring / retraining
````

#How to run the project

1. Clone repository
2. Create virtual enviroment
3. Install libraries/tools using pip install -r requirements.txt in project folder (includes pandas, numpy, flask, joblib, sklearn, matplotlib)
4. Excecute all cells in main.ipynb (not needet if there is occupancy_model.pkl in project folder)
5. Run app.py
6. Application start locally in http://127.0.0.1:5000
