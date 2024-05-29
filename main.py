from fastapi import FastAPI
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome To Tuwaiq Academy!"}

@app.get("/items/")
def create_item(item: dict):
    return{"item": item}


import joblib
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')          

from pydantic import BaseModel
# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

def preprocessing(input_features: InputFeatures):
    dict_f = {
'Year': input_features.Year,
'Engine_Size': input_features.Engine_Size,
'Mileage': input_features.Mileage,
'Type_Accent': input_features.Type == 'Accent',
'Type_Land Cruiser': input_features.Type == 'LandCruiser',
'Make_Hyundai': input_features.Make == 'Hyundai',
'Make_Mercedes': input_features.Make == 'Mercedes',
'Options_Full': input_features.Options == 'Full',
'Options_Standard': input_features.Options == 'Standard'
}
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values
    ())])
    return scaled_features


@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)

@app.post("/predict")
async def predict(input_features: InputFeatures):
    dict_f = preprocessing(input_features)
    y_pred = model.predict(dict_f)
    return {"pred": y_pred.tolist()[0]}
