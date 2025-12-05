from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn

app = FastAPI(title="Cricket Prediction API")

# Load model
model = joblib.load('cricket_rf_model.joblib')

PLAYERS = {
    'Virat Kohli': 0, 'Rohit Sharma': 1, 'Suryakumar Yadav': 2, 'KL Rahul': 3,
    'Jos Buttler': 4, 'Babar Azam': 5, 'Glenn Maxwell': 6, 'Hardik Pandya': 7
}

class PredictionRequest(BaseModel):
    player_name: str
    recent_innings: List[float] = [25, 30, 22]
    career_avg: float = 28.0

@app.post("/predict")
async def predict(request: PredictionRequest):
    if request.player_name not in PLAYERS:
        raise HTTPException(status_code=400, detail="Player not found")
    
    player_id = PLAYERS[request.player_name]
    prev_runs = request.recent_innings[-1]
    roll_avg3 = np.mean(request.recent_innings[-3:])
    roll_avg5 = np.mean(request.recent_innings[-5:])
    
    features = np.array([[player_id, prev_runs, roll_avg3, roll_avg5, request.career_avg]])
    prediction = model.predict(features)[0]
    
    return {
        "player": request.player_name,
        "predicted_runs": float(prediction),
        "confidence_interval": [prediction-11.4, prediction+11.4],  # MAE-based
        "features": {
            "player_id": int(player_id),
            "prev_runs": float(prev_runs),
            "roll_avg3": float(roll_avg3),
            "roll_avg5": float(roll_avg5),
            "career_avg": float(request.career_avg)
        }
    }

@app.get("/players")
async def get_players():
    return {"players": list(PLAYERS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
