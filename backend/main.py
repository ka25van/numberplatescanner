from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import easyocr
import logging
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from utils.plate_detector import preprocess_image

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client.vehicle_database
print(f"DB", db)
vehicles_collection = db.vehicles

reader = None
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Failed to initialize EasyOCR: {e}")

@app.post("/api/scan-plate")
async def scan_plate(file: UploadFile = File(...)):
    global reader
    try:
        if reader is None:
            reader = easyocr.Reader(['en'], gpu=False)
            
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image"}

        processed_img = preprocess_image(img)
        
        # Save debug image
        cv2.imwrite('debug_processed.jpg', processed_img)
        
        results = reader.readtext(
            processed_img,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            # paragraph=False,
            # min_size=10,
            # low_text=0.3,
            # width_ths=0.7,
            # ycenter_ths=0.5
        )
        # print(f"Results", results)

        if not results:
            return {"error": "No text detected"}

        plate_number = results[0][1].upper().replace(" ", "")
        print(f"Detected plate: {plate_number}")

        vehicle = await vehicles_collection.find_one({"plate_number": plate_number})
        if not vehicle:
            return {"error": f"No vehicle found: {plate_number}"}

        return {
            "plate_number": vehicle["plate_number"],
            "owner": vehicle["owner"],
            "model": vehicle["model"],
            "year": vehicle["year"],
            "color": vehicle["color"],
            "registration_date": vehicle["registration_date"],
            "insurance_status": vehicle["insurance_status"],
            "last_service_date": vehicle["last_service_date"],
            "vehicle_type": vehicle["vehicle_type"],
            "fuel_type": vehicle["fuel_type"]
        }


    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}