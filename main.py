import asyncio
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from card_scanner import CardScanner
from db_interface import DBInterface
from config import DATABASE, MODEL_PATH, PORT, HOSTNAME
from PIL import Image

# Initialize components
scanner = CardScanner()
db_interface = DBInterface()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up")
    await scanner.load_model()
    await db_interface.open()
    print("startup complete")
    yield
    # Shutdown
    await db_interface.close()
    print("shutdown complete")

app = FastAPI(title="Scanner API", lifespan=lifespan)

@app.post("/scan")
async def scan_card(image: UploadFile = File(...)):
    try:
        print(f"processing: {image.filename}")
        
        if not image.content_type.startswith('image/'): #Image validation
            raise HTTPException(status_code=400, detail="must be an image")
        
        image_data = await image.read() #reading image
        pil_image = Image.open(io.BytesIO(image_data))
        print(f"Image loaded: {pil_image.size}")
        
        cards = await scanner.scan_cards(pil_image) # Scan cards
        print(f"Found {len(cards)} cards")
        
        results = [] # process results with database matching
        for card in cards:
            match = await db_interface.find_closest_card_match(card.hash_bigints)
            
            card_data = {
                "scanned_hash": card.hash,
                "scanned_hash_bigints": card.hash_bigints,
                "match_found": match is not None
            }
            
            if match:
                card_data.update({
                    "card_id": match['card_id'],
                    "matched_filename": match['filename'],
                    "matched_hash_string": match['hash_string'],
                    "matched_bigint": match['bigint_full_64bit'],
                    "hamming_distance": match['hamming_distance'],
                    "confidence_percentage": match['confidence_percentage'],
                    "is_high_confidence": match['confidence_percentage'] > 90.0  # 90%+ confidence
                })
                
                card_details = await db_interface.get_card_details(match['card_id'])
                if card_details:
                    card_data["card_details"] = card_details
            
            results.append(card_data)
        
        return {
            "cards_found": len(cards),
            "cards": results
        }
        
    except Exception as e:
        print(f"Err in /scan: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Err processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": scanner.model is not None,
        "db_connected": db_interface.connected
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Card Scanner API", 
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/test-db-connection")
async def test_db_connection():
    """Test database connection and sample query"""  # Test a simple query to verify database structure
    try:
        cursor = await db_interface._execute("SELECT name FROM sqlite_master WHERE type='table'")
        if cursor:
            tables = await cursor.fetchall()
            return {
                "database_connected": True,
                "tables": [table[0] for table in tables]
            }
        return {"database_connected": False}
    except Exception as e:
        return {"database_connected": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=HOSTNAME, port=int(PORT))