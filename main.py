import asyncio
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from card_scanner import CardScanner
from db_interface import DBInterface
from config import DATABASE, MODEL_PATH, PORT, HOSTNAME
from PIL import Image

scanner = CardScanner()
db_interface = DBInterface()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Starting up")
    await scanner.load_model()  
    await db_interface.open()
    print("startup complete")
    yield
    # shutdown
    await db_interface.close()
    print("shutdown complete")

app = FastAPI(title="Scanner API", lifespan=lifespan)

@app.post("/scan")
async def scan_card(image: UploadFile = File(...)):
    try:
        print(f"processing: {image.filename}")
        
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="must be an image")
        
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        print(f"Image loaded: {pil_image.size}")
        
        cards = await scanner.scan_cards(pil_image)
        print(f"Found {len(cards)} cards")
        
        results = []
        for card in cards:
            matched_cards = await db_interface.find_cards(card)
            
            card_data = {
                "scanned_hash": card.hash,
                "scanned_hash_bigints": card.hash_bigints,
                "matches_found": len(matched_cards)
            }
            
            if matched_cards:
                best_match = matched_cards[0]
                card_data.update({
                    "best_match": {
                        "card_id": best_match.values.get('card_id'),
                        "matched_filename": best_match.values.get('filename'),
                        "matched_hash_string": best_match.values.get('matched_hash'),
                        "hamming_distance": best_match.distance,
                        "confidence_percentage": best_match.values.get('confidence'),
                        "is_high_confidence": best_match.values.get('confidence', 0) > 90.0,
                        "card_details": best_match.values.get('card_details')
                    },
                    "all_matches": [
                        {
                            "card_id": match.values.get('card_id'),
                            "filename": match.values.get('filename'),
                            "confidence": match.values.get('confidence')
                        }
                        for match in matched_cards
                    ]
                })
            
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

# new endpoints
@app.get("/categories")
async def get_categories():
    """Get available card game categories"""
    try:
        categories = await db_interface.get_categories()
        return {
            "categories": categories,
            "source": "https://tcgcsv.com/tcgplayer/categories"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching categories: {str(e)}")

@app.get("/columns/{group_id}")
async def get_group_columns(group_id: int):
    """Get CSV column headers for a specific game group"""
    try:
        columns = await db_interface.get_group_columns(group_id)
        return {
            "group_id": group_id,
            "columns": columns,
            "source": f"https://tcgcsv.com/{group_id}/cards"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching columns for group {group_id}: {str(e)}")

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
        "health": "/health",
        "categories": "/categories",
        "columns": "/columns/{group_id}"
    }

@app.get("/test-db-connection")
async def test_db_connection():
    """Test database connection and sample query"""
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