from fastapi import FastAPI

app = FastAPI(title="Mock Pharmacy API")

STOCK = {
    "парацетамол": 30,
    "ибупрофен": 24,
    "аспирин": 0,
    "найз": 12,
    "кеторол": 8,
    "кларитин": 15,
    "зиртек": 20,
    "эриус": 10,
    "супрастин": 6,
    "клемастин": 9,
    "амоксициллин": 14,
    "сумамед": 7,
    "ципрофлоксацин": 11,
    "амоксиклав": 5,
    "доксициклин": 13,
    "эналаприл": 18,
    "бисопролол": 16,
    "амлодипин": 19,
    "лозартан": 17,
    "аторвастатин": 21,
    "омепразол": 22,
    "зантак": 4,
    "мотилиум": 10,
    "смекта": 25,
    "афобазол": 12,
    "глицин": 17,
    "мелатонин": 8,
    "терафлю": 9,
    "ацц": 14,
    "амбробене": 18,
}

PRICES_RUB = {
    "парацетамол": 149,
    "ибупрофен": 189,
    "аспирин": 99,
    "найз": 299,
    "кеторол": 249,
    "кларитин": 349,
    "зиртек": 329,
    "эриус": 399,
    "супрастин": 89,
    "клемастин": 119,
    "амоксициллин": 199,
    "сумамед": 499,
    "ципрофлоксацин": 279,
    "амоксиклав": 450,
    "доксициклин": 219,
    "эналаприл": 159,
    "бисопролол": 199,
    "амлодипин": 179,
    "лозартан": 249,
    "аторвастатин": 349,
    "омепразол": 199,
    "зантак": 189,
    "мотилиум": 229,
    "смекта": 199,
    "афобазол": 299,
    "глицин": 129,
    "мелатонин": 349,
    "терафлю": 399,
    "ацц": 179,
    "амбробене": 159,
}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/stock/{drug_name}")
def get_stock(drug_name: str) -> dict:
    normalized_drug = drug_name.strip().lower()

    if normalized_drug not in STOCK:
        return {
            "drug": normalized_drug,
            "found": False,
            "available": False,
            "quantity": 0,
        }

    quantity = STOCK[normalized_drug]

    return {
        "drug": normalized_drug,
        "found": True,
        "available": quantity > 0,
        "quantity": quantity,
    }

@app.get("/price/{drug_name}")
def get_price(drug_name: str) -> dict:
    normalized_drug = drug_name.strip().lower()
    
    if normalized_drug not in PRICES_RUB:
        return {
            "drug": normalized_drug,
            "found": False,
            "price": None,
            "currency": "RUB",
        }

    return {
        "drug": normalized_drug,
        "found": True,
        "price": PRICES_RUB[normalized_drug],
        "currency": "RUB",
    }