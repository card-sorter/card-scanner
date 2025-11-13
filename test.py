import asyncio
from common import Card
from db_interface import DBInterface

async def test_find_cards():
    db = DBInterface()
    if not await db.open():
        print("Failed to open database.")
        return

    # Create a test Card with fake 14 bigint hash parts
    compare_card = Card(None)
    compare_card.hash_bigints = [
        1234567890123456789, 9876543210987654321, 1111111111111111111,
        2222222222222222222, 3333333333333333333, 4444444444444444444,
        5555555555555555555, 6666666666666666666, 7777777777777777777,
        8888888888888888888, 9999999999999999999, 1010101010101010101,
        1212121212121212121, 1414141414141414141
    ]

    print("Searching for similar cards...")
    results = await db.find_cards(compare_card)

    for card in results:
        print(f"\nCard: {card.values.get('name')}")
        print(f"ID: {card.values.get('productId')}")
        print(f"Image URL: {card.values.get('imageUrl')}")
        print(f"Distance: {card.distance}")

    await db.close()

if __name__ == "__main__":
    asyncio.run(test_find_cards())
