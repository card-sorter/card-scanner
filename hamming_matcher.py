import numpy as np
import aiosqlite

class HammingMatcher:
    def __init__(self):
        self.ids = None
        self.hashes = None
        self.num_parts = 0

    async def load_from_db(self, db_path: str, table: str = "hashes"):
        """
        Load all hashes into memory as numpy arrays.
        Automatically detects number of bigint columns
        """
        async with aiosqlite.connect(db_path) as db:
            # Detect bigint columns
            cursor = await db.execute(f"PRAGMA table_info({table})")
            cols = await cursor.fetchall()
            bigint_cols = [c[1] for c in cols if c[1].startswith("Bigint")]
            bigint_cols.sort()
            self.num_parts = len(bigint_cols)

            if self.num_parts == 0:
                raise Exception(f"No bigint columns found in table {table}")

            # Fetch all IDs and hash parts
            query = f"SELECT Card_ID, {', '.join(bigint_cols)} FROM {table}"
            cursor = await db.execute(query)
            rows = await cursor.fetchall()

        if not rows:
            raise Exception(f"No data found in table {table}")

        # Safely clip all bigint values to 64-bit range
        ids = []
        hashes = []
        for r in rows:
            ids.append(r[0])
            # Each bigint value is forced into unsigned 64-bit range
            safe_vals = [int(v) & ((1 << 64) - 1) for v in r[1:1 + self.num_parts]]
            hashes.append(safe_vals)

        self.ids = np.array(ids, dtype=np.uint32)
        self.hashes = np.array(hashes, dtype=np.uint64)

        print(f"Loaded {len(self.ids)} hashes ({self.num_parts} parts each).")

    def find_top_3(self, query_hash):
        """
        Find 3 closest matches for one query hash of fixed length
        """
        if self.ids is None or self.hashes is None:
            raise Exception("Must call load_from_db() first")

        if len(query_hash) != self.num_parts:
            raise ValueError(f"Query hash must have {self.num_parts} parts, got {len(query_hash)}")

        target = np.array(query_hash, dtype=np.uint64)
        xor_result = self.hashes ^ target

        # Support both NumPy >= 2.0 and < 2.0
        if hasattr(np, "bitwise_count"):
            distances = np.bitwise_count(xor_result).sum(axis=1)
        else:
            distances = np.zeros(len(self.hashes), dtype=np.int32)
            for i in range(self.num_parts):
                bits = np.unpackbits(xor_result[:, i].view(np.uint8))
                distances += bits.reshape(-1, 64).sum(axis=1)

        top3_idx = np.argpartition(distances, 3)[:3]
        top3_idx = top3_idx[np.argsort(distances[top3_idx])]

        return [(int(self.ids[i]), int(distances[i])) for i in top3_idx]

    def find_top_3_batch(self, query_hashes):
        """
        Process multiple queries efficiently
        """
        if self.ids is None or self.hashes is None:
            raise Exception("Must call load_from_db() first")

        return [self.find_top_3(q) for q in query_hashes]
