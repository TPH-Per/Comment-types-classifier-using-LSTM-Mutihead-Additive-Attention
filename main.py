import pickle
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from model import BiLSTMSentiment
from preprocess import clean_text_v2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"
WEIGHTS_PATH = BASE_DIR / "bilstm_v22_weights.pth"
VOCAB_PATH = BASE_DIR / "vocab.pkl"
DB_PATH = BASE_DIR / "sentiment.db"

MAX_LEN = 64
LABEL_MAP = {0: "Tích cực", 1: "Tiêu cực", 2: "Trung tính"}

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
model: BiLSTMSentiment
vocab: dict[str, int]


def load_vocab(path: Path) -> dict[str, int]:
    with open(path, "rb") as f:
        return pickle.load(f)


def encode(tokens: list[str], vocab: dict[str, int], max_len: int):
    """Convert tokens to padded tensor + attention mask."""
    unk_id = vocab.get("<UNK>", 1)
    ids = [vocab.get(t, unk_id) for t in tokens][:max_len]
    real_len = len(ids)
    ids += [0] * (max_len - real_len)
    mask = [True] * real_len + [False] * (max_len - real_len)
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.bool),
    )


def predict_sentiment(text: str) -> tuple[str, float]:
    """Run the BiLSTM model on a single text and return (label, confidence)."""
    cleaned = clean_text_v2(text)
    if not cleaned:
        return "Trung tính", 0.0
    tokens = cleaned.split()
    input_ids, input_mask = encode(tokens, vocab, MAX_LEN)
    with torch.no_grad():
        logits, _ = model(input_ids, input_mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())
    return LABEL_MAP[pred_idx], round(probs[pred_idx].item(), 4)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables and seed sample data if the DB is fresh."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            price       REAL    NOT NULL,
            image_url   TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS comments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL,
            user_name   TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            label       TEXT    DEFAULT '',
            confidence  REAL    DEFAULT 0.0,
            created_at  TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
    """)

    # Only seed if products table is empty
    count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if count == 0:
        _seed_data(conn)

    conn.commit()
    conn.close()


def _seed_data(conn: sqlite3.Connection):
    """Insert sample products and comments, then run model to label them."""
    products = [
        ("iPhone 15 Pro Max 256GB", "Điện thoại", 34990000,
         "https://cdn.tgdd.vn/Products/Images/42/305658/iphone-15-pro-max-blue-thumbnew-600x600.jpg"),
        ("Samsung Galaxy S24 Ultra", "Điện thoại", 31990000,
         "https://cdn.tgdd.vn/Products/Images/42/307174/samsung-galaxy-s24-ultra-yellow-thumbnew-600x600.jpg"),
        ("MacBook Air M3 15 inch", "Laptop", 36990000,
         "https://cdn.tgdd.vn/Products/Images/44/314301/macbook-air-m3-15-inch-thumb-600x600.jpg"),
        ("Tai nghe Sony WH-1000XM5", "Phụ kiện", 8490000,
         "https://cdn.tgdd.vn/Products/Images/54/309005/sony-wh-1000xm5-den-thumb-600x600.jpg"),
        ("Áo thun nam cotton basic", "Thời trang", 199000,
         "https://cf.shopee.vn/file/sg-11134201-22120-3tz8y5qwzolv98"),
        ("Nồi chiên không dầu Philips", "Gia dụng", 2990000,
         "https://cdn.tgdd.vn/Products/Images/1943/235497/philips-na221-01-600x600.jpg"),
    ]
    conn.executemany(
        "INSERT INTO products (name, category, price, image_url) VALUES (?, ?, ?, ?)",
        products,
    )

    # Sample comments — realistic Vietnamese e-commerce reviews
    raw_comments = [
        # iPhone 15 Pro Max
        (1, "Nguyễn Minh", "Sản phẩm rất tốt, đóng gói cẩn thận, giao hàng nhanh. Camera chụp đẹp lung linh luôn!"),
        (1, "Trần Hương", "Máy dùng mượt mà, pin trâu hơn đời trước nhiều. Rất hài lòng với đơn hàng này."),
        (1, "Lê Đức", "Hàng đúng chính hãng, nhưng giá hơi cao so với thị trường. Nói chung ổn."),
        (1, "Phạm Linh", "Mình mua về dùng 1 tuần thì bị lỗi màn hình. Rất thất vọng, đã gửi bảo hành."),
        (1, "Hoàng Nam", "Đẹp, nhẹ, cầm rất thích tay. Face ID nhanh. Đáng tiền!"),
        # Samsung Galaxy S24 Ultra
        (2, "Vũ Hà", "S Pen viết rất nhạy, màn hình AMOLED quá đẹp. Chụp ảnh zoom 100x ấn tượng."),
        (2, "Đỗ Quang", "Pin dùng cả ngày không hết, sạc nhanh 45W tiện lợi. Nâng cấp từ S22 lên rất hài lòng."),
        (2, "Ngô Trang", "Máy bị nóng khi chơi game nặng, còn lại mọi thứ đều tuyệt vời."),
        (2, "Bùi Tuấn", "Giao hàng đúng hẹn, hàng mới nguyên seal. Nhưng giá vẫn hơi chát."),
        (2, "Dương Mai", "Mình thất vọng quá, camera selfie bị mờ. Phải mang đi bảo hành lần 2 rồi."),
        # MacBook Air M3
        (3, "Phan Khánh", "Mỏng nhẹ, pin trâu 18 tiếng, màn hình Retina sắc nét. Mình dùng code cả ngày rất mượt."),
        (3, "Trịnh Đức", "Chip M3 quá mạnh, render video 4K không giật. macOS giờ cũng tối ưu cho công việc."),
        (3, "Lý Anh", "Bàn phím gõ sướng, trackpad siêu nhạy. Nhưng chỉ có 8GB RAM thì hơi ít cho dev."),
        (3, "Hồ Tuyết", "Máy đẹp nhưng loa ngoài hơi nhỏ. Nói chung là laptop văn phòng lý tưởng."),
        (3, "Mai Phương", "Mua về mà không cài được phần mềm mình cần, phải dùng Parallel. Hơi bất tiện."),
        # Sony WH-1000XM5
        (4, "Đặng Khoa", "Chống ồn tốt nhất mình từng dùng. Nghe nhạc bass rất đã, đeo cả ngày không đau tai."),
        (4, "Cao Thắng", "Pin 30 tiếng là thật, sạc 3 phút nghe 3 tiếng. Quá đỉnh!"),
        (4, "Tạ Thu", "Mình mua về thì headband bị lỏng, không ôm đầu. Hơi thất vọng với tầm giá này."),
        (4, "Nguyễn Hà", "Âm thanh hay, kết nối multipoint tiện. App Sony Headphones Connect dễ dùng."),
        # Áo thun nam
        (5, "Lương Phong", "Vải cotton mềm, thấm mồ tốt, form áo đẹp. Mua 3 cái rồi, rất ưng."),
        (5, "Châu Giang", "Áo mỏng quá, giặt 2 lần đã bị giãn. Không như hình quảng cáo."),
        (5, "Phạm Hùng", "Giá rẻ mà chất lượng ổn. Giao hàng nhanh, đóng gói cẩn thận."),
        (5, "Trương Lan", "Mình đặt size L mà nhận như XL. Phải đổi lại rất mất thời gian."),
        # Nồi chiên không dầu
        (6, "Bảo Ngọc", "Nướng gà nguyên con rất ngon, giòn đều. Tiết kiệm dầu mỡ, sạch sẽ."),
        (6, "Hoàng Yến", "Dùng ok nhưng ồn quá, kêu to hơn mình nghĩ. Nói chung vẫn nấu được."),
        (6, "Võ Thành", "Sản phẩm tốt, mình chiên khoai tây giòn tan. Nhưng dung tích hơi nhỏ cho gia đình đông người."),
        (6, "Lê Trang", "Mới dùng được 1 tháng đã hỏng quạt. Gọi bảo hành thì chờ cả tuần. Rất bực mình."),
    ]

    # Run model prediction for each comment
    labeled_comments = []
    for pid, user, content in raw_comments:
        label, confidence = predict_sentiment(content)
        labeled_comments.append((pid, user, content, label, confidence))

    conn.executemany(
        "INSERT INTO comments (product_id, user_name, content, label, confidence) VALUES (?, ?, ?, ?, ?)",
        labeled_comments,
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan — load model & init DB once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, vocab

    vocab = load_vocab(VOCAB_PATH)

    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        emb_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=3,
        num_heads=4,
        dropout=0.5,
    )
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    init_db()

    yield


app = FastAPI(
    title="Vietnamese Sentiment API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST_DIR / "assets"),
        name="assets",
    )


@app.get("/")
def serve_ui():
    frontend_index = FRONTEND_DIST_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    return FileResponse(BASE_DIR / "index.html")


@app.get("/stats")
def serve_stats():
    return FileResponse(BASE_DIR / "stats.html")


@app.get("/products")
def serve_products():
    return FileResponse(BASE_DIR / "products.html")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["sản phẩm tốt"])


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float


class CommentCreateRequest(BaseModel):
    product_id: int = Field(..., description="Product ID")
    user_name: str = Field(..., min_length=1, max_length=100, examples=["Nguyễn Văn A"])
    content: str = Field(..., min_length=1, max_length=2000, examples=["Sản phẩm rất tốt"])


class CommentCreateResponse(BaseModel):
    id: int
    product_id: int
    user_name: str
    content: str
    label: str
    confidence: float
    message: str


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text_v2(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning.")

    tokens = cleaned.split()
    input_ids, input_mask = encode(tokens, vocab, MAX_LEN)

    with torch.no_grad():
        logits, _ = model(input_ids, input_mask)

    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    return PredictResponse(
        text=req.text,
        label=LABEL_MAP[pred_idx],
        confidence=round(probs[pred_idx].item(), 4),
    )


# ---------------------------------------------------------------------------
# Product & Comment API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/products")
def list_products():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, category, price, image_url FROM products ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/comments")
def list_comments(
    product_id: int | None = Query(None, description="Filter by product"),
    label: str | None = Query(None, description="Filter by label: Tích cực / Tiêu cực / Trung tính"),
):
    conn = get_db()
    query = """
        SELECT c.id, c.product_id, p.name AS product_name, p.category,
               c.user_name, c.content, c.label, c.confidence, c.created_at
        FROM comments c
        JOIN products p ON c.product_id = p.id
        WHERE 1=1
    """
    params: list = []
    if product_id is not None:
        query += " AND c.product_id = ?"
        params.append(product_id)
    if label is not None:
        query += " AND c.label = ?"
        params.append(label)
    query += " ORDER BY c.created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/comments/{comment_id}")
def get_comment(comment_id: int):
    conn = get_db()
    row = conn.execute(
        """
        SELECT c.id, c.product_id, p.name AS product_name, p.category, p.price,
               p.image_url, c.user_name, c.content, c.label, c.confidence, c.created_at
        FROM comments c
        JOIN products p ON c.product_id = p.id
        WHERE c.id = ?
        """,
        (comment_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Comment not found")
    return dict(row)


@app.post("/api/comments", response_model=CommentCreateResponse)
def create_comment(req: CommentCreateRequest):
    """Submit a new comment — the BiLSTM model auto-predicts sentiment."""
    # Verify product exists
    conn = get_db()
    product = conn.execute("SELECT id FROM products WHERE id = ?", (req.product_id,)).fetchone()
    if not product:
        conn.close()
        raise HTTPException(status_code=404, detail="Product not found")

    # Run sentiment prediction
    label, confidence = predict_sentiment(req.content)

    # Insert into DB
    cursor = conn.execute(
        "INSERT INTO comments (product_id, user_name, content, label, confidence) VALUES (?, ?, ?, ?, ?)",
        (req.product_id, req.user_name, req.content, label, confidence),
    )
    comment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return CommentCreateResponse(
        id=comment_id,
        product_id=req.product_id,
        user_name=req.user_name,
        content=req.content,
        label=label,
        confidence=confidence,
        message="Bình luận đã được ghi nhận và phân tích cảm xúc bởi mô hình BiLSTM",
    )


@app.get("/api/stats")
def get_stats():
    """Return sentiment distribution counts + per-product breakdown."""
    conn = get_db()
    # Overall counts
    overall = conn.execute(
        "SELECT label, COUNT(*) AS cnt FROM comments GROUP BY label"
    ).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]

    # Per-product breakdown
    breakdown = conn.execute("""
        SELECT p.id AS product_id, p.name AS product_name,
               c.label, COUNT(*) AS cnt
        FROM comments c
        JOIN products p ON c.product_id = p.id
        GROUP BY p.id, c.label
        ORDER BY p.id
    """).fetchall()

    # Build per-product structure
    products_map: dict[int, dict] = {}
    for row in breakdown:
        pid = row["product_id"]
        if pid not in products_map:
            products_map[pid] = {
                "product_id": pid,
                "product_name": row["product_name"],
                "Tích cực": 0,
                "Tiêu cực": 0,
                "Trung tính": 0,
                "total": 0,
            }
        products_map[pid][row["label"]] = row["cnt"]
        products_map[pid]["total"] += row["cnt"]

    conn.close()

    return {
        "total": total,
        "distribution": {row["label"]: row["cnt"] for row in overall},
        "by_product": list(products_map.values()),
    }


@app.get("/api/reanalyze")
def reanalyze_comments():
    """Re-run sentiment prediction on all comments (e.g. after model update)."""
    conn = get_db()
    rows = conn.execute("SELECT id, content FROM comments").fetchall()
    updated = 0
    for row in rows:
        label, confidence = predict_sentiment(row["content"])
        conn.execute(
            "UPDATE comments SET label = ?, confidence = ? WHERE id = ?",
            (label, confidence, row["id"]),
        )
        updated += 1
    conn.commit()
    conn.close()
    return {"updated": updated}


# ---------------------------------------------------------------------------
# SPA fallback (must be last)
# ---------------------------------------------------------------------------
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith("assets") or full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not found")

    frontend_index = FRONTEND_DIST_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)

    return FileResponse(BASE_DIR / "index.html")


# ---------------------------------------------------------------------------
# Run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
