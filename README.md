# Vietnamese E-commerce Sentiment Classifier using BiLSTM & Multi-head Additive Attention

Dự án phân loại cảm xúc các bình luận đánh giá sản phẩm trên nền tảng Thương mại điện tử (TMĐT) tại Việt Nam, sử dụng kiến trúc mạng **BiLSTM** kết hợp cơ chế **Multi-Head Additive Attention**.

## 📌 Đặt vấn đề (The Problem)
Phân tích cảm xúc (Sentiment Analysis) dữ liệu bình luận tại Việt Nam gặp nhiều thách thức như: ngôn ngữ phi tiêu chuẩn (teencode), emoji, từ đệm, và đặc biệt là các cảm xúc phức hợp như mỉa mai (sarcasm). Các mô hình truyền thống (BoW, CNN, hay LSTM đơn thuần) thường gặp hiện tượng "nút thắt cổ chai thông tin" và bỏ sót các phụ thuộc xa trong câu.

## 🚀 Giải pháp & Kiến trúc (Architecture)
* **Tiền xử lý (Preprocessing):** Tích hợp pipeline chuẩn hóa teencode, viết tắt, và emoji.
* **BiLSTM:** Xử lý câu theo hai chiều, giúp mô hình nắm bắt được ngữ cảnh toàn diện của từ vựng trước và sau nó.
* **Multi-Head Additive Attention:** Thay vì nén toàn bộ câu vào một vector duy nhất, cơ chế này giúp mô hình "chú ý" (focus) vào các từ/cụm từ quyết định cảm xúc (ví dụ: các từ khóa đảo ngược ý nghĩa, các từ nhấn mạnh).
* **Ứng dụng Web:** Cung cấp giao diện trực quan (`index.html`, `products.html`, `stats.html`) để hiển thị kết quả phân tích dữ liệu thực tế.

## 📁 Thành phần dự án
- `model.py` / `main.py`: Định nghĩa kiến trúc mô hình và luồng thực thi chính.
- `preprocess.py`: Các module tiền xử lý văn bản tiếng Việt chuyên dụng.
- `PIPELINE_ARCHITECTURE.md`: Tài liệu báo cáo kỹ thuật chi tiết về kiến trúc.
- `bilstm_v22_weights.pth`: Trọng số mô hình đã được huấn luyện.
- `*.html`: Bộ giao diện trực quan hóa thông tin và thống kê đánh giá.

## 🛠️ Cài đặt & Sử dụng (Installation)
1. Clone repo: `git clone https://github.com/TPH-Per/Comment-types-classifier-using-LSTM-Mutihead-Additive-Attention.git`
2. Cài đặt môi trường: `pip install -r requirements.txt`
3. Chạy hệ thống thông qua `main.py` hoặc xem giao diện web.

## 📊 Hình ảnh & Đánh giá
Các biểu đồ quá trình huấn luyện (Training Dynamics), Ma trận nhầm lẫn (Confusion Matrix), và Biểu đồ ma trận trọng số Attention được lưu trong các file `Figure*.png` minh họa cho khả năng nắm bắt ngôn ngữ của mô hình.
