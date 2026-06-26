# Báo cáo Kỹ thuật: Phân loại Cảm xúc Đánh giá Sản phẩm TMĐT bằng Kiến trúc BiLSTM và Multi-Head Additive Attention

## 1. Đặt vấn đề: Những thách thức cốt lõi trong Phân loại Cảm xúc Tiếng Việt

Việc xây dựng một hệ thống phân tích cảm xúc (Sentiment Analysis) tự động cho các bình luận trên nền tảng Thương mại điện tử (TMĐT) tại Việt Nam đối mặt với những rào cản kỹ thuật đặc thù. Những rào cản này đến từ cả bản chất của dữ liệu đầu vào lẫn giới hạn của các kiến trúc học máy truyền thống.

### 1.1. Hạn chế về mặt Dữ liệu (Data Complexity & Noise)
Dữ liệu sinh ra từ người dùng (User-Generated Content) trên các nền tảng như Shopee, Tiki, Lazada chứa độ nhiễu (noise) cực kỳ cao:
*   **Ngôn ngữ phi tiêu chuẩn (Teencode & Viết tắt):** Người dùng hiếm khi viết chuẩn định dạng. Các từ ngữ như "ko", "đc", "sp", "rất tốt" bị biến thể thành "k", "dk", "san pam", "rất tút". Điều này làm bùng nổ không gian từ vựng (vocabulary size) và gây hiện tượng Out-of-Vocabulary (OOV).
*   **Ký tự đặc biệt và Emoji:** Cảm xúc đôi khi không được diễn đạt bằng lời mà bằng các biểu tượng (😍, 😡, 🐢). Một mô hình xử lý văn bản thuần túy sẽ bỏ qua các tín hiệu cảm xúc cực kỳ mạnh mẽ này.
*   **Cú pháp lỏng lẻo và Từ đệm (Stopwords):** Sự xuất hiện dày đặc của các từ như "thì", "là", "mà", "cái này" làm loãng mật độ thông tin hữu ích trong một câu đánh giá.
*   **Tính đa chiều và Cảm xúc phức hợp:** Một bình luận có thể chứa nhiều vế đối lập nhau. Ví dụ: *"Giao hàng mất 3 ngày, bọc chống sốc cẩn thận, shipper thân thiện, nhưng hàng bên trong bị vỡ nát"*. Phân loại toàn bộ câu này thành "Tích cực" hay "Tiêu cực" đòi hỏi khả năng cân nhắc trọng số của từng vế.
*   **Hiện tượng mỉa mai (Sarcasm):** Ví dụ *"Shop giao hàng nhanh như rùa bò 🐢 đỉnh thật sự"*. Phân tích từ vựng đơn thuần sẽ thấy các từ "nhanh", "đỉnh" (tích cực), nhưng bản chất câu lại là cực kỳ tiêu cực.

### 1.2. Hạn chế về Kiến trúc Mô hình (Model Architecture Limitations)
Nếu chúng ta tiếp cận bài toán bằng các mô hình cơ bản, sự thất bại là điều có thể dự báo trước:
*   **Bag-of-Words (BoW) / TF-IDF:** Phương pháp này đếm tần suất xuất hiện của từ mà vứt bỏ hoàn toàn cấu trúc ngữ pháp. Nó sẽ đánh đồng hai câu *"Sản phẩm này **không** đẹp, chỉ **tốt** thôi"* và *"Sản phẩm này **không** tốt, chỉ **đẹp** thôi"* vì chúng có chung tập hợp từ.
*   **Mạng Nơ-ron Tích chập (CNN - 1D):** CNN rất xuất sắc trong việc quét các cụm từ (n-grams) cục bộ (ví dụ: tìm thấy cụm "rất tốt"). Tuy nhiên, CNN thất bại trong việc xử lý **phụ thuộc xa (Long-range dependencies)**. Ví dụ: *"Màn hình thì cũng được, pin trâu, thiết kế đẹp, nói chung là **không** nên mua"*. CNN có thể bị đánh lừa bởi mật độ từ tích cực ở đầu câu mà bỏ lỡ từ phủ định chốt hạ ở cuối câu.
*   **Mạng Nơ-ron Hồi quy (RNN) / LSTM truyền thống:** LSTM ra đời để giải quyết bài toán phụ thuộc xa bằng cách duy trì một trạng thái bộ nhớ (cell state). Tuy nhiên, nó vấp phải **Nút thắt cổ chai thông tin (Information Bottleneck)**. Khi xử lý một câu dài, LSTM bị ép buộc phải nén toàn bộ thông tin của câu vào một vector duy nhất ở bước thời gian cuối cùng. Quá trình "nén" này khiến thông tin ở đầu câu bị rơi rụng (forgetting), dẫn đến mất mát dữ liệu nghiêm trọng.

---

## 2. Giải pháp Kiến trúc: Tư duy Tiến hóa và Sự hình thành Logic

Để vượt qua những hạn chế trên, chúng tôi không chọn một mô hình ngẫu nhiên mà thiết kế một kiến trúc dựa trên sự tiến hóa có tính hệ thống. Quá trình suy luận diễn ra như sau:

### Lập luận 1: Xử lý sự phụ thuộc cấu trúc và ngữ cảnh $\rightarrow$ Chọn BiLSTM
*   **Nguyên lý:** Tiếng Việt là ngôn ngữ đơn lập, ý nghĩa của một từ phụ thuộc tuyệt đối vào vị trí của nó và các từ xung quanh. Từ "thích" đứng sau "không" mang ý nghĩa trái ngược hoàn toàn so với khi đứng sau "rất".
*   **Hành động:** Thay vì dùng LSTM chạy một chiều (từ trái sang phải), chúng tôi sử dụng **Bidirectional LSTM (BiLSTM)**. Bằng cách kết hợp hai luồng thông tin (trước $\rightarrow$ sau và sau $\rightarrow$ trước), mỗi từ trong câu khi được mã hóa sẽ mang theo nhận thức về toàn bộ ngữ cảnh xung quanh nó.

### Lập luận 2: Phá vỡ nút thắt cổ chai và Lọc nhiễu $\rightarrow$ Tích hợp Attention
*   **Nguyên lý:** Mặc dù BiLSTM đã làm tốt việc giữ lại ngữ cảnh, nhưng bắt nó nén một câu 64 từ thành 1 vector duy nhất vẫn là quá sức. Hơn nữa, trong 64 từ đó, chỉ có 3-4 từ thực sự mang "trọng lượng cảm xúc" (sentiment-bearing words), phần còn lại là nhiễu.
*   **Hành động:** Chúng tôi áp dụng **Cơ chế Attention**. Thay vì chỉ lấy đầu ra cuối cùng của BiLSTM, Attention cho phép mô hình giữ lại toàn bộ 64 trạng thái ẩn của 64 từ. Sau đó, nó tự động học cách "chấm điểm" (gán trọng số) cho từng từ. Những từ như "tuyệt_vời" sẽ nhận được trọng số 90%, trong khi từ "cái" chỉ nhận 0.1%. Kết quả là một Vector Tóm tắt (Context Vector) được thanh lọc hoàn toàn khỏi nhiễu.

### Lập luận 3: Xử lý sự phức hợp của cảm xúc mạng $\rightarrow$ Nâng cấp lên Multi-Head
*   **Nguyên lý:** Con người không đánh giá một câu chỉ qua một góc nhìn. Chúng ta tìm kiếm tính từ, rà soát từ phủ định, và để ý đến cả biểu tượng cảm xúc. Nếu chỉ dùng 1-Head Attention, mô hình bị ép phải học một bộ tiêu chí quá "ôm đồm", dễ dẫn đến nhiễu loạn khi gặp câu phức tạp.
*   **Hành động:** Chúng tôi phân tách cơ chế Attention thành **4 Heads độc lập (Multi-Head)**. Bắt chước mô hình làm việc nhóm, Head 1 có thể chuyên săn lùng tính từ, Head 2 chuyên rà soát từ phủ định, Head 3 tìm kiếm các từ chỉ cường độ ("rất", "quá"). Khi 4 luồng thông tin này được nối lại (concat), chúng ta có một bản tóm tắt đa chiều và sâu sắc.

### Lập luận 4: Lựa chọn Additive Attention thay vì Scaled Dot-Product (Transformer)
*   **Nguyên lý:** Tại sao không dùng cơ chế Attention của Transformer (Vaswani, 2017) vốn đang rất phổ biến?
*   **Hành động:** Transformer sử dụng Self-Attention (phân rã Input thành Query, Key, Value) để tính toán ma trận tương quan toàn cục. Nó rất mạnh nhưng yêu cầu tài nguyên khổng lồ và tập dữ liệu hàng triệu câu để không bị Overfitting. Trong bài toán của chúng tôi (với dữ liệu vừa phải, độ dài câu ngắn max 64 tokens), việc sử dụng **Additive Attention (Bahdanau-style)** đặt ngay sau khối BiLSTM mang lại hiệu quả vượt trội. Additive Attention sử dụng một mạng Feed-Forward nhỏ để học ra một "Vector Ngữ cảnh Tĩnh" (Learnable Context Vector) dùng để chấm điểm trực tiếp các đầu ra của BiLSTM. Cách này ít tham số hơn, tính toán nhanh hơn, hội tụ dễ hơn mà vẫn đảm bảo khả năng trích xuất đặc trưng chính xác cho bài toán phân loại.

---

## 3. Phân tích Chuyên sâu Kiến trúc Mô hình (Deep Dive Architecture)

### 3.1. BiLSTM: Tối ưu hóa Ngữ cảnh Hai chiều
*   **Cơ chế:** Khối BiLSTM trong mô hình gồm 2 layers, kích thước `hidden_dim = 256`. Đầu vào là các vector nhúng (embedding) 300 chiều. 
*   **Toán học:** Tại bước thời gian $t$, đầu ra của luồng Forward là $\overrightarrow{h_t}$, đầu ra của luồng Backward là $\overleftarrow{h_t}$. Đầu ra thực tế của BiLSTM tại vị trí $t$ là phép ghép nối: $h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$. Vector $h_t$ lúc này có kích thước 512 chiều (256 + 256).
*   **Ý nghĩa:** Nhờ quá trình này, mô hình đã "hiểu" được rằng chữ "thô" trong câu *"Tuy thô nhưng máy chạy rất nhanh"* có liên kết mật thiết với chữ "nhưng" đang chờ đợi ở phía sau, từ đó điều chỉnh trạng thái nội tại để không vội vàng kết luận câu này là tiêu cực.

### 3.2. Multi-Head Additive Attention: Lõi Phân tích Thông minh
Cơ chế này thực thi qua 4 bước lõi tại mỗi Head:
1.  **Biến đổi không gian (Projection):** Đưa các vector 512 chiều của BiLSTM qua một lớp Linear và hàm kích hoạt `tanh` để tạo ra một không gian biểu diễn mới (không gian quyết định cảm xúc).
    $$u_t = \tanh(W_{proj} \cdot h_t + b_{proj})$$
2.  **Tính điểm (Scoring):** Sử dụng một vector trọng số học được ($v_{score}$) để chấm điểm cho từng từ.
    $$e_t = v_{score}^T \cdot u_t$$
    *(Đây chính là điểm tạo nên tên gọi "Additive". Thay vì lấy $Q \cdot K^T$ như Transformer, nó cộng các giá trị qua một mạng nơ-ron truyền thẳng 1 lớp).*
3.  **Chuẩn hóa (Softmax & Masking):** Loại bỏ các điểm số của các từ Padding (gán bằng $-10^9$) và đưa qua hàm Softmax để tạo thành một phân phối xác suất (tổng các trọng số bằng 1).
    $$\alpha_t = \text{softmax}(e_t)$$
4.  **Tổng hợp (Context Vector):** Lấy các trọng số $\alpha_t$ nhân với các vector đầu ra nguyên bản của BiLSTM $h_t$ để tạo thành Vector Đại Diện cho toàn bộ câu.
    $$c = \sum_{t=1}^{T} \alpha_t h_t$$

### 3.3. Luồng dữ liệu Tổng quát của Hệ thống (End-to-End Flow)
1.  **Input:** Một bình luận (đã tiền xử lý) được biểu diễn dưới dạng mảng 64 Token IDs.
2.  **Embedding Layer:** Chuyển đổi 64 IDs thành ma trận `[64, 300]`. Sử dụng trọng số FastText Tiếng Việt đã được tối ưu hóa (Compound Averaging).
3.  **BiLSTM Layer:** Nhận ma trận `[64, 300]` và xuất ra ma trận giàu ngữ cảnh `[64, 512]`.
4.  **Multi-Head Attention Layer:** Nhận ma trận `[64, 512]`, chạy qua 4 Heads, tính toán trọng số, và thu gọn lại thành duy nhất 1 Vector `[512]`.
5.  **Classifier Layer:** Vector `[512]` đi qua các lớp Fully Connected (`Linear 512->256`, kích hoạt `GELU`, `Dropout 0.3`, `Linear 256->3`).
6.  **Output:** 3 giá trị Logits. Giá trị có xác suất Softmax cao nhất sẽ là Nhãn dự đoán (0, 1 hoặc 2).

---

## 4. Đặc tả Tập dữ liệu Huấn luyện (The Dataset Specification)

Chất lượng của mô hình Deep Learning phụ thuộc tuyệt đối vào dữ liệu. Dữ liệu trong dự án này được thu thập, tinh tuyển và gộp từ nhiều nguồn đánh giá uy tín để đảm bảo tính đa dạng.

### 4.1. Nguồn gốc & Kích thước
*   **Shopee Reviews Dataset:** Dữ liệu cào thực tế từ Shopee, đóng góp ~31,460 mẫu.
*   **AIVIVN & Các nguồn khác:** Được nạp từ các file `train.csv`, `test.csv`, `vsa_food_rv_train.csv`.
*   **Tổng quy mô dữ liệu thô:** Đạt **82,056 bình luận**.

### 4.2. Đặc tính Nhãn (Label Distribution)
Dữ liệu được map về 3 nhãn chuẩn: Tích cực (0), Tiêu cực (1), và Trung tính (2).
Sự phân bố ban đầu mang đặc tính mất cân bằng tự nhiên (Imbalanced Class) của các bài đánh giá TMĐT:
*   **Tích cực:** 41,315 mẫu (chiếm ~50%).
*   **Tiêu cực:** 32,269 mẫu (chiếm ~40%).
*   **Trung tính:** 8,472 mẫu (chiếm ~10%).

---

## 5. Quy trình Làm sạch và Chuẩn bị Dữ liệu (Preprocessing Pipeline)

Để đối phó với sự hỗn loạn của văn bản TMĐT, chúng tôi đã phát triển một Pipeline Tiền xử lý 10 bước nghiêm ngặt. Đây là khâu tốn nhiều công sức nhất (chiếm 80% nỗ lực kỹ thuật).

1.  **Chuẩn hóa Unicode (NFC) & text_normalize:** Đảm bảo các ký tự tiếng Việt hiển thị đồng nhất (chống lỗi encoding).
2.  **Emoji Translation (Khâu cực kỳ quan trọng):** Chúng tôi không vứt bỏ Emoji. 59 Emoji thông dụng được ánh xạ thành các token text (Ví dụ: 😍 $\rightarrow$ `positive_strong`, 😡 $\rightarrow$ `negative_strong`). Thao tác này được thực hiện *trước* khi tách từ thông qua các Placeholder (`stkn0z`) để bảo toàn định dạng.
3.  **Chuẩn hóa Dấu câu Nhấn mạnh:** Các chuỗi `!!!` hay `???` được thay thế bằng các token cụ thể như `strong_emphasis`, `confusion` nhằm lưu giữ sắc thái cường điệu.
4.  **Lọc nhiễu Cấu trúc:** Loại bỏ toàn bộ mã HTML (`<br>`, `<b>`), URLs và Email.
5.  **Lowercasing:** Đưa toàn bộ văn bản về chữ thường.
6.  **Teencode Resolution:** Sử dụng một từ điển tùy chỉnh gồm 106 mục để dịch các từ lóng TMĐT về tiếng Việt chuẩn (`sp` $\rightarrow$ `sản phẩm`, `auth` $\rightarrow$ `chính hãng`, `ship` $\rightarrow$ `giao hàng`).
7.  **Word Segmentation (Tách từ):** Sử dụng thư viện `underthesea`. Việc này giúp mô hình nhận diện `sản_phẩm` là một thực thể duy nhất (1 token) thay vì 2 token rời rạc `sản` và `phẩm`.
8.  **Khôi phục Placeholder:** Các chuỗi `stkn0z` được bung ngược trở lại thành các thẻ cảm xúc (ví dụ: `positive_strong`).
9.  **Lọc Ký tự & Stopwords An Toàn:** Xóa các ký tự đặc biệt vô nghĩa. Đặc biệt, chúng tôi định nghĩa một danh sách `SAFE_STOPWORDS` (gồm 15 từ như 'là', 'của', 'thì'). **Tuyệt đối không loại bỏ** các từ phủ định ('không', 'chưa') và từ chỉ mức độ ('rất', 'quá'), vì việc xóa chúng sẽ làm đảo lộn ngữ nghĩa toàn câu.
10. **Xây dựng Vocabulary:** Lọc bỏ các từ xuất hiện dưới 2 lần (MIN_FREQ=2) để giảm chiều dữ liệu nhiễu. Tập từ vựng cuối cùng chốt ở mức **18,478 từ**.

---

## 6. Quy trình Huấn luyện và Đánh giá (Training & Evaluation Protocol)

Quá trình huấn luyện không diễn ra một cách ngẫu nhiên mà áp dụng nhiều kỹ thuật tinh chỉnh (finetuning) để tối ưu hóa hiệu suất (được xác nhận thông qua phiên bản v2.2).

### 6.1. Thiết lập Tập dữ liệu & Khắc phục Mất cân bằng
*   **Chia tập dữ liệu (Splitting):** Dữ liệu được chia theo tỷ lệ 80/10/10 (Train: 65,582 | Val: 8,198 | Test: 8,198) có giữ nguyên tỷ lệ phân phối nhãn (stratified).
*   **Xử lý Imbalance Data:** Do nhóm Trung tính chỉ chiếm 10%, nếu huấn luyện bình thường, mô hình sẽ có xu hướng "bỏ qua" lớp này để tối ưu hóa độ chính xác tổng thể. Chúng tôi giải quyết bằng cách áp dụng **Class Weights** trực tiếp vào hàm Loss (`CrossEntropyLoss`). 
    *   *Sự tinh tế:* Chúng tôi dùng tỷ lệ nghịch đảo căn bậc hai `1 / sqrt(counts)` thay vì hàm tuyến tính, giúp các trọng số ôn hòa hơn (Tích cực: 0.69, Tiêu cực: 0.78, Trung tính: 1.52). Điều này ngăn chặn hiện tượng Overfitting vào lớp thiểu số (điều đã xảy ra ở phiên bản v2.1).

### 6.2. Hyperparameters & Optimization
*   **Optimizer:** `AdamW` với Learning Rate khởi tạo là `3e-4` và `weight_decay = 5e-4` (giúp chống Overfitting).
*   **Scheduler:** Sử dụng `ReduceLROnPlateau`. Nếu Validation Loss không cải thiện sau 3 epochs, Learning Rate sẽ tự động giảm đi một nửa, giúp mô hình hội tụ "mịn" hơn ở giai đoạn cuối.
*   **Early Stopping:** Được thiết lập với `patience = 8`. Nếu mô hình không tiến bộ sau 8 epochs, quá trình train sẽ tự động ngắt để bảo toàn trọng số tốt nhất.

### 6.3. Kết quả Báo cáo (Test Set Evaluation)
Sau gần 40 epochs huấn luyện trên môi trường GPU (CUDA), mô hình đạt trạng thái tối ưu ở Epoch 38 với kết quả xuất sắc:

*   **Overall Accuracy:** **90.6%**
*   **Macro F1-Score:** **0.860** (Tiêu chuẩn khắt khe nhất cho dữ liệu mất cân bằng).
*   **Hiệu suất từng lớp (F1-Score):**
    *   Tích cực: **0.931**
    *   Tiêu cực: **0.920**
    *   Trung tính: **0.729**

Mô hình không chỉ xuất sắc ở mặt chỉ số (Metrics) mà còn chứng minh được tính minh bạch (Interpretability). Thông qua các bản đồ Attention (Attention Heatmaps), chúng ta có thể thấy rõ ràng mô hình đã gán trọng số cực cao cho các từ khóa cốt lõi (như "tệ_lắm", "không", "thất_vọng") đúng như những gì kiến trúc đã được kỳ vọng từ khâu thiết kế.

