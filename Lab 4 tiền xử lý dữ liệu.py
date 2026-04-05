import pandas as pd
import numpy as np
import warnings
import re
import math
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

VIETNAMESE_STOPWORDS = {
    'và', 'của', 'là', 'có', 'được', 'với', 'các', 'trong', 'không',
    'một', 'này', 'để', 'những', 'đã', 'cho', 'rất', 'hay', 'khi',
    'cũng', 'như', 'thì', 'mà', 'về', 'bị', 'từ', 'tôi', 'bạn',
    'họ', 'chúng', 'ta', 'nó', 'vì', 'nên', 'nhưng', 'hơn', 'đó',
    'đây', 'lại', 'vẫn', 'đang', 'sẽ', 'tại', 'theo', 'sau', 'trước',
    'trên', 'dưới', 'ra', 'vào', 'đến', 'hết', 'lên', 'xuống', 'qua',
    'còn', 'thế', 'ai', 'gì', 'đâu', 'sao', 'mình', 'nhiều', 'ít',
    'thật', 'quá', 'cực', 'biết', 'thấy', 'nếu', 'tuy', 'mặc', 'dù',
    'hoặc', 'ngoài', 'kể', 'suốt', 'toàn', 'chỉ', 'khá', 'tốt',
}

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]
    return tokens

def preprocess_to_string(text):
    return ' '.join(preprocess_text(text))


class SimpleWord2Vec:
    def __init__(self, vector_size=50, window=2, min_count=1, epochs=100, lr=0.025, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.word2idx = {}
        self.idx2word = {}
        self.W = None
        self.W2 = None

    def build_vocab(self, sentences):
        counts = Counter(w for sent in sentences for w in sent)
        vocab = [w for w, c in counts.items() if c >= self.min_count]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)

    def _generate_pairs(self, sentences):
        pairs = []
        for sent in sentences:
            idxs = [self.word2idx[w] for w in sent if w in self.word2idx]
            for i, center in enumerate(idxs):
                start = max(0, i - self.window)
                end = min(len(idxs), i + self.window + 1)
                for j in range(start, end):
                    if j != i:
                        pairs.append((center, idxs[j]))
        return pairs

    def fit(self, sentences):
        self.build_vocab(sentences)
        rng = np.random.default_rng(self.seed)
        self.W  = (rng.random((self.vocab_size, self.vector_size)) - 0.5) / self.vector_size
        self.W2 = np.zeros((self.vocab_size, self.vector_size))
        pairs = self._generate_pairs(sentences)
        if not pairs:
            return self
        for epoch in range(self.epochs):
            pairs_arr = np.array(pairs, dtype=np.int32)
            rng.shuffle(pairs_arr)
            for center, ctx in pairs_arr:
                h = self.W[center]
                scores = self.W2 @ h
                exp_s = np.exp(scores - np.max(scores))
                probs = exp_s / exp_s.sum()
                probs[ctx] -= 1
                grad_W2 = np.outer(probs, h)
                grad_h  = self.W2.T @ probs
                self.W2 -= self.lr * grad_W2
                self.W[center] -= self.lr * grad_h
        return self

    def most_similar(self, word, topn=5):
        if word not in self.word2idx:
            raise KeyError(f"'{word}' không có trong vocabulary")
        idx = self.word2idx[word]
        vec = self.W[idx]
        norms = np.linalg.norm(self.W, axis=1)
        norms[norms == 0] = 1e-10
        cosines = (self.W @ vec) / (norms * np.linalg.norm(vec) + 1e-10)
        cosines[idx] = -1
        top_idxs = np.argsort(cosines)[::-1][:topn]
        return [(self.idx2word[i], float(cosines[i])) for i in top_idxs]

    def __contains__(self, word):
        return word in self.word2idx

    @property
    def vocab(self):
        return list(self.word2idx.keys())


def find_similar(model, candidates, topn=5):
    for w in candidates:
        if w in model:
            return w, model.most_similar(w, topn=topn)
    for w in candidates:
        partial = w.split()[0] if ' ' in w else w
        matches = [v for v in model.vocab if partial in v]
        if matches:
            return matches[0], model.most_similar(matches[0], topn=topn)
    return None, []


def print_section(title):
    print('\n' + '=' * 65)
    print(f'  {title}')
    print('=' * 65)

def print_subsection(title):
    print(f'\n--- {title} ---')

def print_similar(word_found, results):
    if word_found and results:
        print(f'\n5 từ gần nghĩa nhất với "{word_found}":')
        for word, score in results:
            print(f'  {word:<20} cosine similarity = {score:.4f}')
    else:
        print('  (Không tìm được từ trong vocabulary — dữ liệu quá nhỏ)')


print_section('BÀI 1: REVIEW KHÁCH SẠN (2đ)')

print_subsection('1.1 Nạp dữ liệu & kiểm tra missing values')
df1 = pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv', encoding='utf-8')
print(f'Kích thước: {df1.shape}  |  Cột: {df1.columns.tolist()}')
print('\nKiểu dữ liệu từng cột:')
print(df1.dtypes.to_string())
print('\nSố giá trị bị thiếu (missing values):')
print(df1.isnull().sum().to_string())
print('\nMẫu dữ liệu (3 hàng đầu):')
print(df1.head(3).to_string())

df1['hotel_name'].fillna(df1['hotel_name'].mode()[0], inplace=True)
df1['customer_type'].fillna(df1['customer_type'].mode()[0], inplace=True)
df1.dropna(subset=['review_text'], inplace=True)
df1.reset_index(drop=True, inplace=True)
print(f'\n→ Sau xử lý missing: {df1.shape}')

print_subsection('1.2 Label Encoding & One-Hot Encoding')

le1 = LabelEncoder()
df1['hotel_name_LE'] = le1.fit_transform(df1['hotel_name'])
print('Label Encoding — hotel_name (5 nhãn đầu):')
mapping_df = pd.DataFrame({'Nhãn gốc': le1.classes_, 'Mã Label': range(len(le1.classes_))})
print(mapping_df.head().to_string(index=False))

df1_ohe = pd.get_dummies(df1[['customer_type']], columns=['customer_type'], prefix='ctype', dtype=int)
print(f'\nOne-Hot Encoding — customer_type → {list(df1_ohe.columns)}')
print(df1_ohe.head().to_string())

print_subsection('1.3 Tiền xử lý văn bản (lowercase → tokenize → remove stopwords)')
df1['tokens']     = df1['review_text'].apply(preprocess_text)
df1['clean_text'] = df1['review_text'].apply(preprocess_to_string)
print('Ví dụ:')
for i in range(3):
    print(f'  Gốc : {df1["review_text"].iloc[i]}')
    print(f'  Sau : {df1["clean_text"].iloc[i]}')
    print()

print_subsection('1.4 Tạo TF-IDF Matrix')
tfidf1 = TfidfVectorizer(max_features=100)
X1 = tfidf1.fit_transform(df1['clean_text'])
print(f'TF-IDF Matrix: {X1.shape[0]} văn bản × {X1.shape[1]} từ')
mean1 = np.asarray(X1.mean(axis=0)).flatten()
top10_1 = np.argsort(mean1)[::-1][:10]
fn1 = tfidf1.get_feature_names_out()
print('\nTop 10 từ theo TF-IDF trung bình:')
for i in top10_1:
    print(f'  {fn1[i]:<20} {mean1[i]:.4f}')

print_subsection('1.5 Word2Vec Embeddings & từ gần nghĩa với "sạch sẽ"')
w2v1 = SimpleWord2Vec(vector_size=50, window=3, epochs=200, seed=42)
w2v1.fit(df1['tokens'].tolist())
print(f'Vocabulary size: {w2v1.vocab_size} từ')
w, sim1 = find_similar(w2v1, ['sạch', 'sẽ', 'thoáng'])
print_similar(w, sim1)

print_subsection('1.6 So sánh: Khi nào dùng TF-IDF? Khi nào dùng Word2Vec?')
print("""
┌──────────────────────┬──────────────────────────────┬────────────────────────────────┐
│ Tiêu chí             │ TF-IDF                       │ Word2Vec                       │
├──────────────────────┼──────────────────────────────┼────────────────────────────────┤
│ Biểu diễn            │ Sparse vector (BOW)          │ Dense vector ngữ nghĩa         │
│ Ngữ nghĩa            │ Không nắm được               │ Nắm được ngữ cảnh & đồng nghĩa│
│ Lượng dữ liệu        │ Ít dữ liệu vẫn dùng tốt     │ Cần nhiều dữ liệu để tốt       │
│ Tốc độ               │ Rất nhanh                    │ Chậm hơn (cần train)           │
│ Ứng dụng phù hợp     │ Search, keyword, phân loại  │ Sentiment, NER, dịch máy       │
└──────────────────────┴──────────────────────────────┴────────────────────────────────┘

→ Dùng TF-IDF khi: Cần tìm từ khóa quan trọng, phân loại tài liệu,
                    dữ liệu ít (< 1000 mẫu).
→ Dùng Word2Vec khi: Cần hiểu ngữ nghĩa, tìm từ đồng nghĩa,
                      phân tích cảm xúc sâu, có nhiều dữ liệu.
""")


print_section('BÀI 2: BÌNH LUẬN TRẬN ĐẤU (2đ)')

print_subsection('2.1 Nạp dữ liệu & mã hóa categorical')
df2 = pd.read_csv('ITA105_Lab_4_Match_comments.csv', encoding='utf-8')
print(f'Kích thước: {df2.shape}  |  Cột: {df2.columns.tolist()}')
print('\nMissing values:')
print(df2.isnull().sum().to_string())

df2['team'].fillna(df2['team'].mode()[0], inplace=True)
df2.dropna(subset=['comment_text'], inplace=True)
df2.reset_index(drop=True, inplace=True)

le2 = LabelEncoder()
df2['team_LE'] = le2.fit_transform(df2['team'])
print('\nLabel Encoding — team:')
print(pd.DataFrame({'team': le2.classes_, 'Mã': range(len(le2.classes_))}).to_string(index=False))

df2_ohe = pd.get_dummies(df2[['author']], columns=['author'], prefix='auth', dtype=int)
print(f'\nOne-Hot Encoding — author: {df2_ohe.shape[1]} cột được tạo')
print(f'  Ví dụ 3 cột đầu: {list(df2_ohe.columns[:3])}')

print_subsection('2.2 Tiền xử lý văn bản bình luận')
df2['tokens']     = df2['comment_text'].apply(preprocess_text)
df2['clean_text'] = df2['comment_text'].apply(preprocess_to_string)
print('Ví dụ:')
for i in range(3):
    print(f'  Gốc : {df2["comment_text"].iloc[i]}')
    print(f'  Sau : {df2["clean_text"].iloc[i]}')
    print()

print_subsection('2.3 Tạo TF-IDF Matrix')
tfidf2 = TfidfVectorizer(max_features=100)
X2 = tfidf2.fit_transform(df2['clean_text'])
print(f'TF-IDF Matrix: {X2.shape[0]} văn bản × {X2.shape[1]} từ')
mean2 = np.asarray(X2.mean(axis=0)).flatten()
top10_2 = np.argsort(mean2)[::-1][:10]
fn2 = tfidf2.get_feature_names_out()
print('\nTop 10 từ theo TF-IDF trung bình:')
for i in top10_2:
  print(f'  {fn2[i]:<20} {mean2[i]:.4f}')

print_subsection('2.4 Word2Vec Embeddings & từ gần nghĩa với "xuất sắc"')
w2v2 = SimpleWord2Vec(vector_size=50, window=3, epochs=200, seed=42)
w2v2.fit(df2['tokens'].tolist())
print(f'Vocabulary size: {w2v2.vocab_size} từ')
w, sim2 = find_similar(w2v2, ['xuất', 'sắc', 'tuyệt', 'hay'])
print_similar(w, sim2)

print_subsection('2.5 So sánh TF-IDF vs Word2Vec về biểu diễn ý nghĩa bình luận')
print("""
TF-IDF với bình luận trận đấu:
  ✓ Xác định từ khóa nổi bật: "bàn thắng", "foul", "penalty",...
  ✓ Tốt cho phân loại bình luận tích cực / tiêu cực đơn giản
  ✗ "hay" và "xuất sắc" bị coi như 2 từ hoàn toàn khác nhau
  ✗ Không hiểu cấu trúc câu hay ngữ điệu cảm xúc

Word2Vec với bình luận trận đấu:
  ✓ Học được "xuất sắc" ≈ "tuyệt vời" ≈ "ấn tượng" (gần nhau trong không gian vector)
  ✓ Hiểu ngữ cảnh: "ghi bàn" liên quan đến "tấn công", "đội hình"
  ✓ Phù hợp hơn để phân tích cảm xúc và entity recognition
  ✗ Cần tập dữ liệu lớn để học embedding chất lượng

→ Kết luận: Word2Vec phù hợp hơn cho bình luận thể thao vì ngôn ngữ
  cảm xúc phong phú, cần hiểu đồng nghĩa và ngữ cảnh phức tạp.
""")


print_section('BÀI 3: FEEDBACK NGƯỜI CHƠI (2đ)')

print_subsection('3.1 Nạp dữ liệu & Label / One-Hot Encoding')
df3 = pd.read_csv('ITA105_Lab_4_Player_feedback.csv', encoding='utf-8')
print(f'Kích thước: {df3.shape}  |  Cột: {df3.columns.tolist()}')
print('\nMissing values:')
print(df3.isnull().sum().to_string())

df3['device'].fillna(df3['device'].mode()[0], inplace=True)
df3.dropna(subset=['feedback_text'], inplace=True)
df3.reset_index(drop=True, inplace=True)

le3 = LabelEncoder()
df3['player_type_LE'] = le3.fit_transform(df3['player_type'])
print('\nLabel Encoding — player_type:')
print(pd.DataFrame({'player_type': le3.classes_, 'Mã': range(len(le3.classes_))}).to_string(index=False))

df3_ohe = pd.get_dummies(df3[['device']], columns=['device'], prefix='dev', dtype=int)
dev_cols = list(df3_ohe.columns)
print(f'\nOne-Hot Encoding — device → {dev_cols}')
print(df3_ohe.head().to_string())

print_subsection('3.2 Tiền xử lý văn bản feedback')
df3['tokens']     = df3['feedback_text'].apply(preprocess_text)
df3['clean_text'] = df3['feedback_text'].apply(preprocess_to_string)
print('Ví dụ:')
for i in range(3):
    print(f'  Gốc : {df3["feedback_text"].iloc[i]}')
    print(f'  Sau : {df3["clean_text"].iloc[i]}')
    print()

print_subsection('3.3 Tạo TF-IDF Matrix')
tfidf3 = TfidfVectorizer(max_features=100)
X3 = tfidf3.fit_transform(df3['clean_text'])
print(f'TF-IDF Matrix: {X3.shape[0]} văn bản × {X3.shape[1]} từ')
mean3 = np.asarray(X3.mean(axis=0)).flatten()
top10_3 = np.argsort(mean3)[::-1][:10]
fn3 = tfidf3.get_feature_names_out()
print('\nTop 10 từ theo TF-IDF trung bình:')
for i in top10_3:
    print(f'  {fn3[i]:<20} {mean3[i]:.4f}')

print_subsection('3.4 Word2Vec Embeddings & từ gần nghĩa với "đẹp"')
w2v3 = SimpleWord2Vec(vector_size=50, window=3, epochs=200, seed=42)
w2v3.fit(df3['tokens'].tolist())
print(f'Vocabulary size: {w2v3.vocab_size} từ')
w, sim3 = find_similar(w2v3, ['đẹp', 'hiệu', 'ứng', 'mượt'])
print_similar(w, sim3)

print_subsection('3.5 Lựa chọn TF-IDF hay Word2Vec để phân loại sentiment người chơi')
print("""
Phân tích tình huống:
  - Feedback game ngắn, cảm xúc trực tiếp: "đẹp", "lag", "chán", "mượt"
  - Người chơi dùng tiếng lóng, từ viết tắt, ngôn ngữ hỗn hợp

So sánh:
  TF-IDF: Đơn giản, nhanh, ổn nếu có từ khóa sentiment rõ ràng
          ✗ Không nhận ra "lag" ≈ "giật" ≈ "chậm"

  Word2Vec: Nắm ngữ nghĩa → phân loại chính xác hơn khi đủ dữ liệu
            ✓ "đẹp" ≈ "ấn tượng" ≈ "lung linh" ≈ "xịn"

→ Khuyến nghị: Dùng Word2Vec để phân loại sentiment người chơi.
  Lý do: Ngôn ngữ game phong phú, nhiều đồng nghĩa và tiếng lóng —
  cần biểu diễn ngữ nghĩa thay vì chỉ đếm từ.

  Lưu ý thực tế: Nếu tập dữ liệu nhỏ (< 500 mẫu), TF-IDF + Logistic Regression
  vẫn ổn định hơn vì Word2Vec chưa hội tụ tốt.
""")


print_section('BÀI 4: REVIEW ALBUM (2đ)')

print_subsection('4.1 Nạp dữ liệu & Label / One-Hot Encoding')
df4 = pd.read_csv('ITA105_Lab_4_Album_reviews.csv', encoding='utf-8')
print(f'Kích thước: {df4.shape}  |  Cột: {df4.columns.tolist()}')
print('\nMissing values:')
print(df4.isnull().sum().to_string())

df4['genre'].fillna(df4['genre'].mode()[0], inplace=True)
df4.dropna(subset=['review_text'], inplace=True)
df4.reset_index(drop=True, inplace=True)

le4 = LabelEncoder()
df4['genre_LE'] = le4.fit_transform(df4['genre'])
print('\nLabel Encoding — genre:')
print(pd.DataFrame({'genre': le4.classes_, 'Mã': range(len(le4.classes_))}).to_string(index=False))

df4_ohe = pd.get_dummies(df4[['platform']], columns=['platform'], prefix='pf', dtype=int)
pf_cols = list(df4_ohe.columns)
print(f'\nOne-Hot Encoding — platform → {pf_cols}')
print(df4_ohe.head().to_string())

print_subsection('4.2 Tiền xử lý review album')
df4['tokens']     = df4['review_text'].apply(preprocess_text)
df4['clean_text'] = df4['review_text'].apply(preprocess_to_string)
print('Ví dụ:')
for i in range(3):
    print(f'  Gốc : {df4["review_text"].iloc[i]}')
    print(f'  Sau : {df4["clean_text"].iloc[i]}')
    print()

print_subsection('4.3 Tạo TF-IDF Matrix')
tfidf4 = TfidfVectorizer(max_features=100)
X4 = tfidf4.fit_transform(df4['clean_text'])
print(f'TF-IDF Matrix: {X4.shape[0]} văn bản × {X4.shape[1]} từ')
mean4 = np.asarray(X4.mean(axis=0)).flatten()
top10_4 = np.argsort(mean4)[::-1][:10]
fn4 = tfidf4.get_feature_names_out()
print('\nTop 10 từ theo TF-IDF trung bình:')
for i in top10_4:
    print(f'  {fn4[i]:<20} {mean4[i]:.4f}')

row0 = pd.Series(X4[0].toarray().flatten(), index=fn4)
nonzero0 = row0[row0 > 0].sort_values(ascending=False)
print('\nTF-IDF vector review đầu tiên (top 8 từ ≠ 0):')
print(nonzero0.head(8).to_string())

print_subsection('4.4 Word2Vec Embeddings & từ gần nghĩa với "sáng tạo"')
w2v4 = SimpleWord2Vec(vector_size=50, window=3, epochs=200, seed=42)
w2v4.fit(df4['tokens'].tolist())
print(f'Vocabulary size: {w2v4.vocab_size} từ')
w, sim4 = find_similar(w2v4, ['sáng', 'tạo', 'độc', 'đáo', 'hòa'])
print_similar(w, sim4)

print('\n' + '=' * 65)
print('  TỔNG KẾT LAB 4')
print('=' * 65)
print("""
╔══════════════════════════════════════════════════════════════╗
║  Công thức TF-IDF:                                          ║
║    TF(t, d)  = số lần t xuất hiện trong d / tổng từ trong d ║
║    IDF(t)    = log( N / df(t) )   với N = tổng số văn bản   ║
║    TF-IDF    = TF × IDF                                     ║
╠══════════════════════════════════════════════════════════════╣
║  Word2Vec (Skip-gram):                                      ║
║    - Đầu vào: từ trung tâm (center word)                    ║
║    - Đầu ra: dự đoán các từ ngữ cảnh xung quanh            ║
║    - Kết quả: vector ngữ nghĩa cho mỗi từ                   ║
╠══════════════════════════════════════════════════════════════╣
║  Khi nào dùng TF-IDF:                                       ║
║    ✓ Ít dữ liệu, cần tốc độ nhanh                          ║
║    ✓ Tìm kiếm tài liệu, trích xuất từ khóa                 ║
║    ✓ Phân loại văn bản đơn giản                             ║
╠══════════════════════════════════════════════════════════════╣
║  Khi nào dùng Word2Vec:                                     ║
║    ✓ Nhiều dữ liệu, cần hiểu ngữ nghĩa sâu                 ║
║    ✓ Phân tích cảm xúc, tìm từ đồng nghĩa                  ║
║    ✓ Làm input cho mô hình học sâu (LSTM, Transformer)      ║
╚══════════════════════════════════════════════════════════════╝
""")
print('✅ Hoàn thành toàn bộ Lab 4!')