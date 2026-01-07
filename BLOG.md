# Đánh bại nhà cái trong đêm cùng Machine Learning và Claude Code

*Câu chuyện về một đêm mất ngủ, vài dòng prompt, và một model có edge thực sự*

---

## Phần 1: Để Claude "nhìn" dữ liệu

### Bước đầu tiên: Hiểu mình đang có gì

Dữ liệu bóng đá từ [football-data.co.uk](https://www.football-data.co.uk/) - miễn phí, đầy đủ từ 2010, cập nhật hàng tuần. Tôi có:

- **2,100 trận đấu** từ mùa 2020-21 đến 2025-26
- Số liệu: góc, sút, phạm lỗi, thẻ vàng/đỏ, bàn thắng...
- Target: **Tổng số phạt góc mỗi trận**

Nhưng số liệu thô thì vô nghĩa. Tôi cần *hiểu* nó.

Thay vì tự viết code visualization, tôi prompt Claude:

```
Tạo một notebook để explore dữ liệu Premier League corners.
Tôi muốn thấy: distribution, trends theo mùa, so sánh team,
và correlation giữa các features.
```

5 phút sau, tôi có một notebook hoàn chỉnh với Plotly charts. Không phải matplotlib xấu xí, mà là interactive charts xịn xò.

### Những insights đầu tiên

**Distribution của Total Corners:**

```
Mean: 10.32
Std:  3.41
Min:  2
Max:  24
```

Trung bình 10 góc mỗi trận. Standard deviation ~3.4. Điều này quan trọng vì nó cho thấy **dự đoán chính xác số góc gần như bất khả thi** - variance quá cao.

**Home vs Away:**

```
Home corners: mean = 5.63
Away corners: mean = 4.68
```

Đội nhà có lợi thế ~1 góc. Không surprising, nhưng cần ghi nhớ.

**Feature Correlations:**

| Feature | Correlation với Total Corners |
|---------|------------------------------|
| Home Corners (HC) | 0.627 |
| Away Corners (AC) | 0.526 |
| Home Shots (HS) | 0.193 |
| Away Shots (AS) | 0.136 |
| Total Fouls | -0.112 |

Shots có correlation dương với corners - logic, vì đội tấn công nhiều sẽ có nhiều cơ hội phạt góc. Fouls thì ngược lại - nhiều phạm lỗi = ít tấn công = ít góc.

**Team Analysis:**

Claude tạo một scatter plot chia 4 góc phần tư:

- **Top-right**: Man City, Liverpool - nhiều góc tấn công, ít góc phòng ngự (strong cả 2 chiều)
- **Bottom-left**: Ipswich, West Brom - ngược lại (yếu cả 2 chiều)
- **Others**: Đa phần ở giữa, không có pattern rõ ràng

Insight quan trọng: **Chỉ có vài đội "elite" có behavior khác biệt đáng kể.**

---

## Phần 2: Regression - Thất bại đầu tiên

### Ý tưởng ban đầu

*"Dự đoán chính xác số góc, sau đó so với threshold để quyết định Over/Under."*

Nghe hợp lý, phải không? Tôi prompt Claude:

```
Train một XGBoost regression model để predict total corners.
Dùng rolling window features từ 5 trận gần nhất.
```

### Kết quả: Thảm họa

```
RMSE: 3.37
R²: -0.03 (tệ hơn cả dự đoán bằng mean!)
Prediction Range: 10.0 - 11.0 (actual: 3-24)
```

Model predict **10-11 góc cho MỌI trận đấu**. Nó học được một điều duy nhất: mean là 10.32, nên cứ predict gần đó là "an toàn" nhất.

Đây là **mean regression** - model quá sợ sai nên không dám predict extreme values.

### Iteration với Claude

Tôi không bỏ cuộc. Prompt tiếp:

```
Model bị mean regression. Thử:
1. Giảm regularization
2. Đổi sang Poisson objective (count data)
3. Tăng max_depth
```

**Iteration 2-6:** Thử đủ thứ - LightGBM, XGBoost, different hyperparameters. Kết quả tốt nhất:

```
RMSE: 3.32
Correlation: 0.15
Prediction Range: 7.7 - 12.6
```

Tốt hơn, nhưng vẫn không đủ. Và quan trọng hơn: **win rate khi betting không tốt hơn random.**

### Bài học từ thất bại

Sau 6 iterations, Claude và tôi nhận ra một điều:

> **Predicting exact corner count là bài toán quá khó. Nhưng ta không cần predict exact - ta chỉ cần predict Over hay Under.**

Đây là pivotal moment. Thay vì solve hard problem (regression), ta solve easier problem (classification).

---

## Phần 3: Classification - Ánh sáng cuối đường hầm

### Reframe bài toán

Tôi prompt Claude:

```
Chuyển sang classification approach. Thay vì predict số góc,
predict trực tiếp Over/Under cho từng threshold (8.5, 9.5, 10.5, 11.5, 12.5).

Dùng ensemble: LightGBM + XGBoost + RandomForest với probability calibration.
```

### Tại sao điều này quan trọng?

Với regression: Predict 10.8 khi actual là 11 → sai hoàn toàn cho O/U 10.5
Với classification: Predict P(Over 10.5) = 0.65 → đúng nếu > 0.5

Classification cho phép model tập trung vào **decision boundary** thay vì exact value.

### Kết quả đầu tiên: Có hy vọng!

```
O/U 9.5:  Confidence > 0.58 → 138 bets, 58.0% win, +10.7% ROI
O/U 10.5: Confidence > 0.55 → 37 bets, 64.9% win, +23.9% ROI
```

Nhưng khoan - 58% win rate có phải là "edge" thực sự không?

### Phân biệt Edge thực vs Edge ảo

Đây là trap mà nhiều người rơi vào. Nhìn kết quả này:

| Threshold | Win Rate | ROI |
|-----------|----------|-----|
| O/U 8.5   | 65.8%    | +25.7% |
| O/U 11.5  | 68.9%    | +31.6% |
| O/U 12.5  | 80.5%    | +53.8% |

Wow, O/U 12.5 có 80% win rate! Giàu to rồi!

**Sai.** Base rate của Under 12.5 là ~78%. Model chỉ đang bet Under mọi lúc và "ăn may" vì Under xảy ra nhiều hơn.

**Real Edge = Model Accuracy - max(Base Rate, 1 - Base Rate)**

| Threshold | Win Rate | Base Rate | Real Edge |
|-----------|----------|-----------|-----------|
| O/U 8.5   | 65.8%    | 65%       | ~0%       |
| O/U 9.5   | 58.0%    | 52%       | **+6%**   |
| O/U 10.5  | 64.9%    | 58%       | **+7%**   |
| O/U 12.5  | 80.5%    | 78%       | ~2%       |

**Chỉ có O/U 9.5 và O/U 10.5 có real edge đáng kể** - vì đây là những threshold gần 50/50, nơi model thực sự phải "học" thay vì follow base rate.

---

## Phần 4: Fine-tuning đến lúc "đủ tốt"

### Iterative Improvement với Claude

Từ đây, tôi và Claude bắt đầu một vòng lặp:

```
1. Train model với config X
2. Analyze results: win rate, edge, ROI, bet volume
3. Identify issues: overfitting? underfitting? wrong features?
4. Adjust và repeat
```

**Iteration 1: Thêm data**

```
Extend training data từ 2015-2026 thay vì 2020-2026.
~5,400 matches thay vì 2,100.
```

Kết quả: Edge tăng từ +3.5% lên +5.5% cho O/U 9.5. More data = better generalization.

**Iteration 2: Rolling Window**

```
Test windows: 5, 7, 10 trận gần nhất
```

Kết quả:
- Window=5 tốt nhất cho O/U 9.5 (form gần quan trọng)
- Window=10 tốt nhất cho O/U 10.5 (longer pattern)
- Window=7 underperform (không đủ long-term, không đủ recent)

**Iteration 3: Feature Selection**

Claude analyze feature importance:

```
Top features: shots, volatility, O/U historical rates
Least important: fouls, individual team O/U rates
```

Tôi thử:
- Remove low-importance features → Edge giảm
- Add new features (momentum, streaks) → Edge giảm
- Keep baseline 32 features → Best performance

**Bài học: Không phải lúc nào thêm features cũng tốt. Noise có thể overwhelm signal.**

**Iteration 4: Hyperparameter Tuning**

```
Test: depth, learning_rate, regularization
```

Final best configs:

| Threshold | Window | Confidence | Edge | ROI |
|-----------|--------|------------|------|-----|
| O/U 9.5   | 5      | > 0.58     | +9.1%| +21.5% |
| O/U 10.5  | 10     | > 0.54     | +4.1%| +18.6% |

---

## Phần 5: Deploy với Claude Code

### Từ notebook đến production

Có model rồi, giờ cần deploy. Tôi prompt:

```
Tạo web app với:
- Backend: Flask API trên Railway
- Frontend: React single-file trên Vercel
- Load models từ HuggingFace khi startup
- Call OpenRouter LLM để generate analysis
```

**6 giờ sau**, tôi có:

1. **API endpoint** nhận home_team + away_team, trả về predictions cho tất cả thresholds
2. **Frontend** với team selector, probability bars, LLM analysis
3. **Auto-deploy** khi push to GitHub

Cái hay của Claude Code: nó handle được cả những edge cases như:
- CORS configuration cho cross-origin requests
- Model download từ HuggingFace (vì Heroku/Railway có size limit)
- Error handling khi API fails

### Kết quả cuối cùng

Live tại: **corner.qnguyen3.dev**

Features:
- Chọn 2 đội từ dropdown
- Xem predictions cho 5 thresholds
- Team statistics và head-to-head
- AI-generated analysis với betting recommendations

---

## Phần 6: Bài học rút ra

### 1. Reframe bài toán quan trọng hơn tune model

Chuyển từ regression sang classification là breakthrough lớn nhất. Không phải hyperparameter tuning, không phải feature engineering - mà là **hiểu đúng bài toán cần solve.**

### 2. Edge ≠ Win Rate

Đừng tự lừa mình với số đẹp. 80% win rate nghe hay nhưng có thể chỉ là follow base rate. Always calculate **real edge against naive baseline.**

### 3. Claude Code như một pair programmer

Tôi không code một mình. Mọi iteration đều là:
- Tôi: idea + domain knowledge
- Claude: implementation + technical suggestions

Nó không replace tôi - nó **amplify** tôi. 48 giờ với Claude = có lẽ 2 tuần nếu code một mình.

### 4. Know when to stop

Model cuối cùng có edge +4% đến +9% tùy threshold. Không phải "đánh bại nhà cái mọi lúc" - nhưng đủ để có lợi nhuận kỳ vọng dương **nếu bet đủ nhiều và đủ lâu.**

Và đó là realistic expectation. Ai hứa hẹn 90% win rate đang nói dối bạn.

---

## Kết luận

Tôi bắt đầu đêm đó với câu hỏi: *"Liệu ML có thể predict corners không?"*

Câu trả lời: **Có, nhưng chỉ ở mức edge nhỏ, với điều kiện đúng, và cần patience.**

Quan trọng hơn cả kết quả: **Process**. Cách Claude Code cho phép tôi iterate nhanh, fail fast, và learn từ mỗi failure.

Bạn có thể dùng approach này cho bất kỳ bài toán nào - không nhất thiết phải là betting. Sports analytics, stock prediction, customer churn... Pattern là giống nhau:

1. Hiểu data
2. Start simple, fail fast
3. Reframe nếu cần
4. Iterate với tight feedback loop
5. Know when good enough is good enough

---

*P.S: Đừng cá độ bằng tiền bạn không sẵn sàng mất. Model này có edge dương, nhưng variance vẫn cao. Đây là educational project, không phải financial advice.*

*P.P.S: Code đầy đủ tại [github.com/qnguyen3/SCOPE](https://github.com/qnguyen3/SCOPE)*

---

*Viết lúc 4 giờ sáng, sau một đêm nữa không ngủ. Lần này vì viết blog, không phải train model.*
