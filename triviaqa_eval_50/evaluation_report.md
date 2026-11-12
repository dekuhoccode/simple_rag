# Báo Cáo Đánh Giá Hiệu Năng Simple RAG (TriviaQA Benchmark)
- **Thời gian thực hiện:** 2026-11-12 08:22:45
- **Số lượng câu hỏi:** 50 / 50
- **Mô hình Embedding:** `BAAI/bge-small-en-v1.5` (Top-K=3)
- **Mô hình Sinh (LLM):** `gemini-3.8-flash` (https://aiplatform.googleapis.com/v1/projects/my-first-rag-514304/locations/global/publishers/google/models)

## 1. Kết Quả Tổng Hợp (Summary Metrics)

### A. Tầng Truy Xuất (Retrieval Metrics)
| Chỉ số | Điểm số | Ý nghĩa |
| :--- | :--- | :--- |
| **Hit Rate@3** | `68.0%` | Tỷ lệ câu hỏi tìm thấy đáp án trong Top-3 |
| **MRR@3** | `0.57` | Thứ hạng đảo trung bình của kết quả đúng đầu tiên |
| **Recall@3** | `36.0%` | Tỷ lệ bao phủ từ khóa đáp án |

### B. Tầng Sinh Câu Trả Lời (Generation Metrics)
| Mô hình | Exact Match (EM) | F1-Score |
| :--- | :--- | :--- |
| **Simple RAG** | **44.0%** | **47.0%** |
| **No-RAG (Parametric)** | 0.0% | 0.0% |
| **Chênh lệch (RAG - No-RAG)** | **+44.0%** | - |

### C. Tầng Đánh Giá Chất Lượng Toàn Diện (RAG Triad Metrics)
| Chỉ số | Điểm số | Ý nghĩa |
| :--- | :--- | :--- |
| **Context Relevance** | `74.15%` | Tỷ lệ ngữ cảnh truy xuất thực sự hữu ích cho câu hỏi |
| **Answer Relevance** | `38.38%` | Mức độ phù hợp và thỏa đáng của câu trả lời |
| **Groundedness (Faithfulness)** | `64.0%` | Mức độ trung thực dựa trên tài liệu, không ảo giác |

## 2. Chi Tiết Từng Câu Hỏi Kiểm Thử (Samples Breakdown)

### #1. Who first hosted Family Feud?
- **Đáp án chuẩn (Ground Truth):** `Richard Dawson`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.5333, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #2. Who directed the classic 30s western Stagecoach?
- **Đáp án chuẩn (Ground Truth):** `John Ford`
- **Simple RAG:** `John Ford` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.5714, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #3. What are the international registration letters of a vehicle from Iceland?
- **Đáp án chuẩn (Ground Truth):** `IS`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.3333)
- **RAG Triad:** Context Rel=0.9091, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #4. In which North African country was Mohammed Al-Fayed, owner of Harrods department store in London, born?
- **Đáp án chuẩn (Ground Truth):** `Egypt`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7556, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #5. Which grand slam did Pete Sampras not win in the 20th century?
- **Đáp án chuẩn (Ground Truth):** `French`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7222, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #6. In which American state is the Isabella Stewart Gardner Museum?
- **Đáp án chuẩn (Ground Truth):** `Massachusetts`
- **Simple RAG:** `Massachusetts` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=1.0, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #7. Who was the first person after Scott to reach the South Pole overland?
- **Đáp án chuẩn (Ground Truth):** `Edmond Hillary`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=1.0, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #8. Who bought Chelsea football club for £1 in 1982?
- **Đáp án chuẩn (Ground Truth):** `Ken Bates`
- **Simple RAG:** `Ken Bates` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.963, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #9. River Phoenix died during the making of which movie?
- **Đáp án chuẩn (Ground Truth):** `Dark Blood`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.8519, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #10. How many pieces are there in a standard set of dominoes?
- **Đáp án chuẩn (Ground Truth):** `28`
- **Simple RAG:** `28` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.7879, Answer Rel=0.5, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #11. Which 90s sci fi series with James Belushi was based on Bruce Wagner's comic strip of the same name?
- **Đáp án chuẩn (Ground Truth):** `Wild Palms`
- **Simple RAG:** `Wild Palms` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.7193, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #12. Which was the only eastern bloc country to participate in the 1984 LA Olympics?
- **Đáp án chuẩn (Ground Truth):** `Rumania`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.8462, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #13. "In which movie did Garbo say, ""I want to be alone""."
- **Đáp án chuẩn (Ground Truth):** `Grand Hotel`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.4848, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #14. Who was the only female victim to be killed off in the opening credit sequence in Police Squad?
- **Đáp án chuẩn (Ground Truth):** `Florence Henderson`
- **Simple RAG:** `Florence Henderson` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.6667, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #15. Who wrote My Life and Loves in 1926, originally banned in the US?
- **Đáp án chuẩn (Ground Truth):** `Frank Harris`
- **Simple RAG:** `Frank Harris` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.8889, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #16. From which animal does Haggis come?
- **Đáp án chuẩn (Ground Truth):** `Sheep`
- **Simple RAG:** `sheep` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.5556, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #17. Dave Gilmore and Roger Waters were in which rock group?
- **Đáp án chuẩn (Ground Truth):** `Pink Floyd`
- **Simple RAG:** `Pink Floyd` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.8, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #18. What name is given to the home of a beaver?
- **Đáp án chuẩn (Ground Truth):** `Lodge`
- **Simple RAG:** `lodges` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.9333, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #19. What is Harrison Ford's real name?
- **Đáp án chuẩn (Ground Truth):** `Harrison Ford`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.3333, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #20. What is the most popular natural attraction in Alaska?
- **Đáp án chuẩn (Ground Truth):** `Mount McKinley`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7407, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #21. Who beat Tim Henman in his first Wimbledon singles semifinal?
- **Đáp án chuẩn (Ground Truth):** `Pete Sampras`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.9333, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #22. What is the longest word can be typed using only the top row of letters on a typewriter?
- **Đáp án chuẩn (Ground Truth):** `Typewriter`
- **Simple RAG:** `uropyoureter` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7843, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #23. In WWII, who was the head of the Nazi party's security service?
- **Đáp án chuẩn (Ground Truth):** `Reinhard Heidrich`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.9697, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #24. What claimed the life of singer Kathleen Ferrier?
- **Đáp án chuẩn (Ground Truth):** `Cancer`
- **Simple RAG:** `cancer` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.75, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #25. Nigel Hawthorne was Oscar nominated for The Madness of which King?
- **Đáp án chuẩn (Ground Truth):** `George`
- **Simple RAG:** `King George III` (EM=0.0, F1=0.5)
- **RAG Triad:** Context Rel=0.9091, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #26. Which builder of steam engines formed a successful partnership with Matthew Boulton?
- **Đáp án chuẩn (Ground Truth):** `James Watt`
- **Simple RAG:** `James Watt` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.8889, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #27. Which film director guested as the FBI Director in The Silence of the Lambs?
- **Đáp án chuẩn (Ground Truth):** `Roger Corman`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.9697, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #28. In Hanna and Barbera's TV cartoons base on The Addams Family who was the voice of Gomez?
- **Đáp án chuẩn (Ground Truth):** `John Astin`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7917, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #29. "What is the more common name of the ""simple plurality"" system of voting?"
- **Đáp án chuẩn (Ground Truth):** `First past the post`
- **Simple RAG:** `first-past-the-post` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.6667, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #30. What is the name of the large buttock and thigh muscle?
- **Đáp án chuẩn (Ground Truth):** `Gluteus maximus`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.8, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #31. Who was the defending champion when Martina Navratilova first won Wimbledon singles?
- **Đáp án chuẩn (Ground Truth):** `Virginia Wade`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.8333, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #32. What type of seaweed grows up to 100 meters in length, and is farmed for its alginates?
- **Đáp án chuẩn (Ground Truth):** `Kelp`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.6275, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #33. What raw material is used for making glass?
- **Đáp án chuẩn (Ground Truth):** `Sand`
- **Simple RAG:** `pure chemicals` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.9167, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #34. Throughout the 80s and 90s Phil Collins recorded on which record label?
- **Đáp án chuẩn (Ground Truth):** `Atlantic`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.5556, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #35. What was President Gerald Ford's middle name?
- **Đáp án chuẩn (Ground Truth):** `Rudolph`
- **Simple RAG:** `Rudolph` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.6667, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #36. In which arcade game, would you find ghosts called Inky, Pinky, Blinky and Clyde?
- **Đáp án chuẩn (Ground Truth):** `Pac-Man`
- **Simple RAG:** `Pac-Man` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.6667, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #37. The Canary Islands are named after what animal?
- **Đáp án chuẩn (Ground Truth):** `Dog`
- **Simple RAG:** `dogs` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.9167, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #38. Who was born first, Kiefer Sutherland or Christian Slater?
- **Đáp án chuẩn (Ground Truth):** `Kiefer Sutherland`
- **Simple RAG:** `Kiefer Sutherland` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.5185, Answer Rel=0.83, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #39. How many chambers has the heart?
- **Đáp án chuẩn (Ground Truth):** `Four`
- **Simple RAG:** `four` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.3333, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #40. Who had a 70s No 1 hit with Let's Do It Again?
- **Đáp án chuẩn (Ground Truth):** `The Staple Singers`
- **Simple RAG:** `The Staple Singers` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.5556, Answer Rel=0.5, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #41. Which film features the characters Sally Albright and Harry Burns?
- **Đáp án chuẩn (Ground Truth):** `When Harry Met Sally`
- **Simple RAG:** `When Harry Met Sally...` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.8, Answer Rel=0.53, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #42. In which city was John Lennon murdered?
- **Đáp án chuẩn (Ground Truth):** `New York`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.8571, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #43. What is the most abundant substance in the plant kingdom, which no mammal produces the enzyme to digest?
- **Đáp án chuẩn (Ground Truth):** `Cellulose`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.5833, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #44. The Kray twins were convicted of the murder of George Cornell in the East End of London in 1966. What was the name of the pub in which he was shot?
- **Đáp án chuẩn (Ground Truth):** `The Blind Beggar`
- **Simple RAG:** `Blind Beggar` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.9365, Answer Rel=0.33, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #45. In which country was the first permanent bungee jumping site situated?
- **Đáp án chuẩn (Ground Truth):** `New Zealand`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.7273, Answer Rel=0.3, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #46. How old was Jimi Hendrix when he died?
- **Đáp án chuẩn (Ground Truth):** `27`
- **Simple RAG:** `27` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.6667, Answer Rel=0.5, Groundedness=0.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #47. Art Garfunkel trained for which profession although he didn't qualify?
- **Đáp án chuẩn (Ground Truth):** `Architect`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.4667, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #48. On the Internet what is Spam?
- **Đáp án chuẩn (Ground Truth):** `Junk mail`
- **Simple RAG:** `an unsolicited message` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.5556, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #49. Who did Dr. Crippen murder?
- **Đáp án chuẩn (Ground Truth):** `His wife`
- **Simple RAG:** `Cora Henrietta Crippen` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.5333, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)

### #50. What was the Paramount Film Company originally called?
- **Đáp án chuẩn (Ground Truth):** `Famous Players Film Company`
- **Simple RAG:** `Famous Players-Lasky Corporation` (EM=0.0, F1=0.6667)
- **RAG Triad:** Context Rel=0.8333, Answer Rel=0.3, Groundedness=1.0
- **No-RAG:** `[Lỗi gọi LLM API (403)]` (EM=0.0, F1=0.0)
