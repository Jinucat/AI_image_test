# AI_image_test
embed and tsne, umap check

- AI Hub 데이터 수집 ( data directory )

- 이미지 임베딩
- TSNE 변환
- 데이터셋이 수만개 이상일 경우 UMAP로 변환

*데이터는 따로 빼두었음
---구조---
+---data
|   +---academic_data
|   |   \---paper_001
|   |       \---images
|   |           +---Humanities
|   |           +---Society
|   |           \---Technology
|   +---academic_paper
|   |   +---Humanities
|   |   +---Society
|   |   \---Technology
|   \---skin_faces
|       +---건선
|       |   +---Psoriasis_Frontal
|       |   \---Psoriasis_side
|       +---아토피
|       |   +---Atopy_Frontal
|       |   \---Atopy_Side
|       +---여드름
|       |   +---Acne_Frontal
|       |   \---Acne_Side
|       +---정상
|       |   +---Normal_Front
|       |   \---Normal_Side
|       +---주사
|       |   +---Injection_Frontal
|       |   \---Injection_Side
|       \---지루
|           +---Seborrhea_Frontal
|           \---Seborrhea_Side
