# GPU RUNBOOK — Chạy LLM parser trên máy GPU của thành viên

Người chạy: thành viên có GPU NVIDIA. Người nhận kết quả: lead (Khôi). Quyết định liên quan: `DECISIONS.md` D14, D15, D17 (pin vLLM).
Thời gian dự kiến: **3–6 giờ máy chạy** cho cả 8B và 14B (model gọi từng câu, ~4000 câu mỗi model, 14B chiếm phần lớn),
~30 phút thao tác. Nên bắt đầu buổi tối hoặc chạy qua đêm. Hạn: **10/11** (để kịp calibrate và đóng image trước 12/11).

## 0. Bạn cần làm gì, gửi lại gì

Máy bạn chỉ **chạy model và ghi lại đầu ra**. Mọi phân tích, chọn model, chỉnh chính sách đều do lead làm trên CPU từ
file bạn gửi. Bạn không cần đọc hay hiểu câu hỏi, không cần sửa code.

| Gửi lại | Ở đâu | Vì sao |
|---|---|---|
| `bundle_<run>.tgz` (replay, submission public, log) | **Thư mục Drive riêng của nhóm**, không chỗ nào khác | Có đầu ra model trên câu hỏi thi |
| `summary.md` (chỉ số liệu) | PR vào GitHub tại `docs/gpu_runs/<run>.md` | Lưu vết phiên bản vLLM, thời gian, VRAM |
| Tin nhắn ngắn | Nhóm chat | Báo xong, kèm lỗi nếu có |

## 1. Luật dữ liệu (đọc trước, bắt buộc)

Repo GitHub để **public**. Đưa dữ liệu thi lên đó là vi phạm (D13, D15).

1. **Không commit, không push, không dán lên issue/PR/chat công khai**: thư mục `data/`, mọi file `.jsonl`
   (replay, submission, log, few-shot), file `.tgz`, weights. `.gitignore`, pre-commit và CI đều chặn, nhưng đừng thử
   lách (ví dụ đổi đuôi file).
2. **Không mở** `data/public_test/**/questions.jsonl`, không mở dữ liệu hay câu hỏi của 6 shop holdout
   (`biz_103, biz_110, biz_117, biz_123, biz_125, biz_128`). Script chạy trên chúng là được. Không mở file replay,
   submission hay log sinh ra từ public/holdout.
3. Nếu có lỗi, gửi lead **traceback** (dòng lỗi), không gửi nội dung câu hỏi.
4. Không đổi tên thư mục model (`models/Qwen3-8B`…): tên thư mục là một phần khoá replay.
5. Không chỉnh code trên máy GPU. Có lỗi thì báo lead, lead sửa và push, bạn `git pull` rồi chạy lại.

## 2. Chuẩn bị máy

**Hệ điều hành:** vLLM chỉ chạy trên Linux. Máy Windows thì dùng **WSL2 Ubuntu 22.04/24.04** với driver NVIDIA bản
Windows mới (đã hỗ trợ WSL sẵn, không cài driver trong Ubuntu). Đặt repo trong ổ Linux (`~/...`), không đặt ở `/mnt/c`:
đọc file qua `/mnt/c` chậm hơn nhiều.

**RAM cho WSL2:** mặc định WSL chỉ được 50% RAM máy, trong khi ingest 30 shop cộng load 14B (~30 GB RAM) dễ bị kill
giữa chừng (log dừng đột ngột, không traceback). Tạo `%UserProfile%\.wslconfig` trên Windows:

```ini
[wsl2]
memory=48GB   # để lại ~8 GB cho Windows; máy 32 GB RAM thì đặt 26GB
swap=32GB
```

rồi chạy `wsl --shutdown` trong PowerShell và mở lại Ubuntu. Kiểm tra bằng `free -g`.

**Kiểm tra:** `nvidia-smi` phải thấy GPU. Ổ trống ≥ 60 GB (14B BF16 ~30 GB, 8B ~16 GB, môi trường Python ~10 GB).

**Chọn model theo VRAM** (gửi lead dòng `nvidia-smi` trước khi tải nếu không chắc):

| VRAM | Qwen3-8B | Qwen3-14B | Biến môi trường thêm |
|---|---|---|---|
| ≥ 40 GB 1 card (A6000, L40S, A100) | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B` | – |
| 2 card × 24 GB trở lên (vd 2× 5090, giống máy chấm) | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B` | `DATAMATE_LLM_TP=2` khi chạy 14B |
| 32 GB 1 card (RTX 5090) | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B-FP8` | – (14B BF16 ~29.5 GB, không vừa 1 card). FP8 lỗi kernel thì dùng `-AWQ` + float16 |
| 24 GB (3090, 4090) | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B-AWQ` | `DATAMATE_LLM_DTYPE=float16` khi chạy bản AWQ |
| 16 GB (4060 Ti 16G, 4080) | `Qwen/Qwen3-8B-AWQ` | `Qwen/Qwen3-14B-AWQ` | `DATAMATE_LLM_DTYPE=float16` |
| < 16 GB, hoặc card trước dòng RTX 30 | báo lead, không chạy | | |

**Mọi bản AWQ đều đặt `DATAMATE_LLM_DTYPE=float16`** (mặc định là bfloat16, kernel AWQ có thể từ chối). Chạy 8B BF16 và
14B AWQ trong cùng lệnh thì tách làm hai lệnh `run_all.sh`, biến env chỉ đặt cho lệnh AWQ. Bản FP8 giữ mặc định.
Bản AWQ/FP8 chỉ để so sánh tương đối. Nếu chỉ chạy được bản lượng tử hoá, ghi rõ trong tin nhắn.
Hết VRAM (`CUDA out of memory`): thử `DATAMATE_LLM_GPU_UTIL=0.92`, rồi `DATAMATE_LLM_MAX_LEN=10240`.

## 3. Cài đặt (một lần)

```bash
git clone https://github.com/dinhkhoi124/DateMate.git && cd DateMate
git checkout feat/llm-parser          # nhánh lead báo; sau khi merge thì dùng main
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r docker/requirements.txt   # pandas/numpy… đúng bản máy lead: few-shot sinh ra phải giống hệt
pip install -e ".[dev,llm]"           # vLLM 0.10.1.1 (pin, D17) + torch 2.7.1 CUDA 12.8
pip freeze | grep -iE '^(pandas|numpy|openpyxl|python-calamine|duckdb|pyarrow)=='
pre-commit install                    # chặn commit nhầm dữ liệu
```

So dòng `pip freeze` với `docker/requirements.txt`. Nếu vLLM kéo bản numpy/pandas khác (pip báo conflict hoặc số khác
nhau), **gửi lead output của pip**, đừng tự ép phiên bản. Bước kiểm sha256 few-shot bên dưới sẽ cho biết có lệch thật không.

Không có `python3.11`: `sudo apt install python3.11 python3.11-venv` (Ubuntu 24.04 cần PPA deadsnakes).
vLLM được pin **0.10.1.1**, trùng base image BTC khuyến nghị, để replay khớp với image nộp bài (D17). **Không nâng
cấp vLLM, torch hay numpy**, kể cả khi pip gợi ý. Driver NVIDIA phải hỗ trợ CUDA 12.8 (`nvidia-smi` góc phải trên ghi
`CUDA Version: 12.8` trở lên, tức driver ≥ 570). Thấp hơn thì cập nhật driver trên Windows trước.
Bản này dùng API `GuidedDecodingParams`; smoke test ghi `structured_api` để xác nhận.

**Dữ liệu:** lấy từ Drive nhóm (cùng bộ lead đang dùng), giải nén để có đúng cấu trúc:

```
data/train/train/{questions.jsonl, answers.jsonl, rulebook.md, calendar.csv, data/...}
data/public_test/public_test/{questions.jsonl, rulebook.md, calendar.csv, data/...}
```

**Few-shot** (sinh từ câu dev, mất vài phút, phải giống hệt máy lead):

```bash
PYTHONPATH=src python analysis/build_fewshot.py
sha256sum src/datamate/parse/fewshot.jsonl
```

Mã sha256 phải là `072b28612da1a1c2bc1730d24e50ca70fea20c9e3e3cf069fc49eae4d81b121a` (commit hiện tại). Khác thì
**dừng**, báo lead: replay sẽ không dùng được.

**Tải model** (cần mạng, chỉ bước này):

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-8B  --local-dir models/Qwen3-8B
hf download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
# bản lượng tử (nếu bảng trên bảo vậy): hf download Qwen/Qwen3-14B-AWQ --local-dir models/Qwen3-14B-AWQ, tương tự -FP8
```

Ví dụ máy 24 GB:

```bash
bash scripts/gpu/run_all.sh models/Qwen3-8B
DATAMATE_LLM_DTYPE=float16 bash scripts/gpu/run_all.sh models/Qwen3-14B-AWQ
```

## 4. Smoke test (5–10 phút, làm trước khi chạy lớn)

```bash
source .venv/bin/activate
PYTHONPATH=src python eval/gpu_smoke.py --model models/Qwen3-8B --out eval/runs/gpu/smoke_Qwen3-8B.json
```

Câu hỏi trong smoke test là câu tự viết, không thuộc bộ thi. Cần thấy `SMOKE OK`. Smoke kiểm:
phiên bản (torch, CUDA, vLLM, xgrammar), API ép JSON đang dùng, thời gian load, mọi output là JSON hợp lệ có `op`
đúng danh sách, và chạy lại temp 0 ra y hệt.

- `SMOKE FAILED` vì `deterministic_n1: false`: vẫn chạy bước 5, ghi chú cho lead.
- `valid` < 4/4 hoặc traceback: **dừng**, gửi lead file `eval/runs/gpu/smoke_*.json` (an toàn, không có dữ liệu thi)
  và traceback.

## 5. Chạy chính

```bash
source .venv/bin/activate
bash scripts/gpu/run_all.sh models/Qwen3-8B models/Qwen3-14B
```

Có biến môi trường ở bảng mục 2 thì đặt trước lệnh, ví dụ `DATAMATE_LLM_TP=2 bash scripts/gpu/run_all.sh ...`.
Chạy trong `tmux` hoặc `nohup` để tắt terminal không làm dừng.

Với mỗi model, script chạy: smoke → dev (24 shop) → holdout (6 shop, chỉ ra số tổng) → public (sinh submission).
Model được gọi trên **mọi** câu (`DATAMATE_LLM_CALL_ALL=1`) để lead thử được mọi chính sách ensemble trên CPU mà
không cần chạy lại GPU. Câu trả lời cuối vẫn theo chính sách mặc định, nên bảng PA trong summary là PA thật.

Sau mỗi split, summary có dòng `llm_calls=N llm_failed=K`. Nếu terminal in `WARNING <split>` (K > 0 hoặc không có
dòng này) thì model đã lỗi và split đó chỉ chạy bằng rule: gửi lead các dòng `[llm] generate failed` cùng traceback
trong `eval/runs/gpu/<run>/logs/<split>_<model>.txt` (chỉ dòng lỗi, không gửi cả file).

Kết thúc, script in ra hai đường dẫn:

- `eval/runs/gpu/<run>/summary.md`: chỉ có số liệu.
- `eval/runs/gpu/bundle_<run>.tgz`: **chỉ upload lên Drive nhóm**.

**Pha 2** (chỉ khi lead nhắn, sau khi đã chọn model): `SAMPLES=5 bash scripts/gpu/run_all.sh models/<model>` để
lấy 5 mẫu mỗi câu cho self-consistency.

## 6. Gửi kết quả

1. Upload `bundle_<run>.tgz` lên thư mục Drive nhóm `DataMate/gpu_runs/`.
2. Mở PR chỉ chứa summary. Cần quyền ghi vào repo (lead mời làm collaborator). Chưa có quyền thì fork repo, push lên
   fork rồi mở PR từ fork. Tên run có `hostname` máy bạn và sẽ hiện trên repo public; muốn ẩn thì đổi tên máy hoặc
   đổi tên file khi copy (giữ phần ngày giờ).
   ```bash
   git checkout -b chore/gpu-run-<run>
   mkdir -p docs/gpu_runs && cp eval/runs/gpu/<run>/summary.md docs/gpu_runs/<run>.md
   git add docs/gpu_runs/<run>.md && git commit -m "GPU run <run>: summary"
   git push -u origin chore/gpu-run-<run>
   ```
   Trước khi commit, mở `summary.md` xem lại: chỉ có phiên bản, thời gian, VRAM, bảng PA và 3 dòng cuối smoke test.
   Thấy câu hỏi tiếng Việt nào khác 4 câu smoke thì **không commit**, báo lead.
3. Nhắn nhóm: tên run, GPU, model nào chạy được, lỗi nếu có.

## 7. Lỗi hay gặp

| Triệu chứng | Cách xử lý |
|---|---|
| Log dừng giữa chừng, không traceback, `dmesg` có `Killed process` | Hết RAM trong WSL: tăng `memory`/`swap` ở mục 2 |
| `WARNING <split>` / `llm_failed` > 0 | Mục 5: gửi lead dòng `[llm] generate failed` + traceback |
| `No module named vllm` | Chưa `source .venv/bin/activate`, hoặc cài thiếu `.[llm]` |
| `CUDA out of memory` | Mục 2: tăng `DATAMATE_LLM_GPU_UTIL`, giảm `DATAMATE_LLM_MAX_LEN`, hoặc dùng bản AWQ |
| `Bfloat16 is only supported on GPUs with compute capability >= 8.0` | Card cũ: `DATAMATE_LLM_DTYPE=float16` |
| `missing data/train/train` | Giải nén sai cấu trúc, xem mục 3 |
| Một split `exit 1` trong summary | Gửi lead dòng cuối của `eval/runs/gpu/<run>/logs/<split>_<model>.txt` (traceback), không gửi cả file |
| pre-commit báo `no-competition-data` hoặc `no-verbatim-questions` | Đúng là đang commit nhầm dữ liệu. Bỏ file đó khỏi commit |
