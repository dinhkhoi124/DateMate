# DECISIONS — Nhật ký quyết định

Mỗi quyết định ghi: **bối cảnh → lựa chọn → lý do → hệ quả**. Quyết định mới thêm lên đầu. Đổi quyết định cũ thì ghi entry mới, đánh dấu entry cũ là `Superseded by Dxx`.

---

### D18 — Đối chiếu rulebook theo dòng lúc chạy: tham số đọc lại, định nghĩa đổi thì abstain (2026-11-07)
- **Bối cảnh:**
  - D16 bước 3: phải dùng rulebook của đúng split. Bảng kê `docs/RULEBOOK_PARAMS.md` cho thấy mới đọc lúc chạy được
    4 tham số. Khoảng 35 dòng định nghĩa (doanh thu, đơn hợp lệ, nợ, kho…) đặt cứng trong executor.
  - Rulebook khác train mà executor vẫn tính theo train thì ra câu sai tự tin (−1). Public đã thêm ngành mới và luật
    mới (EXPERIMENTS 04/11); private chắc chắn có rulebook riêng.
- **Lựa chọn:** `datamate/rulebook.py` tách rulebook split thành dòng luật (mỗi dòng bảng là một dòng; phần văn xuôi
  của mỗi mục là một dòng), so với chỉ mục sinh từ rulebook train (`rulebook_index.json`, chỉ có hash và mã `§m.k`,
  không có chữ):
  - Trùng hash nguyên văn đã chuẩn hoá → luật không đổi.
  - Chỉ trùng **khuôn** (số và tên thứ thay bằng ký hiệu), dòng là tham số (ca, Tết, đầu tuần) → đọc giá trị mới.
  - Khác, hoặc dòng train bị mất → **chặn phạm vi của dòng đó**: theo op, theo filter (ca, size, kê đơn…) hoặc theo
    loại kỳ (`week`, `last_n`, `tet`…). Intent chạm phạm vi thì abstain, log ghi mã dòng.
  - Dòng mới (khoá không có trong train) → các cụm từ của dòng không xuất hiện ở đâu trong rulebook train (khoá, chữ
    in đậm) thành từ khoá. Câu hỏi chứa từ khoá thì abstain.
  - Mục toàn cục (§0 mở đầu, §7 không trả lời được, §8 cách ghi file, §9 format, phần giới thiệu §6): đổi thì **chỉ
    ghi log**, không chặn.
- **Lý do:**
  - Precision trước (D11, D16 mục 4): sai do lệch luật là lỗi tự tin, không gate nào khác bắt được.
  - Chặn theo phạm vi hẹp (op/filter/loại kỳ) thay vì cả mục, để một dòng đổi không xoá cả hệ thống.
  - Mục toàn cục chỉ ghi log vì chặn chúng nghĩa là abstain toàn bộ. Ingest có bất biến riêng (D11), format do code
    `answer/` giữ.
  - Suy hoàn toàn từ rulebook train và rulebook của split lúc chạy, không nhìn rulebook public.
- **Hệ quả:**
  - Đổi chữ vô hại (sửa chính tả) trong một dòng định nghĩa sẽ làm abstain thêm, không làm sai.
  - Dòng mới định nghĩa lại bằng từ đã có trong train (không có cụm từ mới) thì không bắt được: ghi số dòng như vậy vào log.
  - Đổi rulebook train (khi BTC cập nhật) phải chạy lại `analysis/build_rulebook_index.py`; test kiểm chỉ mục khớp.
  - Sau này các câu bị chặn là ứng viên cho nhánh LLM/code-gen có rulebook trong context.

### D17 — Pin vLLM 0.10.1.1 (base Docker Guide) và numpy 2.2.6 (2026-11-07)
- **Bối cảnh:**
  - D16 yêu cầu bản vLLM của máy GPU thành viên khớp base image `vllm/vllm-openai:v0.10.1.1`. Runbook cũ để
    `pip install vllm` lấy bản mới nhất, hiện là 0.33.0. Image dựng từ bản đó có thể cần CUDA mới hơn driver máy chấm,
    trong khi BTC chỉ cam kết CUDA ≥ 12.8.
  - vllm 0.10.1.1 pin `torch==2.7.1` (cu128) và `numba==0.61.2`. numba bản này cần `numpy<2.3`, còn
    `docker/requirements.txt` đang pin `numpy==2.4.6`: xung đột thật, cả trong image lẫn trên máy thành viên.
- **Lựa chọn:**
  - Extra `llm` pin `vllm==0.10.1.1`. Runbook cấm nâng cấp vLLM, torch, numpy.
  - Hạ `numpy` về 2.2.6 cho cả `.venv` dev, `docker/requirements.txt` và image rule-only.
- **Lý do:**
  - Base được BTC thử trên 5090 giảm rủi ro phần cứng. Replay ghi trên đúng bản vLLM của image thì calibrate mới có
    nghĩa (D14).
  - Đo trên CPU: numpy 2.2.6 cho submission dev 2447/2447 và holdout 553/553 giống hệt từng byte, 155 test pass,
    few-shot sha256 không đổi (`072b2861…`). pandas 3.1.0 nhận numpy ≥ 2.0.2.
- **Hệ quả:**
  - API ép JSON của bản này là `GuidedDecodingParams` (code đã có nhánh fallback). Smoke test ghi lại API đang dùng.
  - Kernel FP8 cho 14B trên 5090 (sm120) ở 0.10.1.1 chưa được kiểm. Nếu lỗi thì dùng TP=2 BF16 (máy chấm có 2 card)
    hoặc AWQ.
  - Image vLLM dùng Python 3.12, `.venv` dev dùng 3.11. Phải so container với local như đã làm với bản rule-only.

### D16 — Public là thước đo khả thi; đọc rulebook lúc chạy lên đầu; giữ GPU cho vòng private (2026-11-07)
- **Bối cảnh:**
  - Từ 04/11 chưa nộp public lần nào (D11: tối đa 1 lần mỗi milestone). Proxy synth đạt 0.985 nhưng public chỉ 0.076: proxy chỉ biến đổi shop dev, không có ngành mới hay luật mới, nên không đo được rủi ro chính.
  - Đối chiếu portal ngày 07/11:
    - Rule 6 cấm *"collect answers to test questions by any means"*.
    - Mục Data yêu cầu agent dùng rulebook của đúng split đang trả lời, không dùng bản copy từ train.
    - Vòng private: **đội tự chạy** bản đã đóng băng (20/11 09:00 → 23/11), điểm cuối là điểm BTC chạy lại.
    - Image nộp bằng link Google Drive public kèm checksum.
    - Rule 5: agent được kỳ vọng sinh và chạy code.
- **Lựa chọn:**
  1. **Public là thước đo khả thi chính, cùng với proxy.** Mỗi milestone nộp đúng **phiên bản hệ thống hoàn chỉnh**, tối đa khoảng 2 lượt mỗi ngày, chọn loại model "<=15B", ghi vào `EXPERIMENTS.md`. Thay phần giới hạn tần suất của D11. Phần còn lại của D11 giữ nguyên: không sửa theo từng shop, không đọc câu hỏi hay giá trị ô.
  2. **Không thiết kế bài nộp để dò nhãn:** không nộp hai bản chỉ khác nhau ở một nhóm câu nhỏ (ví dụ tắt bất biến trên 2 shop), không chọn op hay shop theo điểm public. Theo Rule 6, những việc đó gần với thu thập đáp án.
  3. **Đọc rulebook lúc chạy đứng ngang LLM, làm trước phần calibrate LLM.** Tham số luật (ca, đầu tuần, Tết, định nghĩa) đọc từ `/data/rulebook.md`, mặc định lấy từ train khi không đọc được. Kiểm thử bằng rulebook tổng hợp tự đổi tham số trên shop dev. **Không xem rulebook public để định hướng code** (SPEC §7.1).
  4. **Mục tiêu precision public ước tính ≥ 90%** trước khi mở rộng độ phủ. Bản nộp nào làm precision giảm thì không nhận.
  5. **Hậu cần vòng private:**
     - Giữ GPU cho 20–23/11.
     - Chạy thử toàn bộ quy trình chạy private trước 16/11.
     - Chuẩn bị link Drive public cho image kèm checksum.
     - README bài nộp có khai báo model.
- **Lý do:**
  - Public cùng format với private và có ngành, luật lạ, nên là tín hiệu tốt nhất hiện có. Portal ghi rõ điểm public để tham khảo trong lúc phát triển.
  - Đọc rulebook là yêu cầu của đề. Private chắc chắn có rulebook riêng, và tham số đặt cứng từ train là nghi phạm hàng đầu của các câu sai trên public.
  - Thiếu GPU trong 20–23/11 thì không có submission private, và không được BTC chạy lại.
- **Hệ quả:**
  - Ngay bây giờ: nộp baseline v0.6 rule-only.
  - Thêm mốc go/no-go 11/11 theo PA public (ngưỡng chốt sau khi có baseline).
  - Ngân sách khoảng 40 giây/câu (500 câu trong 6 giờ) đủ cho agent nhiều bước hoặc code-gen. Code-gen chỉ làm khi đọc rulebook và LLM parser đã xong.
  - Docker Guide khuyến nghị base `vllm/vllm-openai:v0.10.1.1`. Bản vLLM pin từ máy GPU thành viên (D15) phải khớp base này, hoặc chứng minh chạy được trên 5090.

### D15 — Chạy GPU trên máy thành viên, dữ liệu thi đi qua Drive, repo chỉ nhận code và số liệu (2026-11-07)
- **Bối cảnh:** máy lead không có GPU. Một thành viên có GPU NVIDIA. Repo GitHub public. Rà soát cho thấy ngoài
  `data/` còn nhiều đường lộ khác: few-shot (nguyên văn câu train), replay LLM (đầu ra model có mention chép từ câu hỏi
  train/holdout/public), submission và log public, ghi chú QA tính tay. `tests/` đã có 5 câu chép nguyên văn từ train.
  Portal Zalo (public submission) bắt chọn loại model: API bên thứ ba, open-source > 15B, small ≤ 15B.
- **Lựa chọn:**
  - Thành viên chạy theo `docs/GPU_RUNBOOK.md` bằng một lệnh (`scripts/gpu/run_all.sh`): smoke test
    (`eval/gpu_smoke.py`, câu tự viết) rồi dev, holdout, public cho Qwen3-8B và 14B, ghi replay. Chế độ ghi
    `DATAMATE_LLM_CALL_ALL=1` gọi model trên mọi câu (`config.LLMConfig.call_all`), quyết định ensemble không đổi.
    Lead calibrate và chọn model trên CPU từ replay (D14). Tham số GPU qua env (`DATAMATE_LLM_{DTYPE,TP,GPU_UTIL,MAX_LEN}`),
    không nằm trong khoá replay.
  - Kênh trả kết quả: bundle (replay, submission, log) lên **Drive nhóm**. GitHub chỉ nhận `summary.md` (phiên bản,
    thời gian, VRAM, bảng PA) qua PR vào `docs/gpu_runs/`.
  - Chặn nhiều lớp: `.gitignore` bỏ mọi `*.jsonl`, `*.tgz`; pre-commit và CI chặn `data/`, archive, mọi `.jsonl`;
    hook `eval/check_leak.py` so mọi file text với toàn bộ câu hỏi thi (chỉ in tên file và số lượng, không in câu;
    không có data thì bỏ qua). Viết lại 5 câu test trùng train, test vẫn pass.
  - Mọi lần nộp public chọn **"Based on small models (<=15B)"**, kể cả bản rule-only (không dùng model nào), vì chỉ
    hạng mục SLM được xếp hạng (D02).
- **Lý do:**
  - Thành viên không cần hiểu hệ thống: một lệnh, đầu ra cố định, lỗi chỉ cần gửi traceback.
  - GPU dùng một lần cho mỗi model và prompt. Mọi vòng lặp sau đó chạy trên CPU.
  - Ghi trên mọi câu thì một lần chạy đủ để thử cả `rescue_gate`, `require_agreement` và `llm_only`.
  - Kiểm tra tự động bắt được cả các đường lộ không phải file data (câu chép trong test, docs).
- **Hệ quả:**
  - Few-shot phải giống hệt giữa hai máy: runbook kiểm sha256. Đổi few-shot hoặc prompt thì phải chạy GPU lại.
  - Máy chấm là 2× RTX 5090 32 GB (SPEC): image cuối chạy 14B BF16 với TP=2. Replay ghi bằng TP=1 hoặc bản AWQ/FP8 có
    thể lệch vài token so với image, nên calibrate ensemble theo thống kê, không kỳ vọng image khớp replay từng câu.
    Trước khi nộp phải đo lại PA dev bằng chính image đó.
  - Backend vLLM nuốt lỗi để không làm sập run, nên `LLMParser` đếm `llm_calls/llm_failed`. Summary GPU in số này và
    cảnh báo khi > 0.
  - Lịch sử git vẫn còn 5 câu train cũ trong test. Không rewrite history (câu ngắn, chung chung, không kèm đáp án).
    Nếu BTC yêu cầu thì xử lý sau.
  - Ghi chú QA tính tay (`docs/qa/`, WORKFLOW §3) không được chép câu hỏi hay đáp án: chỉ ghi id và luật áp dụng.
    File paraphrase của TV3 phải là câu tự viết và qua `check_leak`, hoặc để trên Drive.
  - Proxy `eval/synth` dùng lại câu dev nên replay dev dùng được cho synth trên CPU. Cần thêm cờ `--llm` vào
    `eval/synth/run.py` (việc CPU của lead).
  - 14B BF16 (~29.5 GB weights) **không vừa một RTX 5090 32 GB** với `gpu_memory_utilization=0.85`. Image cuối dùng
    `tensor_parallel_size=2` trên 2 card của BTC, hoặc bản FP8. Sửa lại ARCHITECTURE §6 khi chốt model.

### D14 — LLM parser: model chỉ chọn op + mention, ensemble mặc định chỉ "cứu" câu bị gate chặn (2026-11-07)
- **Bối cảnh:** chưa có GPU (máy dev không có NVIDIA, chưa tải model). Phase LLM (PLAN 08–10/11) cần phần lớn code chạy và test được trên CPU, để khi có GPU chỉ còn chạy model và calibrate. Public cho thấy gate từ lạ (D10) bỏ ~368 câu gần như không phân biệt đúng/sai.
- **Lựa chọn:**
  - Contract (`parse/llm_schema.py`): model trả JSON `{op, filters, args}` theo JSON schema (guided decoding). `op` thuộc đúng tập handler của executor, thêm `none` = không op nào khớp. **Model không ghi thời gian**: kỳ lấy từ `timeparse` giống rule parser. Filter thực thể (`product, customer, category, ingredient, staff`) là mention chép từ câu hỏi, link bằng chính `Dictionary` của rule parser; không link được đúng 1 mã thì thành `?…` và executor abstain. Arg ngoài contract của op bị bỏ qua.
  - So khớp bằng `signature()` (op, kiểu đáp án, kỳ, filter đã link, arg của contract, điền mặc định của executor).
  - Backend thay được (`parse/llm.py`): `VLLMBackend` (offline, guided JSON, temp 0 khi n=1, seed cố định, prefix caching), `ReplayBackend` ghi/đọc jsonl theo hash(model, prompt, n, temp, seed). Ghi output trên máy GPU một lần, sau đó calibrate chính sách ensemble trên CPU.
  - Ensemble (`parse/ensemble.py`, cờ trong `config.LLMConfig`). Mặc định: (a) rule qua gate → trả lời như cũ, không gọi model; (b) rule bị gate chặn → trả lời khi **mọi** mẫu LLM cùng signature với rule; (c) rule không parse được → abstain (`llm_only` tắt). `require_agreement` (abstain khi rule qua gate mà LLM khác) và `llm_only` (k/k phiếu) để bật sau khi đo.
  - Few-shot chỉ lấy từ câu dev mà pipeline trả lời đúng (`analysis/build_fewshot.py`), 2 câu/op, ưu tiên khác shop. Prompt = hướng dẫn + bảng op + few-shot (tĩnh) + câu hỏi gốc có dấu.
  - Log: `thinking` = list output thô của model (D12), `inputs.source` = rule | rule+llm | llm.
- **Lý do:**
  - Giữ D04: model không ra số, không ra ngày, không ra mã. Sai của model chỉ có thể là chọn sai op hoặc arg, và đều bị kiểm bằng rule parser.
  - Mặc định (b) không làm mất câu nào đang đúng, chỉ có thể thêm câu. Rủi ro duy nhất là rule và LLM cùng sai giống nhau trên câu lạ.
  - Replay giúp tách tiền GPU khỏi vòng lặp calibrate.
- **Hệ quả:**
  - Backend `off` (mặc định) cho kết quả y hệt bản rule-only.
  - `llm_only` cần nhiều mẫu (n_samples ≥ 3, temp > 0), tốn thời gian hơn. Đo trên holdout/proxy trước khi bật.
  - Chưa chốt model (việc riêng, cần GPU): ứng viên dòng Qwen3 8B/14B (ARCHITECTURE §6).
  - `parse/fewshot.jsonl` chứa nguyên văn câu hỏi train nên bị gitignore (D13). Mỗi máy và mỗi lần build image phải sinh lại bằng `analysis/build_fewshot.py`. Thiếu file thì prompt không có ví dụ.

### D13 — Quy trình nhóm: pre-commit + GitHub Actions CI + ruleset cho main (2026-11-06)
- **Bối cảnh:** sắp có thêm 2 thành viên clone repo và mở PR. Image nộp bài build từ `main`, nên `main` hỏng là mất lượt nộp. Repo để public. Dữ liệu thi không được đưa lên GitHub.
- **Lựa chọn:**
  - `ruff==0.14.4` (lint, luật mặc định E4/E7/E9/F, bỏ E741) và `pre-commit` thêm vào extra `dev`. Không bật formatter để tránh diff lớn trên code đang chạy ổn.
  - `pre-commit`: ruff, chặn file > 1MB, conflict marker, private key, và hook chặn commit `data/`, `*.zip`, `questions/answers.jsonl`.
  - CI (GitHub Actions) chạy trên mọi PR: lint, pytest (test cần data tự skip), kiểm repo không có data/token/file lớn, build image Docker với thư mục data rỗng rồi smoke test `--network none`.
  - PR template có bảng kết quả đo và checklist (gồm xác nhận không đọc public/holdout). CODEOWNERS = lead. Dependabot chỉ cập nhật GitHub Actions.
  - `main` bảo vệ bằng ruleset: bắt buộc PR, 1 approval từ code owner, CI xanh, không force-push, squash merge.
  - **Cập nhật 2026-11-06:** bỏ yêu cầu approval (0) và bỏ yêu cầu review từ code owner. Nhóm 3 người, tác giả PR không tự approve được, nên mỗi PR của lead phải chờ thành viên khác, quá chậm trong 10 ngày cuối. Ruleset giờ chỉ bắt buộc PR, 4 check CI xanh, squash merge, cấm force-push và xoá nhánh. Review là quy ước nhóm: ít nhất 1 người khác xem PR, có ô xác nhận trong PR template. CODEOWNERS giữ lại để lead được tự động gán làm reviewer.
- **Lý do:** đây là bộ tối thiểu mà nhóm phần mềm thường dùng (GitHub flow + CI gate + review bắt buộc). Chi phí thấp, chặn được các lỗi nguy hiểm nhất ở đây: commit nhầm dữ liệu thi, làm hỏng `main` hay image.
- **Hệ quả:**
  - CI không đo được độ chính xác, nên người mở PR phải tự chạy eval dev và dán số vào PR.
  - Không làm CD đẩy image lên registry, vì image chứa dữ liệu train.
  - Dọn 11 import thừa và 1 biến thừa: dev không đổi (2410/0/37).

### D12 — Docker: cache ở /tmp, log từng câu cạnh submission, image rule-only trước (2026-11-05)
- **Bối cảnh:** Docker Guide (SPEC §7 câu 2–3): `/data` read-only, file tạm phải ở `/tmp`, BTC chạy `bash /code/predict.sh` không tham số. Mỗi câu phải log `id, answer, elapsed_ms, thinking` (thinking là reasoning gốc). Trước đây cache raw/Store ghi vào `<repo>/cache`.
- **Lựa chọn:**
  - `config.cache_root()` đọc `DATAMATE_CACHE`, mặc định `<repo>/cache` cho dev. `predict.sh` đặt `/tmp/datamate_cache`. Ghi cache lỗi thì bỏ qua, không crash.
  - `run.py` ghi `/result/logs/log.jsonl` (cờ `--log`, mặc định `<thư mục submission>/logs/log.jsonl`; Guide §3.3), flush từng câu. `answer` là dòng submission bỏ `id`. `thinking` = `null` vì pipeline luật không có reasoning của model (contract: không có reasoning thì ghi null). Vết xử lý nằm ở `code` (hàm executor), `inputs` (câu chuẩn hoá, Intent, note gate/invariant/abstain) và `output` (`pipeline.trace_of`). Khi có LLM, `thinking` là reasoning gốc, gọi nhiều lần thì là list. Lỗi thì ghi traceback vào `error`.
  - Image rule-only `python:3.11-slim`, deps pin đúng bản `.venv` (`docker/requirements.txt`), kèm code. `/code/training_code/` chứa code train, train data, docs và `data_manifest.json` (SHA-256 từng file, `docker/make_manifest.py`).
- **Lý do:** có image dự phòng chạy được sớm (mốc 12/11), độc lập với phần LLM. Pin phiên bản để lần chạy lại của BTC lệch ≤ 0.03.
- **Hệ quả:**
  - Đường dẫn log đã chốt theo Guide §3.3 (`/result/logs/`). Nếu cần đổi thì chỉ sửa cờ `--log` trong `predict.sh`.
  - `elapsed_ms` của câu đầu mỗi shop gồm cả thời gian ingest.
  - Image LLM sẽ đổi base sang CUDA ≥ 12.8 + vLLM, giữ nguyên layout `/code`.

### D11 — Giới hạn dùng public; ingest tự kiểm tra, không chạy theo lỗi từng shop public (2026-11-05) — mục 3 (tần suất nộp) Superseded by D16
- **Bối cảnh:** sau public v0.4 vẫn còn 107 câu abstain do không nhận ra bảng đơn (2 shop). Proxy synth đã sạch (6 transform đều 24/24) nhưng không tái hiện được kiểu dữ liệu của 2 shop này. BTC xác nhận được xem cấu trúc dữ liệu public để cải thiện bộ đọc (`SPEC.md` §7 câu 4). Rủi ro còn lại là overfit: sửa cho đúng 14 shop public trong khi private là shop mới. Các câu abstain đó vẫn an toàn (được 0 điểm). Mối nguy thật là câu ingest đọc sai mà vẫn trả lời (bị −1).
- **Lựa chọn:**
  1. Không tối ưu theo từng lỗi public. Được xem cấu trúc public (tên file, sheet, header, README), nhưng chỉ để rút ra **kiểu dữ liệu**, không phải bản sửa cho một shop. Không đọc giá trị ô hay câu hỏi để viết luật (luật 1 vẫn giữ). Ghi lại mỗi lần xem: shop nào, xem gì.
  2. Mỗi kiểu mới phải có một transform `eval/synth` tái hiện trên shop dev. Bản sửa chỉ được nhận khi đo trên proxy, `fp_snapshot` shop gốc không đổi và holdout không giảm. Không nhận bản sửa chỉ làm tăng điểm public.
  3. Điểm public chỉ để quan sát: tối đa một lần nộp mỗi milestone, không sửa code theo con số đó.
  4. Ưu tiên tự kiểm tra trước khi mở rộng phủ: kiểm tra bất biến lúc chạy trên Store. Shop nào vi phạm thì abstain cả shop, hoặc abstain những câu dùng đến phần dữ liệu bị nghi.
- **Lý do:** private là shop mới. Bộ đọc tổng quát, biết tự nghi ngờ thì đáng giá hơn bộ đọc thuộc lòng 14 shop public. PA phạt câu sai nặng hơn nhiều so với câu bỏ qua.
- **Hệ quả:** 107 câu của 2 shop đó tạm vẫn abstain. Việc tiếp theo là bất biến lúc chạy, đo bằng transform cố ý làm hỏng ingest: bất biến phải chuyển câu sai thành abstain mà không làm mất câu đúng trên shop gốc.

### D10 — Chốt chặn phủ từ (coverage gate) trước khi trả lời (2026-11-04)
- **Bối cảnh:** lần nộp public đầu được PA 0.066 (ước ~372 đúng / ~306 sai, precision ~55%), trong khi holdout là 0.872. Public có ngành mới (mẹ & bé, sửa xe, điện thoại) và ~10 định nghĩa mới trong rulebook. Parser luật gán câu lạ vào op gần nhất rồi trả lời tự tin nhưng sai.
- **Lựa chọn:** `parse/gate.py` che thực thể catalog, mốc thời gian và số. Nếu còn từ nào ngoài `parse/gate_vocab.txt` thì abstain. Từ vựng gồm từ của các câu dev đã trả lời đúng, mỗi từ phải xuất hiện ở ≥ 2 shop (`analysis/build_gate_vocab.py --min-shops 2`, 206 từ, chỉ dùng dev).
- **Lý do:**
  - Leave-one-industry-out trên dev: gặp ngành chưa thấy thì chốt chặn bỏ ~11% câu đúng (coverage 0.889).
  - Holdout: số câu sai 20 → 10, precision 96.2% → 98.0%, PA 0.872 → 0.859.
  - Đánh đổi có lợi khi precision thấp; public và private là ngành lạ.
- **Hệ quả:**
  - Tỉ lệ abstain trên public 32% → 69%.
  - Thêm cách diễn đạt mới thì phải build lại từ vựng.
  - Khi có LLM parser: câu bị chốt chặn chặn sẽ chuyển sang LLM thay vì abstain.

### D09 — Ingest 3 tầng gán vai trò + các luật làm sạch (2026-11-04)
- **Bối cảnh:**
  - 4/30 shop không mô tả cột. biz_124 đặt tên cột cố tình lạ (`Order ID`, `Slg`, `Đ.vị`, `Bằng`, `Hạng`).
  - Sổ tay có dòng tổng phụ. Tiền ghi dạng `3,185tr`, `1tr250`.
- **Lựa chọn:**
  - Gán vai trò theo thứ tự: (1) mô tả README → vai trò; (2) tên cột → vai trò (từ điển tay + `learned_names.json` học từ README dev); (3) suy luận từ từ khoá tên cột + profile giá trị (mẫu mã `ORD/SP/KH/PN`, ngày, số).
  - Loại bảng xác định từ tập vai trò kết hợp gợi ý tên file. Vai trò chung (`name`, `qty`, `price`…) được cụ thể hoá theo loại bảng.
- **Luật làm sạch:**
  - Bỏ dòng trùng theo mã đơn **sau khi** điền xuôi.
  - Bỏ dòng tổng phụ (`TỔNG NGÀY…`).
  - Trạng thái ưu tiên huỷ > trả hàng (mọi bằng chứng) > đang giao > hoàn thành.
  - Shop chỉ có hoá đơn thì lấy tiền hàng từ cấp đơn.
- **Lý do:** đã đo được doanh thu tháng khớp tuyệt đối 68/68 dev, 8/8 holdout.
- **Hệ quả:** tầng (3) là heuristic. Dự định dùng LLM làm schema mapper dự phòng cho shop lạ ở private.

### D08 — Đọc Excel bằng calamine, fallback openpyxl (2026-11-04)
- **Bối cảnh:** đọc toàn bộ 30 shop train bằng openpyxl mất 587 giây, trong khi private có ngân sách 6 giờ gồm cả LLM.
- **Lựa chọn:** dùng `python-calamine` (Rust) qua `pd.read_excel(engine="calamine")`. Lỗi thì fallback sang openpyxl. Cache pickle theo file vào `cache/raw/`.
- **Lý do:** nhanh hơn 8,5 lần trên cùng file (2,5s so với 21,5s), giá trị giống hệt.
- **Hệ quả:** thêm dependency `python-calamine`. Cache có thể ghi được trong container (không ghi vào `/data`).

### D07 — Đọc rulebook lúc chạy, chỉ hardcode luật của train (2026-11-04)
- **Bối cảnh:** rulebook mỗi split khác nhau (public đã thêm luật ngành). Việc hardcode luật chỉ có trong rulebook public là vùng xám (thể lệ 4.2).
- **Lựa chọn:**
  - ✅ Đọc `rulebook.md` trong `/data` lúc chạy, đưa vào prompt hoặc parser. Đây là cách an toàn nhất.
  - ✅ Hardcode các luật nghiệp vụ chung học từ rulebook train.
  - ⚠️ Chưa hardcode luật chỉ xuất hiện trong rulebook public, đợi BTC trả lời.
- **Lý do:** đọc lúc chạy là cách duy nhất vừa an toàn về luật vừa tổng quát được, vì private có thể có luật mới.
- **Hệ quả:**
  - Executor tham số hoá luật theo rulebook đọc được, ví dụ giờ ca, khoảng Tết.
  - Câu hỏi chạm tới luật lạ đi vào nhánh fallback, có rulebook trong context.
  - Cần ghi trên forum câu hỏi cho BTC (`SPEC.md` §7.1).

### D06 — Holdout 6 shop theo kiểu xuất (2026-11-04)
- **Bối cảnh:** private gồm shop chưa từng thấy. Đo trên chính những shop đã dùng để viết code sẽ cho kết quả lạc quan quá mức.
- **Lựa chọn:** giữ lại `biz_103, biz_110, biz_117, biz_123, biz_125, biz_128`, không mở dữ liệu các shop này khi viết code.
- **Lý do:** mỗi shop đại diện một kiểu xuất khác nhau (Sapo-like, CSV hoá đơn, sổ tay VLXD, POS sqlite, CSV phẳng, KiotViet-like).
- **Hệ quả:** PA holdout là chỉ số chính trong `EXPERIMENTS.md`.

### D05 — Map schema theo mô tả README, không theo tên cột (2026-11-04)
- **Bối cảnh:** 244 tên cột khác nhau, nhưng chỉ có khoảng 120 mô tả trong README, theo một từ vựng có kiểm soát.
- **Lựa chọn:** từ điển `mô tả → field chuẩn`, khớp mềm (fuzzy). LLM chỉ dùng khi không khớp.
- **Lý do:** deterministic, tổng quát hoá được sang shop mới miễn là BTC giữ phong cách viết README.
- **Hệ quả:** cần có test với README tổng hợp có mô tả biến thể.

### D04 — LLM dịch, code tính (2026-11-04)
- **Bối cảnh:** chấm exact match tới từng đồng. Model ≤ 15B tính toán không ổn định.
- **Lựa chọn:** LLM chỉ sinh Intent JSON, còn executor deterministic tính kết quả. Có nhánh code-gen dự phòng cho câu OOD, kèm abstain nghiêm ngặt.
- **Lý do:** dễ kiểm thử từng luật, ổn định khi chạy lại (yêu cầu lệch ≤ 0.03).
- **Hệ quả:** coverage bị giới hạn bởi tập intent đã cài, cần nhánh dự phòng cho câu mới.

### D03 — Không dùng public test để phát triển (2026-11-04)
- **Bối cảnh:** thể lệ cấm train hoặc tune trên public/private. Public đã bão hoà (≥ 17 đội đạt 1.0).
- **Lựa chọn:** chỉ đọc train. Public chỉ dùng để nộp kiểm tra định kỳ.
- **Hệ quả:** ❓ việc dùng luật ngành từ rulebook public cần xác minh trên forum (`SPEC.md` §7).

### D02 — Chỉ hạng mục SLM, model open-source ≤ 15B offline (2026-11-04)
- **Lý do:** chỉ hạng mục SLM được xếp hạng. Kết quả dùng API chỉ để tham khảo.
- **Hệ quả:** cần GPU để phát triển phần LLM, vì máy dev không có GPU NVIDIA.

### D01 — Python 3.11, src layout (2026-11-04)
- **Lý do:** tương thích vLLM/CUDA image. `src/datamate` tách package khỏi script.
- **Hệ quả:** chạy bằng `.venv/Scripts/python` với `PYTHONPATH=src` (hoặc `pip install -e .`).
