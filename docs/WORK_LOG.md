# Work log

## 2026-10-20 17:10 +07:00 — Đọc và tiếp nhận quy tắc làm việc

- Mục tiêu: đọc đầy đủ tệp quy tắc người dùng cung cấp và áp dụng cho các bước tiếp theo.
- Nguồn: tệp đính kèm `Pasted text.txt`, attachment ID `eeb71457-552e-4cfa-b6f4-e62e87af65c9`.
- Git đã kiểm tra: branch hiện tại `master`, chưa có commit, chưa cấu hình remote. Quy ước người dùng là `main` và `origin` trỏ tới `https://github.com/lee-vtruong/training-free-fact-verification`. Chưa thay đổi branch hoặc remote trong bước đọc quy tắc này.
- Commands: đọc tệp bằng `Get-Content -Raw -Encoding UTF8`; chạy `git status --short --branch`, `git branch --show-current`, `git rev-parse --verify HEAD`, `git remote -v`, và kiểm tra sự tồn tại của work log.
- Files tạo/sửa trong bước này: `docs/WORK_LOG.md`.
- Tests: không chạy lại vì chỉ thêm nhật ký Markdown. Kết quả bước triển khai trước: 22 tests passed; không coi đó là kết quả thực nghiệm.
- Quyết định: ưu tiên Accuracy/Macro-F1; không cập nhật parameters; ERRORFlow chỉ đọc; knowledge construction phải loại toàn bộ evaluation fold và near-duplicate groups; mọi artifact có provenance; số liệu chưa đo để null/not measured.
- Vận hành: người dùng review/commit/push và pull trên server. Không tự động push/pull, chạy GPU jobs dài, tải checkpoints lớn, thay đổi protocol hoặc chuyển experimental phase.
- Giới hạn: kết nối server chưa thành công. Host trong quy tắc (`stackops@hvtham-server`) và endpoint người dùng đã cung cấp trước đó (`hvtham@10.44.101.230`, alias `nvidia`) đều chưa được xác minh kết nối thành công; không giả định tài nguyên server đã tồn tại.
- Trạng thái milestone Base Error Audit: **INCOMPLETE**. Scaffold đã được kiểm thử bằng dữ liệu synthetic; chưa có research cache hợp lệ để tính Base metrics.
- Bước tiếp theo: chờ nhiệm vụ tiếp theo từ người dùng; khi tiếp tục setup Git/server, đối chiếu branch, remote và endpoint thực tế trước các thao tác tương ứng.

## 2026-10-20 17:28 +07:00 — M1.1 local retrieval implementation và bàn giao lệnh

- Mục tiêu: triển khai R1 BM25 Top-50, R2 frozen dense Top-50, R3 RRF Top-40 và R4 frozen reranker trên cùng R3 Top-40; xuất cùng Top-8 schema; giới hạn ở retrieval.
- Người dùng xác nhận M0 là setup/đọc quy tắc trước đó, không cần một báo cáo M0 riêng. Đã đọc inventory, configuration, validation và nhật ký hiện có trước triển khai.
- Điều chỉnh quy trình theo nhắc nhở trực tiếp: **Codex chỉ code/test local và gửi lệnh; người dùng chạy terminal server rồi gửi outputs. Không tự SSH/server jobs.** Đã lưu quy tắc trong `AGENTS.md` để các lượt sau áp dụng.
- Failure vận hành: trước nhắc nhở đã thử SSH read-only tới endpoint cũ; timeout, không có kết nối hoặc server command thành công. Sau nhắc nhở chỉ kiểm tra session cũ đã kết thúc, không mở kết nối mới. Không coi lỗi kết nối là bằng chứng về server resources.
- Git: branch `master`, chưa có commit (commit=null), chưa có remote. Không stage/commit/push/pull hoặc thay đổi branch. Repo ERRORFlow không bị chỉnh sửa.
- Files tạo: `AGENTS.md`; `src/tffv/retrieval/{__init__,data,ranking,frozen,benchmark,metrics,evaluate,cli,synthetic}.py`; `scripts/m11_resources.py`, `scripts/m11_smoke.py`; `tests/test_retrieval.py`; hai templates `configs/retrieval_m11.template.json`, `configs/retrieval_data_manifest.template.json`; `docs/M11_RETRIEVAL_PROTOCOL.md`, `docs/M11_SERVER_COMMANDS.md`; `reports/M11_RETRIEVAL_REPORT.md`.
- Files sửa: `pyproject.toml` (optional dependencies/CLI), `.env.example`, `README.md`, work log này.
- Triển khai: strict label-free inputs; segmentation chung với char offsets; scores/URLs/provenance; RRF deterministic; Top-8 shortfall rõ ràng; Top-10 sau dedup cho metrics @10; corpus vector cache content-addressed và kiểm tra hash; model chỉ load local, eval/inference mode, parameters frozen; snapshots/revisions được pin; embeddings/reranker load tuần tự.
- Leakage/reproducibility: chặn gold QA/label/justification fields; review attestations bắt buộc; scan verdict patterns, URL flags, exact/near-duplicate claims, source duplicates; giữ official splits; qrels chỉ đọc sau khi evidence cache đã seal. Lexical flags không chứng minh semantic contamination đã được giải quyết.
- Dependencies/hardware local đã kiểm tra: Python 3.13.11; torch 2.11.0; transformers 4.45.2; numpy 2.4.4; pytest 9.1.1; RTX 3050 Laptop 6144 MiB; free disk 5,930,987,520 bytes tại lúc probe. Transformers local chưa tương thích Qwen3; không tự sửa environment hoặc tải model.
- Nguồn kỹ thuật đã đọc: model cards chính thức `Qwen/Qwen3-Embedding-0.6B` và `Qwen/Qwen3-Reranker-0.6B`. Không dùng nội dung label/evidence từ external evaluation để chỉnh prompt.
- Commands local: kiểm tra `git status --short --branch`, `git remote -v`; `nvidia-smi --query-gpu=...`; đọc docs/files; kiểm tra versions/disk; `python -m pytest -q`; `python -m compileall -q src scripts tests`; `python -m tffv.retrieval.cli --help`; `python scripts/m11_smoke.py --output outputs/m11-synthetic-smoke-001` với `PYTHONPATH=src`; `git diff --check` (repo chưa track files).
- Tests: ban đầu 44 passed, sau bổ sung bounded smoke/near-duplicate checks **46 passed**. Lần cuối 46 passed trong 1.54s; compilation thành công. Không thực hiện training hay real-model inference.
- Smoke: 3 claims/64 documents synthetic, non-neural test doubles, xuất `outputs/m11-synthetic-smoke-001/run/{R1,R2,R3,R4}.jsonl` và evaluation comparison. Đây không phải research evidence caches hoặc metrics AVeriTeC. Không sinh relevance labels để giả lập kết quả nghiên cứu.
- Actual research results: Recall@5/8/10, MRR@10, nDCG@10, semantic coverage, latency/VRAM Qwen và good/bad AVeriTeC cases đều **not measured**. Nếu không có valid passage qrels, relevance metrics sẽ null.
- Known limitations: chưa xác minh corpus label-free, raw schema/conversion, checkpoints/revisions, GPU/environment server hoặc real-model adapter behavior. Qrels cần passage-level mapping hợp lệ. Timing là thành phần tái sử dụng; không phải bốn cold deployments độc lập. Corpus/BM25 in-memory, near-duplicate claim scan quadratic; cần resource check trước full benchmark.
- Trạng thái: **M1.1 INCOMPLETE** ở phần benchmark thực nghiệm. Local code và test/hướng dẫn sẵn sàng; không chọn winner hoặc baseline M1.2. Không chạy Qwen3-8B, không tạo Historical Error Memory.
- Next step: người dùng chuyển code lên repo/server theo quy trình của họ; chạy bước 1 trong `docs/M11_SERVER_COMMANDS.md` (metadata-only resources), gửi `resources.json`. Sau đó mới chốt paths/adapter/config, chạy bounded real-model smoke và quyết định full workload.

## 2026-10-20 17:31 +07:00 — Bổ sung push/pull vào mọi lần bàn giao

- Mục tiêu: áp dụng yêu cầu luôn gửi lệnh Windows push lên GitHub và Linux pull về server, để người dùng tự chạy.
- Git kiểm tra: `git status --short --branch`, `git remote -v`; vẫn `master`, chưa có commit (null), chưa có remote.
- Files sửa: `AGENTS.md`, `docs/M11_SERVER_COMMANDS.md`, `docs/WORK_LOG.md`.
- Hướng dẫn bổ sung: setup main/origin lần đầu theo trạng thái đã kiểm tra; test/stage cụ thể/review/commit/push local; pull --ff-only hoặc clone mới trên server; đối chiếu commit. Không force-push hoặc ghi đè history.
- Tests: không chạy lại vì chỉ sửa tài liệu/quy tắc; kết quả code gần nhất là 46 passed.
- Không chạy branch mutation, remote add, commit, push, pull, clone hoặc SSH. Không kiểm tra remote history qua mạng và không giả định GitHub repo đang trống.
- Next step: người dùng chạy block local, sau đó block server tương ứng, rồi resource reconnaissance. M1.1 thực nghiệm vẫn INCOMPLETE.

## 2026-10-20 17:40 +07:00 — Fix pytest temporary/cache permission failures

- Input: user log tại attachment `5be35548-056c-4c37-a311-19ca80b425e8`; kết quả người dùng 27 passed, 19 setup errors và 2 cache warnings.
- Nguyên nhân quan sát: WinError 5 khi pytest truy cập `AppData/Local/Temp/pytest-of-ASUS` và ghi `.pytest_cache`. Không có assertion failure được chứng minh trong log; không suy đoán ACL cụ thể khi chưa kiểm tra ownership của các thư mục đó.
- Git hiện tại kiểm tra read-only: branch `main`, chưa có commit (null), `origin` đã trỏ đúng GitHub repo. Không chạy lại setup branch/remote trong lệnh bàn giao.
- Files tạo/sửa: `scripts/run_tests.py`, `README.md`, `docs/M11_SERVER_COMMANDS.md`, `docs/WORK_LOG.md`.
- Fix: runner tạo một TemporaryDirectory mới mỗi lần, truyền basetemp riêng vào pytest và disable cacheprovider. Cleanup chỉ phạm vi temporary directory của chính lần chạy. Không sửa ACL, không xóa thư mục pytest/cache cũ, không dùng Administrator.
- Commands: đọc attachment UTF-8; đọc AGENTS/docs/Git state; `python scripts/run_tests.py`.
- Validation local: **46 passed in 1.06s**, exit code 0. Cần người dùng chạy lại bằng tài khoản ASUS để xác nhận môi trường của họ.
- Không thay đổi retrieval/scoring protocol, dataset, model hoặc ERRORFlow. Không SSH, commit/push/pull hoặc GPU inference.
- Next step: người dùng chạy runner local; nếu pass, review/stage files, commit/push rồi pull server theo hướng dẫn. M1.1 thực nghiệm vẫn INCOMPLETE.
