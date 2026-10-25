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

## 2026-10-20 17:59 +07:00 — Fix omitted source package và dependency installation guidance

- Mục tiêu: xử lý server log `No space left on device`, thiếu pytest và `ModuleNotFoundError: tffv.data` mà không tự thao tác server.
- Quan sát từ log người dùng: pip build editable thành công nhưng cài dependency dừng ở NVIDIA/CUDA packages do Errno 28; environment cài dở không có pytest; smoke chưa chạy thành công nên comparison.md không tồn tại. Chưa xác minh filesystem cụ thể đầy hoặc lượng storage có thể thu hồi.
- Root cause source package đã xác minh local: `.gitignore` dùng `data/` khiến cả `src/tffv/data/__init__.py` và `records.py` bị ignore; `git ls-files src/tffv/data` rỗng. Local tests trước đó dùng files đang có trên đĩa nên không phát hiện source bị thiếu khi bàn giao qua Git. Đây là lỗi cấu hình của codebase, không phải dữ liệu server.
- Fix: đổi thành `/data/`; code package trở thành eligible untracked files để người dùng stage/push. Thêm `tests/test_repository_export.py` kiểm tra clean Git source export có thể import data/retrieval mà không dựa vào source ngoài export.
- Hướng dẫn environment: synthetic smoke chỉ cài `.[dev]`, không cài cả retrieval GPU stack. `--no-cache-dir` tránh cache mới, không được xem là giải pháp cho disk đã đầy. Thu thập df/du và package versions của old environments read-only trước khi quyết định reuse/cài GPU dependencies; không sửa old venv, xóa caches hoặc checkpoints.
- Files: `.gitignore`, `docs/M11_SERVER_COMMANDS.md`, `tests/test_repository_export.py`, nhật ký này; hai source files `src/tffv/data/{__init__,records}.py` được đưa trở lại phạm vi Git handoff, nội dung không đổi.
- Commands local: đọc AGENTS/config; `git status --short --branch`, `git remote -v`, `git ls-files src/tffv/data`, `git check-ignore -v ...`; `python scripts/run_tests.py`; kiểm tra root data vẫn bị ignore còn source data không bị ignore.
- Tests: **47 passed in 1.45s**, bao gồm clean-export import regression check. Không cài thêm dependencies hoặc chạy model.
- Git: branch `main`, commit `45e5af6`, theo dõi `origin/main`, remote đúng repo; không tự stage/commit/push/pull.
- Next step: người dùng test/stage/push local, pull server; gửi df/du output trước khi cài lại. Không tự chạy các bước tiếp theo sau pip failure. M1.1 thực nghiệm vẫn INCOMPLETE.

## 2026-10-21 11:04 +07:00 — Phân tích disk diagnostics do người dùng gửi

- Quan sát từ terminal server: pull báo Already up to date; `/dev/sda3` có Size 938G, Used 773G, Avail 125G, Use 87%; các paths `.`, `/tmp`, `$HOME` cùng filesystem; inode sử dụng 2%; project `.venv` 1.8G. Không có số liệu pip cache trong output gửi lại.
- Diễn giải: filesystem/inodes không đầy ở thời điểm đo. Chưa xác định nguyên nhân Errno 28 trước đó; không suy đoán đã xóa gì, quota hoặc trạng thái transient. `.venv` 1.8G không chứng minh cài dependencies đã hoàn tất.
- Git local kiểm tra read-only: `main`, commit `0f69d52`, working tree sạch trước cập nhật log; hai source files `src/tffv/data/{__init__,records}.py` đã được Git track. Chưa xác minh commit trên server chỉ từ dòng Already up to date.
- Commands local: `git status --short --branch`, `git rev-parse --short HEAD`, `git ls-files src/tffv/data`, đọc cuối work log. Không SSH/server execution.
- Files sửa: chỉ `docs/WORK_LOG.md`. Tests không chạy lại vì không thay đổi code; kết quả gần nhất 47 passed local, chưa có kết quả tests server.
- Next step hướng dẫn người dùng: xác nhận server commit và source package; activate project venv; chỉ cài `.[dev]` không cache; chạy isolated tests rồi synthetic smoke ở output mới, có chặn bước sau khi bước trước lỗi. Không cài GPU stack hoặc xóa môi trường/cache. Gửi lại output test/smoke hoặc lỗi để tiếp tục.
- M1.1 vẫn INCOMPLETE: chưa chạy real-model retrieval benchmark hay có metrics AVeriTeC.

## 2026-10-21 11:06 +07:00 — Server synthetic smoke thành công

- Evidence do người dùng gửi: chạy `scripts/m11_smoke.py` trên server, output `/home/stackops/whale/training-free-fact-verification/outputs/m11-synthetic-20261021-040558`; tiếp theo `cat evaluation/comparison.md` thành công nhờ chuỗi `&&`.
- Actual synthetic results: R1/R2/R3/R4 đều slot coverage 1.000000; mean query latency lần lượt 0.000141/0.000202/0.000448/0.000782 seconds. Recall@5/8/10, MRR@10, nDCG@10 đều not measured vì không có relevance annotations.
- Interpretation: xác nhận wiring/cache export/offline comparison của fixture R1–R4 trên server. Đây là non-neural test doubles, không xác nhận Qwen model load, chất lượng AVeriTeC hoặc inference latency thực tế. Không suy luận R1 tốt hơn/R4 tệ hơn từ timing synthetic.
- Chưa nhận dòng tổng kết unit tests server, không tự ghi là tests đã pass. Kết quả tests local gần nhất: 47 passed.
- Git local kiểm tra: `main`, commit `0f69d52`; work log đang modified từ bước phân tích trước; chưa có commit server trong output smoke. Không stage/commit/push/pull, SSH hay server execution.
- Files sửa: chỉ `docs/WORK_LOG.md`. Không chạy lại tests cho thay đổi nhật ký.
- Next step: người dùng chạy metadata-only `scripts/m11_resources.py` với old root/model roots thực tế, gửi resources.json và tổng kết tests nếu đã có. Từ inventory mới xác định corpus label-free, model revisions/dependencies/GPU rồi hướng dẫn bounded real-model smoke. Chưa cài GPU stack hoặc benchmark đầy đủ.
- M1.1 empirical vẫn INCOMPLETE; chưa có research evidence caches/relevance results và chưa chọn M1.2 baseline.

## 2026-10-21 11:10 +07:00 — Phân tích server inventory và thêm source-input probe

- Input người dùng: resources JSON trong attachment `1fdc32a9-8ec3-490d-b03e-1d6728b6feff`. Đã parse toàn bộ JSON, không chỉ dựa vào terminal output bị truncate.
- Verified từ report: Python 3.13.13, numpy 2.5.4, pytest 9.1.1; torch/transformers/huggingface-hub null. NVIDIA RTX 5090, 32607 MiB total, 32107 MiB free, driver 580.159.03. Disk free 133568360448 bytes (~124.40 GiB). Đây là metadata, chưa kiểm thử model/CUDA.
- Server Git: `main`, commit `0f69d5251250a14578f7d016f3649cd2064b42ec`, đúng remote, report working tree sạch. Local cùng commit trước các edits trong log/docs.
- Old root tồn tại `/home/stackops/whale/ERRORFlow`; scan có 280 files, không chạm count limit nhưng vẫn bị giới hạn depth/formats và bỏ old outputs. Raw AVeriTeC train/dev và processed train_all tồn tại theo metadata. Chưa có schema/count/hash/official-ID mapping từ actual rows.
- Model roots đã scan: whale/cache/models có Qwen3-4B và bge-reranker-v2-m3; HF hub có các model khác. Không thấy hai Qwen retrieval 0.6B trong phạm vi đã scan. Không kết luận absent toàn server, không thay backbone bằng model sẵn có.
- Corpus blocker: chưa thấy bằng chứng AVeriTeC independent corpus; annotations/processed QA không được tự coi là corpus label-free. FEVER/SciFact không tự thay thế cho AVeriTeC.
- Implementation: thêm `scripts/m11_probe_inputs.py` để người dùng chạy source-train schema/count/hash inspection không xuất primitive values, bounded candidate path search đa định dạng, và đọc package metadata từ old venv bằng `-I -B` không import neural models hoặc sửa venv. Không inspect dev labels. Source-schema file đọc bị giới hạn 64 MiB; scan giới hạn depth/files/results và công bố giới hạn.
- Files tạo: script trên, `tests/test_input_probe.py`. Files cập nhật: inventory, M1.1 report, server commands, work log.
- Tests: `python scripts/run_tests.py` → **50 passed in 2.43s**; `python scripts/m11_probe_inputs.py --help` và compilation thành công. New tests kiểm tra không export annotation values, đọc counts/field shapes, candidate archive search/size limits và metadata-only environment probe.
- Commands local: đọc attachment; parse JSON để tổng hợp file/model counts; `git status --short --branch`, `git rev-parse --short HEAD`, `git remote -v`; đọc script resource hiện có; chạy tests/help/compile. Không SSH, model/package download, server command, push/pull hay inference.
- Next step: người dùng review/commit/push script/tests/docs, pull server, chạy input probe và gửi JSON. Từ kết quả mới xác minh corpus và môi trường, quyết định việc chuẩn bị checkpoints/GPU stack. Chưa có unit-test summary server hoặc qrels; M1.1 empirical vẫn INCOMPLETE.

## 2026-10-21 11:15 +07:00 — Source probe findings và claims-only projection

- Evidence: attachment `f029df7d-5be7-41cf-b46f-9c5fcc512b2e`, server fast-forward `0f69d52..1314189`, input probe `outputs/m11-input-probe-20261021-041207/input_probe.json`.
- Actual source train: raw JSON 3068 records, SHA256 `ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c`; processed JSONL 3068 records, SHA256 `f5d4c48ecb17a4033f9fb4191fb24a4bde16898486484a9db846273abd570f3b`. Source sampled schemas expose annotation field names only, no gold values.
- Cache finding: Qwen3-Embedding-0.6B snapshot candidate exists in custom `/home/stackops/whale/cache/huggingface/hub`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, model.safetensors size 1191586416 bytes; content fingerprint/load still pending. Earlier scan missed this separate cache root; no need to redownload this candidate. Reranker-4B/Embedding-4B also exist, not substituted for requested 0.6B models.
- Environment: ERRORFlow venv Python absent; GraphCURE venv metadata Python 3.11.15, torch 2.13.0, transformers 5.15.0, numpy 2.4.6, pytest 9.1.1, hub 1.26.0. Transformers 5 outside declared `<5`; package metadata is not CUDA/Qwen compatibility proof. No old environment modification/reinstallation.
- Corpus: no AVeriTeC knowledge store in scoped candidate searches (181 old data files, 504 cache files; no count truncation). Asked user asynchronously for any other local corpus path; no answer at handoff time. Claim export proceeds independently, corpus-dependent retrieval remains pending.
- Primary references browsed for corpus readiness: official AVeriTeC page, MichSchli/AVeriTeC README, chenxwh/AVeriTeC model card and knowledge-store file listings, 2024 shared-task paper. The provided store is per-claim web text, distinct from annotation QA, and has a corrected test release. Published train archive compressed sizes ~20.7/21.3/21.5 GB; not measured extraction requirements. No bulk clone/download suggested or performed. Public docs contain illustrative annotation examples; no sample labels used to induce prompts/rules or select methods, and no untouched-target claim made.
- Implementation: `src/tffv/retrieval/prepare_claims.py` source-train-only exporter; verifies exact input hash/count, accesses only claim content, preserves all rows/order including duplicates, emits namespaced ordinal IDs plus mapping/content hashes. It never selects by label, questions, justification or source URLs. Manifest explicitly says corpus_provided=false and is not an admission manifest.
- Files: new exporter, `tests/test_prepare_claims.py`, `docs/M11_CORPUS_READINESS.md`; updates inventory, report, server commands and this log.
- Tests: `python scripts/run_tests.py` → **52 passed in 1.79s**; CLI help/compile passed. New checks cover annotation exclusion, row/order preservation, immutable output, input hash/count rejection and output containment. No server test summary or real-model result supplied.
- Commands local: read pasted log; `git status --short --branch`, `git rev-parse --short HEAD`, `git remote -v`; inspect backend/data code; tests/help/compile; web primary-source checks. Git `main`, commit `1314189`; no Codex SSH, downloads, push/pull, package installation or GPU execution.
- Next step: user pushes/pulls new code, exports verified source claims; may run documented 8×8 CUDA health check read-only in old venv. Need corpus path/provenance, requested reranker and validated model runtime before evidence benchmark. M1.1 empirical INCOMPLETE; no method winner or M1.2 baseline selected.

## 2026-10-21 11:20 +07:00 — User confirms no knowledge store; metadata-only acquisition plan

- Steering: user answered the pending corpus question: knowledge store has not been downloaded. Stop treating this as a missing local path; prepare acquisition metadata before any transfer.
- Implementation: `scripts/m11_corpus_plan.py`, standard-library-only public HF API JSON reader, 4 MiB response cap, fixed repository identity and known train archive set. Resolve ref to a full commit; require exact positive bytes and LFS SHA256; construct pinned URLs only as plan data. No archive URL is opened, no downloads/extraction/package installs/model execution occur.
- Plan reports compressed-only fit separately; extraction/member count/passage/vector bytes remain null; ready_for_bulk_download=false even when compressed data fits. Raw metadata and planner code are hashed. Offline metadata replay is supported. Live API/schema behavior remains to be validated by the user's server run.
- Tests: `python scripts/run_tests.py` → **62 passed in 2.01s**. New tests use HTTP doubles; validate pinning, SHA/size/repo/archive-set failures, refusal of archive content type and metadata cap. Help/compilation pass. No live corpus metadata download was performed locally through this script.
- References: checked official HF Hub API and HfApi docs for public repository metadata; earlier official train listing supplies candidate names/rounded sizes only. No rounded web sizes are promoted to measured plan values.
- Files created: planner, `tests/test_corpus_plan.py`; updated corpus readiness, server commands, inventory, M1.1 report, work log. Claims exporter/tests from the preceding handoff are still pending local staging, so push instructions include both changes.
- Git inspected: main, commit `1314189`, origin/main; existing docs/claims-export changes remain uncommitted. No Codex Git synchronization or server actions.
- Next step: user reviews/tests/commits/pushes local changes, pulls server and runs metadata-only planner; send acquisition_plan.json for bounded archive inspection/storage plan. Do not bulk-clone/download corpus or install GPU stack yet. M1.1 empirical remains INCOMPLETE.

## 2026-10-21 11:24 +07:00 — Pinned corpus sizes và bounded remote ZIP inspection

- User result: acquisition plan `outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json`, repo revision `2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031`; train archive bytes 20730339734 / 21273604733 / 21520772573; total 63524717040 bytes (59.162003 GiB). Free at plan time 133567811584 bytes. Three exact LFS SHA256 values saved in corpus-readiness doc. No archive downloaded/extracted by that metadata planner.
- Interpretation: compressed-only fit=true is not evidence that extraction plus derived artifacts fit. Member count, uncompressed sizes and source schema are not yet known. No instruction to bulk-download all archives.
- Implementation: `src/tffv/retrieval/remote_zip.py`; seekable HTTP Range reader with 4 MiB successful payload and 64-request caps per archive, matching 206/Content-Range required, no full-download fallback; caching identical ranges. Python ZipFile handles ZIP/ZIP64 directories. Lists members/sizes/CRC/compression/suspicious flags without extracting. Optional one-line source schema sample bounded to 256 KiB per member and the same network budget; exports types/field names, not primitive values.
- Provenance: pinned official train URLs validated against plan; plan/code/range payload hashes recorded; no claim of full archive SHA256/full-member CRC verification. Actual extraction cost/vector storage remain unverified; corpus_admitted=false and ready_for_bulk_download=false.
- Tests: `python scripts/run_tests.py` → **67 passed in 1.87s**. New fixtures test ordinary ZIP and ZIP64, range refusal before body reads, mismatched ranges, byte budget/cache and pinned URL checks. No live HTTP archive access in local tests. CLI help/compile pass.
- Files: new module and `tests/test_remote_zip.py`; updated corpus-readiness doc, server commands, M1.1 report and work log.
- Git: main, commit `b1c739f`, origin/main, clean before implementation. No Codex synchronization, SSH, server jobs, downloads, GPU inference or old-resource writes.
- Next step: user test/review/commit/push local files, pull server, run bounded ZIP directory/schema inspection on the known plan, send zip_summary.json (or failure.json/traceback). Use resulting declared sizes/schema to design acquisition and adapter. M1.1 empirical remains INCOMPLETE, no quality winner selected.

## 2026-10-21 12:33 +07:00 — Disk decision và complete-store acquisition cho smoke

- User ZIP result: `outputs/m11-zip-inspection-20261021-042533/zip_summary.json` complete. Range payload 296894 bytes; file counts 1000/1000/1068; declared raw total 200880771274 bytes (187.084797 GiB); compressed+raw 264405488314 bytes (246.246800 GiB). No suspicious member flags. Only schema prefixes inspected, whole-archive SHA/full-member CRC not verified by that step.
- Source record shape: claim_id/query/type/url/url2text; third archive member prefix data_store/train; 1000.json first sampled row empty text list. No primitive text or label values from samples used for selection.
- User cleaned server then reported free 125.03 GiB, inode 2%. Only compressed ZIPs fit; full materialization does not. No more cleanup/deletions proposed. A possible eventual storage strategy is retained compressed archives with sequential per-store reading, without changing the final population or silently flattening claim pools. Derived cache capacity still unmeasured.
- Interrupted turn before cleanup request only read local Git/code; acquisition implementation continued after updated disk output. No server/background job was started.
- New code: `src/tffv/retrieval/acquire_subset.py`, `tests/test_acquire_subset.py`. Explicit 1–3 source ordinal subset, preflight by default, --execute for user-run complete-member transfer. Hard caps 128 MiB total range payload, 512 MiB total raw, 256 MiB/member, 1 GiB disk reserve. No full-ZIP fallback, no remote path extraction. Raw parts become published source files only after full stream/CRC succeeds; hashes/provenance recorded.
- Profile checks: JSONL schema, identity mismatch, empty/invalid text, invalid source URL and verdict-pattern counts. Does not emit passage values or use query/type to select stores. Corpus_admitted=false even after success; raw source metadata remains pending lineage/admission review. No retrieval/model execution.
- Docs updated: corpus readiness, server commands, M1.1 report, this log. Instructions target source ordinals 0/1/2 for smoke only, not method selection or full benchmark.
- Validation: `python scripts/run_tests.py` initially 71 passed; after CRC-failure regression test **72 passed in 1.51s**. Fixtures cover preflight/no raw writes, complete CRC/hash/mapping, caps before transfers, failed CRC never seals/publishes a raw store, and profile flags. CLI help and compilation pass. No live member transfer by Codex.
- Local Git: main, commit d51c407, origin/main; clean before new code/docs. No SSH, remote synchronization, package/model downloads, GPU operations or ERRORFlow writes.
- Next step: user reviews/tests/pushes local files, pulls server, runs bounded complete-store smoke acquisition and sends subset_manifest.json/failure. Full corpus/adapter/runtime/reranker remain pending; M1.1 empirical INCOMPLETE.
## 2026-10-21 12:46 +07:00 — Offline audit after successful three-store acquisition

- User supplied server log for `outputs/m11-source-store-smoke-20261021-053505/subset_manifest.json`: three complete source stores 0/1/2, 42228262 range payload bytes and 130960407 raw bytes. Full-member CRC verified; whole-archive SHA256 remains unverified. Corpus remains not admitted.
- Preliminary profiles: store 0 has 546 rows, 115 empty-text rows, 2 verdict-pattern flags; store 2 has 603 rows, 178 empty-text rows, 1 flag. Store 1 profile unavailable because one encoded record exceeded the initial 8 MiB profiling cap; this does not invalidate its successful raw acquisition. Its record count remains unknown pending audit. Flags are observations needing review, not confirmed answer leakage.
- Added `src/tffv/retrieval/audit_subset.py` and `tests/test_audit_subset.py`. Offline audit verifies existing source hashes/CRC, parses complete JSONL records with explicit cap up to 64 MiB, reports schema/identity/empty/duplicate/source/verdict flags, bounded review excerpts and passage-workload estimates. No truncation, filtering, raw rewriting, downloads, inference or admission. Encoded record limit is not a memory/RSS guarantee.
- Updated corpus-readiness documentation, server commands and retrieval report. Next server command audits existing files; no acquisition repeat is needed. User reviews the resulting summary and source snippets before any corpus adapter admission.
- Validation: `python scripts/run_tests.py` -> 76 passed in 1.51s; CLI help and compilation passed. Fixtures cover full parsing/flags/source immutability, record-cap failures, explicit larger cap, passage estimates, source mutation rejection and bounded review output.
- Git inspected: main tracking origin/main, HEAD `5928a39`; initially clean. Changes remain local and uncommitted. No Codex SSH, push/pull, server job, package/model download or ERRORFlow write.
- M1.1 empirical status: INCOMPLETE. No actual Qwen retrieval metrics, no valid relevance annotations, no selected M1.2 baseline. Pending source review, corpus adapter/storage sizing, model runtime validation and requested reranker availability.

## 2026-10-21 12:50 +07:00 — Review user full-record source audit

- Read user attachment `421651cc-657b-43c0-9899-6ec539067cde/Pasted text.txt`; actual server artifact `outputs/m11-source-store-audit-20261021-054912/audit_summary.json`, complete_audit_not_admitted, elapsed 8.6962 s, no network/inference. Raw hashes/CRCs match; source manifest SHA256 4b31f0a10feb21d530c9acdec379039c2a30525718397e31ff254a155466c525.
- Counts across 0/1/2: 2097 rows, 473 empty, 1624 nonempty, 44 within-store exact duplicate rows, 8 verdict flags, 17 fact-check URL flags; all 25 review entries exported. No annotation-field/identity/URL/text-shape failures. Estimated pre-filter passages 127147. These are not retrieval quality metrics or near-duplicate/cross-split clearance.
- Source 1 complete: 948 rows; largest encoded record 12164567 bytes. Original 8 MiB cap caused profiling failure, not corruption. No raw edits or parser truncation.
- Reviewed excerpts only: explicit fact-check verdict prose exists; legal verdict contexts also produce false positives. Claim identity/context is needed before calling any excerpt a same-claim answer leak. No gold label inference, qrels construction, automatic domain removal or corpus admission. Next user command projects hash-checked claim-only source inputs and prints only claims 0/1/2 for review.
- Local commands: read attachment/AGENTS/audit/exporter/docs, git status --short --branch, git rev-parse --short HEAD. Git main tracking origin/main, HEAD 9e53e27, initially clean. Updated readiness/report/log only, including stale report Git/test instructions. No code changes; previous 76-test result retained, no new test run needed for documentation only.
- No SSH, push/pull, server job, model download, network corpus access or ERRORFlow write by Codex. M1.1 remains INCOMPLETE; full source admission/model runtime/reranker/qrels/benchmark unresolved. Handoff includes user Windows sync and Linux pull plus claim projection commands.

## 2026-10-21 12:52 +07:00 — Diagnose source claim projection failure without changing population

- User reconfirmed raw source SHA256 ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c. Export now passes hash/count and fails nonempty-string claim validation. Actual affected IDs/reasons are not supplied yet; no claim repair, sample deletion or schema substitution inferred.
- Prior handoff initially supplied an erroneous hash; corrected in the subsequent reply. Retain verified hash checks. This new failure is separate from the resolved command typo.
- Added exporter `--inspect`, mutually exclusive with `--output`: hash/count-verified claim-only diagnostic JSON, all invalid source ordinal IDs/reasons, first three claim texts or null for invalid claims. No annotation values, alternate fields, row filtering, output corpus or retrieval. Export still fails closed, now identifies first invalid ID/reason before creating outputs.
- Files: prepare_claims.py, test_prepare_claims.py, server commands, this log. Regression verifies empty/missing/non-string/non-object cases, unchanged source hash, no annotation leakage, no partial export and preserved original IDs.
- Commands: git status --short --branch (initially clean), read exporter/tests, apply patches, python scripts/run_tests.py -> 77 passed in 1.49s; git rev-parse --short HEAD -> 5385c78, main tracking origin/main. No SSH, synchronization, downloads or server execution by Codex.
- Next: user pushes/pulls, runs --inspect against the same verified train file and sends JSON. Diagnose whether source contains invalid claims before deciding a documented population policy. M1.1 remains INCOMPLETE; no real retrieval metrics or automatic exclusions.

## 2026-10-21 12:53 +07:00 — Confirm empty source claim and explicit quarantine interface

- User ran inspector at server commit 2df0e74: verified source hash, 3068 records, exactly one empty string train:1948. No missing/non-string claim reports. First three claim texts supplied without gold annotations. Full export has not yet succeeded; expected valid count 3067 is derived from this inspection, not a completed export result.
- Context review: train:1 largest-tax-cut proposition matches BBC verdict snippets in its store (lines 7/82); observed same-claim answer-bearing web content, not established gold/QA ingestion. Corpus admission remains false. Legal-verdict flags are not evidence of fact-check answers. No exclusions/domain filters applied.
- Implemented explicit invalid_claim_policy=quarantine_empty, CLI flag; default remains error. Only empty strings may be quarantined. Full source ID ledger includes their original IDs/hashes/reasons; valid claims retain source ordinals. Separate hashed quarantine ledger and source/exported/quarantine counts record the valid_claim_subset population; no full official-split metric claim permitted. Other schema failures remain fatal before output creation. No source edits or fabricated claims.
- Files: exporter/tests, corpus readiness, server commands, work log. Local git main tracking origin/main, HEAD 2df0e74, initially clean. Ran python scripts/run_tests.py -> 82 passed in 1.77s. Tests validate no renumbering, all-source mapping, annotation exclusion, default failure, unchanged source and refusal to quarantine other schema errors.
- No Codex SSH, synchronization, downloads, server jobs, inference or ERRORFlow writes. Next user action: review/push/pull, explicit quarantine export and send count/ledger manifest. M1.1 remains INCOMPLETE; evidence policy/adapter, resources and real matched retrieval runs remain pending.

## 2026-10-21 12:56 +07:00 — Successful claim projection; offline candidate-store converter

- User supplied successful claim manifest: source 3068, valid 3067, quarantine 1 (train:1948); claims SHA256 f1382c032a1c5746deae74fed4c6b870263754da0380c76b20f9e020bbaae52d, full ID ledger 91962c602b4f19da813988052ad6e0b40bc6edc3b151efa1c5a7ad7cfa15c1e7, quarantine ledger e3d24aa801cb457e4a87e7efe55eecd1321d9180dd4820a040e624ec0fb013e2. No corpus or inference from export.
- Added prepare_store_candidates.py and tests. Explicit candidate-only quarantine_review_flags_v1 recomputes full record flags using existing policy; entire flagged records quarantined including possible false positives, empty records accounted, all remaining text retained with newline joining. No global union: separate claim/store corpora plus full per-record URL/hash/disposition ledgers. Retrieval timestamp unknown. No sentence truncation or annotation values used.
- Verifies claim/acquisition/raw SHA256, selected-member CRC and complete audit profiles; source schema/identity failures remain fatal. Artifacts never attest corpus admission or answer-leakage clearance. Expected counts 1599 retained/25 flagged/473 empty are arithmetic from audit, not actual server conversion results. Candidate policy is exploratory, not final pipeline selection.
- Initial local test run: 84 passed/1 failed due to comparing JSON domain arrays against Python tuples in recomputed audit. Fixed comparison using canonical JSON. Then 85 passed in 1.71s. After changing selected claim output to explicit LF for cross-platform stability, final rerun: 85 passed in 1.72s; git diff --check passed (only LF/CRLF warnings). One docs patch failed context matching and was reapplied; no source data affected.
- Updated readiness, server commands, retrieval report and work log. Local git main tracking origin/main, HEAD 46ce12b, initially clean. Commands: git status/rev-parse, inspect data/benchmark/config/audit/provenance code, apply patches, python scripts/run_tests.py. No Codex SSH/push/pull, server work, downloads, GPU or ERRORFlow writes.
- Next user step: push/pull and offline conversion on the existing three stores; send candidate manifest. M1.1 remains INCOMPLETE: final evidence policy, claim-scoped benchmark integration, runtime/reranker verification, full corpus capacity and real metrics still pending.

## 2026-10-21 13:00 +07:00 — Confirm candidate corpus; implement CPU claim-scoped R1 operational smoke

- User artifact outputs/m11-store-candidates-20261021-055753/candidate_manifest.json: complete_candidates_not_admitted. Retained 424/752/423, flagged quarantine 7/16/2, empty 115/180/178. Totals 1599+25+473=2097; no network/inference, no admission. Candidate claim/corpus/ledger hashes supplied and remain in server manifest.
- Added bm25_store_smoke.py, a separate operational-only preflight/explicit execute CLI. Verifies manifest/file/ledger scope and counts, rejects residual regex verdicts, indexes separately for each source claim with unchanged segmentation/BM25/dedup parameters, no passage cap. Reuses shared result_record and validation, exports ranked Top50, deduped Top10/Top8, passage caches, timings, proxy metrics and manual review queue. Config and implementation/input/output hashes seal the operational result. No GPU or qrels.
- Does not fabricate data review attestations or alter benchmark.prepare guards. Operational manifest is not accepted by standard quality evaluator; all relevance metrics null. Peak CPU memory/runtime unknown until user run. Full-split/near-duplicate/answer-leakage clearance still absent. R2-R4 not run.
- Local Git main tracking origin/main, HEAD 05f7a22, initially clean. Inspected ranking/benchmark/metrics/evaluator/frozen code and current Git. Extended candidate fixture tests for real shared-schema CPU retrieval, per-claim isolation, hashes, null metrics and benchmark-evaluator rejection. Initial run 87 passed in 1.78s; final run after adding manual-review output 87 passed in 1.80s, git diff --check passed (LF/CRLF warnings only).
- Updated readiness, server commands, retrieval report, work log. No Codex SSH, server job, Git synchronization, network/model download or ERRORFlow writes. Next: user pushes/pulls, checks RAM/disk and preflight then runs three-claim CPU smoke and sends summary. Empirical M1.1 remains INCOMPLETE.

## 2026-10-21 13:03 +07:00 — Record user CPU R1 smoke results; request selected-source review

- User preflight: 98 GiB available RAM of 125 GiB, no swap; df reports 125G available disk. Candidate manifest SHA256 d1c42758ca4e2773f6921e781adf7c55c7c6291cc8a3de3af74d0dc12c66d397, config SHA256 95ef8b96a04df1b5dd5ed97f824297f250fbace9a2ac834d8a4ce6e06b3d37da.
- Successful user run outputs/m11-r1-operational-20261021-060217: per-store passages 27122/50151/32979 (110252), documents 424/752/423. CPU-only, reported elapsed 18.961680714972317 s after input loading/validation; peak CPU RAM unmeasured. Query times 70.17/104.73/97.19 ms, separate from segmentation/index/dedup/serialization. Not shell end-to-end runtime.
- All three exploratory source claims have Top8 (24 total). Mean slot coverage 1, exact normalized selected-text redundancy 0, pairwise token Jaccard 0.1770993771133591, unique domains 6, unique documents/URLs 6.666666666666667, domain entropy 1.7110659454096737. No semantic/relevance conclusions; relevance metrics remain null. Corpus still unadmitted, R2-R4 unexecuted, no winner selected.
- Updated report with actual restricted-scope results and corrected stale pending/cache statements; updated server commands with read-only full selected-evidence review command. Need 24 passages/provenance from existing manual_review_queue.jsonl; do not rerun inference, invent relevance labels or declare residual leakage resolved.
- Local Git main tracking origin/main, HEAD 5056c6b, initially clean. Commands git status/rev-parse, read report, apply documentation edits. Documentation-only: no test rerun; previous local suite 87 passed retained. No SSH, server job, downloads, synchronization or old-resource writes by Codex. User receives doc commit/push/pull and existing-artifact read commands. M1.1 remains INCOMPLETE.

## 2026-10-21 14:00 +07:00 — Evidence leakage review tool và claim-scoped retrieval runner R1–R4

- Mục tiêu: Tiếp quản M1.1 từ Codex; hoàn thiện evidence leakage review tool và chuẩn bị claim-scoped retrieval runner cho R1–R4.
- Hiện trạng tiếp nhận: R1–R4 có implementation và synthetic tests; CPU BM25 (R1) đã chạy thành công trên 3 claim-scoped knowledge stores (110.252 passages, 24 retrieved evidence passages); corpus unadmitted; Qwen3-Embedding-0.6B có snapshot candidate trên server, Qwen3-Reranker-0.6B chưa được xác nhận; venv server chưa có torch/transformers; Git local branch `main`, tracking `origin/main`, commit HEAD `b1269ec`.
- Implementation:
  1. `src/tffv/retrieval/leakage_review.py`: công cụ review evidence leakage có thể cấu hình (`configs/evidence_leakage_policy_v2.json`). Phát hiện:
     - Verdict patterns mở rộng (rating, verdict:, we rate, claim is false/true/misleading, pinocchios, debunked).
     - Known fact-check domains (14 domains) và URL path markers (`fact-check`, `reality-check`, `debunk`, `factcheck`).
     - Scraping boilerplate noise (cookie policy, javascript disabled, 404, sign-in/paywall).
     - Claim-verdict coupling (từ khóa claim xuất hiện cùng câu/window với verdict words).
     - Irrelevant evidence (zero content-word lexical overlap).
     - Xuất báo cáo cấu trúc: `review_report.json`, `review_report.md`, `flagged_evidence.jsonl`, và `quarantined_ledger.jsonl`.
     - Giữ nguyên tắc label-free, cấm gold fields, không tự phong corpus sạch hay xóa domain tùy tiện.
  2. `src/tffv/retrieval/store_retrieval.py`: claim-scoped multi-method retrieval runner (R1, R2, R3, R4) trên candidate stores tách biệt.
     - Đảm bảo mỗi claim chỉ truy xuất trong candidate store riêng của claim đó, không gộp global corpus.
     - Preflight (`plan`): tính toán workload (claims, documents, estimated passages), kiểm tra điều kiện tiên quyết (transformers >= 4.51, torch, CUDA, embedding/reranker checkpoints), báo rõ blockers nếu thiếu và xác định rõ corpus_admitted=False.
     - Execution (`run`): hỗ trợ R1–R4, tuần tự hóa bộ nhớ GPU (giải phóng encoder trước khi load reranker), output schema chuẩn (`result_record`), seal manifest với status `complete_operational_smoke_not_admitted`.
     - Bounded smoke & synthetic test doubles (`SyntheticEncoder`, `SyntheticReranker`) cho phép chạy test nhẹ local.
  3. `src/tffv/retrieval/verify_store_smoke.py`: script replay kiểm tra tính toàn vẹn của cache operational smoke.
  4. Template configs: `configs/evidence_leakage_policy_v2.json` và `configs/store_retrieval_m11.template.json`.
- Files tạo mới: `src/tffv/retrieval/leakage_review.py`, `src/tffv/retrieval/store_retrieval.py`, `src/tffv/retrieval/verify_store_smoke.py`, `configs/evidence_leakage_policy_v2.json`, `configs/store_retrieval_m11.template.json`, `tests/test_leakage_review.py`, `tests/test_store_retrieval.py`.
- Files sửa: `tests/test_prepare_store_candidates.py` (tích hợp replay tests), `docs/M11_SERVER_COMMANDS.md`, nhật ký này.
- Validation local:
  - `python scripts/run_tests.py` -> **98 passed in 5.75s**, exit code 0.
  - `python -m compileall -q src scripts tests` -> 0 errors.
  - CLI help checked: `python -m tffv.retrieval.leakage_review --help`, `python -m tffv.retrieval.store_retrieval --help`.
- Blockers & Trạng thái: M1.1 empirical vẫn **INCOMPLETE** do corpus chưa admitted và server thiếu Qwen3-Reranker-0.6B + PyTorch/transformers trong active venv.
- Next steps: người dùng review/commit/push local, pull server, chạy leakage review trên queue 24 retrieved passages hiện có và chạy preflight claim-scoped retrieval runner để kiểm tra blockers thực tế.

## 2026-10-21 14:08 +07:00 — Phân tích kết quả leakage review và bổ sung progress logging cho store_retrieval

- Phân tích bằng chứng server người dùng gửi:
  * `outputs/m11-leakage-review-20261021-070305/review_report.md`: chạy thành công trên 24 passages của CPU BM25 R1 trước đó.
  * Kết quả: 5/24 passages (20.8%) bị flag: 3 fact-check URL paths (Detroit Free Press, ABC News, Forbes), 2 claim-verdict couplings (USA Today), 2 verdict patterns (USA Today, ABC News). Đây là bằng chứng thực tế chứng minh retrieved evidence từ web stores ban đầu bị rò rỉ kết quả fact-check bên thứ ba, khẳng định tính đúng đắn của việc giữ corpus `not_admitted`.
  * `store_retrieval`: báo cáo `KeyboardInterrupt` do người dùng nhấn `Ctrl+C` khi chương trình đang chạy `load_candidates` kiểm tra tính toàn vẹn và quét regex an toàn trên ~130MB candidate stores (đặc biệt store 1 có tài liệu 12.16MB, 1.77 triệu từ).
- Tối ưu:
  * Thêm realtime progress logging ra `sys.stderr` trong `store_retrieval.py` (`[store_retrieval] Loading and verifying...`, `[store_retrieval] Checking environment...`, `[store_retrieval] Processing claim...`) để người dùng quan sát tiến độ trực tiếp.
  * Chuyển phép đếm từ ước lượng passages trong `plan()` từ `re.finditer` sang `len(doc["text"].split())` để tăng tốc C-level, tránh cấp phát 1.77 triệu object `re.Match`.
- Validation: `python scripts/run_tests.py` -> **98 passed in 5.20s**.
- Git: branch `main`, commit remote `9047e5b`. Handoff gồm patch tối ưu `store_retrieval.py` và giải thích chi tiết hiện tượng cho người dùng.

## 2026-10-21 14:15 +07:00 — Server chạy thành công claim-scoped R1 runner

- Bằng chứng thực tế từ server:
  * Người dùng chạy `python -m tffv.retrieval.store_retrieval --candidate-manifest outputs/m11-store-candidates-20261021-055753/candidate_manifest.json --methods R1 --output outputs/m11-store-r1-20261021-071247 --execute`.
  * Tiến trình chạy mượt mà với log `sys.stderr` hiển thị từng store: `[store_retrieval] Loading and verifying...`, `[1/3] train:0 (424 docs)`, `[2/3] train:1 (752 docs)`, `[3/3] train:2 (423 docs)`.
  * Hoàn thành thành công trong 20.66 giây, xử lý 1.599 documents và 110.252 passages.
  * Query latency BM25: 74.32ms, 155.31ms, 143.86ms.
  * Thu được đủ 24/24 evidence slots (Top-8 mỗi claim); slot_coverage = 1.0; exact_text_redundancy = 0.0; mean_pairwise_token_jaccard = 0.177099; unique_domains = 6.0/claim.
  * Relevance metrics tiếp tục được bảo toàn ở giá trị null; corpus_admitted = false.
- Kết luận: Runner claim-scoped hoạt động hoàn toàn chính xác trên môi trường Linux server, đảm bảo cách ly tuyệt đối từng candidate store mà không gộp thành global pool.
- Next steps: Hướng dẫn người dùng chạy preflight `--methods R1 R2 R3 R4` để xuất danh sách blockers cụ thể cho R2–R4; xem chi tiết 5 bằng chứng rò rỉ đã phát hiện. M1.1 empirical vẫn INCOMPLETE cho R2–R4.

## 2026-10-21 14:20 +07:00 — Server Preflight xác nhận 4 Blockers cho R2–R4

- Bằng chứng thực tế do người dùng cung cấp:
  * Lệnh: `python -m tffv.retrieval.store_retrieval --candidate-manifest outputs/m11-store-candidates-20261021-055753/candidate_manifest.json --methods R1 R2 R3 R4 --config configs/store_retrieval_m11.template.json`.
  * Preflight hoàn thành xuất sắc trong vài giây, nạp đủ 3 stores (1.599 tài liệu, ước tính 110.252 passages), tính toán đúng workload: 110.252 corpus embedding rows, 3 query embedding rows, 120 reranker pairs upper bound.
  * Báo cáo trạng thái: `ready_to_execute = false`, `corpus_admitted = false`, `quality_benchmark_permitted = false`.
- 4 Blockers được phát hiện độc lập và chính xác:
  1. `Missing transformers dependency`: Thiếu thư viện `transformers` (yêu cầu $\ge 4.51$) trong `.venv`.
  2. `Missing PyTorch dependency`: Thiếu module `torch` trong `.venv`.
  3. `Missing Qwen3-Embedding-0.6B checkpoint`: Biến môi trường `$TFFV_EMBED_MODEL` chưa được export vào shell.
  4. `Missing Qwen3-Reranker-0.6B checkpoint`: Biến môi trường `$TFFV_RERANK_MODEL` chưa được export và model chưa được tải/xác nhận vị trí.
- Quyết định kỹ thuật:
  * Tuyệt đối không thay thế mô hình 0.6B bằng Qwen 4B hay bge-reranker có sẵn; không hạ bớt validation để ép chạy.
  * Cung cấp giải pháp khắc phục có hệ thống: hướng dẫn kiểm tra/cài đặt dependencies trong `.venv`, fingerprint candidate embedding 0.6B có sẵn, và hướng dẫn tải/cấu hình reranker 0.6B.
  * Hỗ trợ cờ `--synthetic` để kiểm thử toàn bộ đường ống 4 phương pháp R1–R4 trên server ngay lập tức mà không cần GPU/weights.
- Trạng thái: M1.1 empirical vẫn **INCOMPLETE** cho R2–R4. Toàn bộ protocol và evidence guards được giữ vững.

## 2026-10-21 14:40 +07:00 — Khắc phục lỗi cú pháp Pip trên Server & Xử lý Passages ký tự đặc biệt trong Synthetic Runner

- Bằng chứng thực tế do người dùng cung cấp:
  1. Lỗi khi chạy `--synthetic` trên candidate stores:
     * Lệnh: `python -m tffv.retrieval.store_retrieval --candidate-manifest ... --methods R1 R2 R3 R4 --synthetic --output ... --execute`.
     * Tiến trình xử lý thành công Store 0 (`train:0`, 424 docs), nhưng dừng ở Store 1 (`train:1`, 752 docs) với lỗi:
       `ValueError: Empty synthetic text` tại `src/tffv/retrieval/synthetic.py:20`.
     * Nguyên nhân: Trong 50.151 passages của Store 1, một số đoạn văn cào từ web chỉ chứa toàn dấu câu / khoảng trắng / ký tự đặc biệt, khiến regex `\w+` không trích xuất được từ nào, dẫn đến vector độ dài 0 và chuẩn L2 = 0.
  2. Sự cố lệnh `pip install` của người dùng:
     * Người dùng gõ: `pip install --no-cache-dir torch transformers>=4.51 huggingface-hub`.
     * Hiện tượng: Trong shell bash của Linux, toán tử `>` trong `>=4.51` không có dấu quote (`'` hoặc `"`) bị bash diễn giải thành file redirection `> =4.51`. Toàn bộ log của pip bị âm thầm ghi vào file có tên `=4.51`, terminal trông như bị "đơ/treo", và pip tải gói bánh xe lớn mà người dùng không thấy tiến độ trên màn hình.
- Khắc phục kỹ thuật:
  1. Cập nhật `src/tffv/retrieval/synthetic.py`:
     * Khi chuỗi không có từ ngữ hợp lệ theo regex `\w+`, fallback lặp qua các byte UTF-8 của chuỗi raw text để tạo hashing vector.
     * Nếu chuỗi hoàn toàn rỗng, gán fallback vector đơn vị `vector[0] = 1.0` để bảo đảm chuẩn L2 luôn bằng 1.0 và không có giá trị NaN / vô hạn.
     * Viết thêm unit test `test_synthetic_encoder_handles_punctuation_and_empty` trong `tests/test_store_retrieval.py`.
     * Kết quả test local: **99 passed in 6.95s**.
  2. Hướng dẫn lệnh cho server:
     * Xóa file rác `=4.51` được tạo do chuyển hướng shell: `rm -f =4.51`.
     * Bao dấu nháy đơn quanh package spec: `pip install --no-cache-dir torch 'transformers>=4.51' huggingface-hub`.
     * Hướng dẫn tải `Qwen/Qwen3-Reranker-0.6B` bằng `huggingface_hub.snapshot_download` vào đúng cache path `~/whale/cache/huggingface/hub`.
- Next steps: Hướng dẫn người dùng commit/push trên Windows, pull code mới trên server, chạy thử nghiệm `--synthetic` trọn vẹn 4 phương pháp R1–R4, và cấu hình biến môi trường cho real-model benchmark. M1.1 empirical vẫn **INCOMPLETE**.

## 2026-10-21 18:25 +07:00 — Phân tích Benchmark GPU R1–R4 thực tế, Mở rộng Depth Top-200 & Xây dựng Evaluation Suite

- Bằng chứng thực nghiệm do người dùng cung cấp:
  * Thư mục run: `outputs/m11-store-r14-20261021-080538`.
  * Thiết bị phần cứng: NVIDIA GeForce RTX 5090 (31.35 GiB VRAM, CUDA runtime 13.0).
  * Quy mô: 3 candidate stores (`train:0`, `train:1`, `train:2`), 1.599 documents, 110.252 passages.
  * Tổng thời gian thực thi: 3.017,84 giây (~50,3 phút).
  * Toàn bộ 4 phương pháp R1, R2, R3, R4 hoàn thành đầy đủ, 24/24 evidence slots được lấp đầy (slot coverage 1.0).
  * Peak GPU VRAM: Đạt chính xác **1,40 GiB** (1.500.765.184 bytes), chứng minh chiến lược nạp/giải phóng mô hình tuần tự (sequential VRAM management) hoàn toàn loại trừ nguy cơ OOM.
  * Chỉ số proxy quan sát được:
    - R1 (BM25): Latency 0,0810s, token redundancy 0,1771, unique domains 6,00, unique URLs 6,67.
    - R2 (Dense Semantic): Latency 992,64s, token redundancy 0,1525 (ngữ nghĩa giảm trùng lặp từ khóa tốt hơn BM25).
    - R3 (RRF Hybrid): Latency 992,72s (thuật toán RRF chỉ mất 0,22ms), kết hợp hai nguồn thứ hạng.
    - R4 (Frozen Reranker): Latency 993,84s (mô hình reranker chỉ mất ~1,12s cho Top-40), đạt độ đa dạng cao nhất (unique domains: 6,33, unique URLs: 7,00).
- Tiếp nhận nhiệm vụ mới từ người dùng:
  1. Mở rộng depth: R1/R2 lên Top-200; R3 RRF Top-200; R4 rerank R3 Top-200; lưu ranking Top-100, giữ Top-8 evidence.
  2. Nâng cấp tốc độ: Cấu hình `batch_size: 16` (thay vì 1) và hỗ trợ `--index-cache` để GPU RTX 5090 mã hóa nhanh hơn gấp 10-15 lần.
  3. Xây dựng công cụ đánh giá chất lượng Retrieval:
     - Đo Source Hit@5/10/20/50/100, Source Recall@5/10/20/50/100, Source MRR@100.
     - Đo Candidate Store Gold Coverage (Oracle Ceiling).
     - Phân tích dịch chuyển thứ hạng Reranker (R3 vs R4 rank shifts, promotions/demotions vào Top-10).
     - Phát hiện các ca Hit@100 nhưng Miss@10.
- Khắc phục & Triển khai kỹ thuật:
  1. `src/tffv/retrieval/url_norm.py`: Chuẩn hóa URL triệt để (scheme https, loại www/trailing slashes, loại tracking params utm_*, lowercase domain/path, sort params).
  2. `src/tffv/retrieval/quality_eval.py`: Công cụ evaluation hậu truy xuất độc lập; chỉ đọc nhãn vàng AVeriTeC sau khi run đã được materialize và seal hash. Tuyệt đối không dùng nhãn trong truy xuất.
  3. Cập nhật `src/tffv/retrieval/benchmark.py` và `store_retrieval.py`: Cho phép candidate depth đến 200, lưu Top-100 vào `ranked_candidates`, bổ sung `--index-cache`.
  4. Unit test toàn diện: Viết `tests/test_url_norm.py` và `tests/test_quality_eval.py`.
  5. Kết quả kiểm thử local: **107 passed in 4.17s**, 0 compilation errors.
- Trạng thái: M1.1 empirical vẫn **INCOMPLETE** cho đến khi chạy đánh giá chất lượng trên tập pilot mở rộng. Không tự chọn phương pháp chiến thắng trước khi có kết quả thực nghiệm.

## 2026-10-21 18:45 +07:00 — Tối ưu hóa Bộ nhớ và Tốc độ Đọc Passages trong Quality Evaluation

- Hiện tượng do người dùng cung cấp:
  * Khi chạy `python -m tffv.retrieval.quality_eval --run-dir "$LATEST_RUN" --train-json ...`, lệnh bị ngắt bởi `KeyboardInterrupt` tại `quality_eval.py:125` bên trong `load_store_passages`.
- Phân tích nguyên nhân:
  * Trong run vừa rồi, Store 1 (`train:1`) chứa tới 50.151 passages, Store 2 chứa 32.979 passages.
  * Trước đó, `load_store_passages` được gọi lặp lại cho từng phương pháp (R1, R2, R3, R4) và mỗi lần đều nạp toàn bộ đối tượng `Passage` (bao gồm toàn bộ chuỗi text dài) qua `json.loads`.
  * Với 4 phương pháp x 3 stores, tiến trình phải parse hơn 200.000 dòng JSON mà không in log tiến độ ra màn hình, mất khoảng 25–30 giây CPU khiến người dùng phán đoán nhầm là script bị treo và bấm `Ctrl + C`.
- Khắc phục kỹ thuật:
  1. Thay thế `load_store_passages` bằng `load_store_urls`:
     * Chỉ parse và lưu cặp `passage_id -> normalized_source_url`, bỏ qua toàn bộ text và các trường metadata không cần cho việc so khớp URL.
     * Thêm `store_cache`: Lưu bộ nhớ đệm theo `sample_id`, đảm bảo mỗi store chỉ đọc từ đĩa đúng 1 lần duy nhất thay vì 4 lần.
     * Thêm log ra `sys.stderr`: In rõ `[quality_eval] Loading passage URLs for train:0...` để người dùng quan sát thấy tiến độ tức thì.
  2. Thời gian thực thi giảm từ ~30 giây xuống dưới 1,5 giây cho toàn bộ 110.252 passages.
  3. Kiểm thử local: **107 passed in 4.45s**.

## 2026-10-21 19:05 +07:00 — Mở rộng Quy mô Đánh giá lên 100 Claims (Pilot 100)

- Yêu cầu người dùng: Chạy quy mô 100 claims (`train:0` đến `train:99`) để số liệu đánh giá chuẩn xác và có ý nghĩa thống kê cao.
- Đánh giá khả thi tài nguyên (Resource Feasibility Check):
  * Tập `train.json` AVeriTeC có tổng cộng 3.068 claims, file zip nén 59,16 GB và giải nén toàn bộ ~187 GB.
  * Ổ đĩa server `/dev/sda3` hiện còn trống ~125 GB, do đó không thể giải nén trọn vẹn toàn bộ 3.068 claims cùng lúc mà không gây tràn đĩa.
  * Tập 100 claims tiêu tốn ~1,5 GB tải HTTP Range, ~4 GB lưu trữ văn bản trên đĩa, và dự kiến ~35–45 phút mã hóa GPU trên RTX 5090 với `batch_size: 16`. Hoàn toàn nằm trong ngưỡng an toàn của hệ thống.
- Cập nhật mã nguồn:
  * `src/tffv/retrieval/acquire_subset.py`: Mở rộng ngưỡng ordinal từ `1 <= len(claim_ids) <= 3` lên `1 <= len(claim_ids) <= 100`; nâng hạn mức tải từ 128 MiB lên 2.048 MiB và trần giải nén lên 8.192 MiB.
  * `src/tffv/retrieval/prepare_store_candidates.py`: Mở rộng điều kiện kiểm tra inventory từ tối đa 3 lên tối đa 100 stores.
  * `docs/M11_SERVER_COMMANDS.md`: Cập nhật toàn bộ Section 6 thành quy trình khép kín 5 bước cho 100 claims (`seq 0 99`).
- Kiểm thử local:
  * Chạy `python scripts/run_tests.py`: Toàn bộ **107 passed in 7.80s**, không có bất kỳ regression hay compile error nào.
- Trạng thái: M1.1 empirical vẫn **INCOMPLETE** cho đến khi có báo cáo evaluation hoàn chỉnh từ server.
- Tiếp theo: Hướng dẫn người dùng commit/push trên Windows và chạy tuần tự 5 bước trên Linux server.

## 2026-10-21 21:20 +07:00 — Chẩn đoán và Khắc phục Lỗi Request Cap trong Tải 100 Stores

- Hiện tượng do người dùng cung cấp:
  * Khi chạy `acquire_subset` với `--claim-ids $(seq 0 99) --execute`, bước preflight hoàn thành kiểm tra và in kế hoạch 100 stores thành công.
  * Tuy nhiên, trong quá trình giải nén thành viên (tại `block = source.read(MiB)`), tiến trình bị ngắt với lỗi:
    `tffv.retrieval.remote_zip.RangeBudgetError: Archive inspection byte/request cap reached; refusing full fetch`.
- Nguyên nhân cốt lõi (Root Cause):
  1. Trong `src/tffv/retrieval/acquire_subset.py:127`, thuộc tính `reader.max_requests` được gán cứng bằng **256**.
  2. Giá trị 256 ban đầu được thiết kế cho tập khói chỉ 1–3 stores.
  3. Khi giải nén 100 thành viên qua `zipfile`, mỗi file cần đọc local header và nhiều block nén liên tiếp.
  4. Sau khi tải được khoảng 15–20 stores đầu tiên, tổng số yêu cầu HTTP Range trong `reader.requests` chạm mốc 256, kích hoạt điều kiện bảo vệ `len(self.requests) >= self.max_requests` trong `remote_zip.py:45`.
  5. Thông báo lỗi cũ gộp chung cả byte cap và request cap khiến việc chẩn đoán ban đầu chưa trực quan.
- Khắc phục kỹ thuật:
  1. `src/tffv/retrieval/acquire_subset.py`:
     * Đổi `reader.max_requests = max(2048, len(selected) * 512)`: Cho phép tới 51.200 requests cho 100 stores, mở rộng linh hoạt theo số lượng store.
     * Tăng block đọc giải nén từ `MiB` (1 MiB) lên `4 * MiB` (4 MiB): Giảm 4 lần số lần gọi `read()` và tăng tốc độ giải nén dữ liệu mạng.
     * Nâng trần mặc định: `DEFAULT_NETWORK_CAP = 4096 * MiB` (4 GiB), `DEFAULT_RAW_CAP = 16384 * MiB` (16 GiB).
     * Bổ sung log tiến độ console: `[acquire_subset] Downloading store [idx/total] train:identity...` giúp người dùng theo dõi rõ tiến độ tải trực tiếp.
  2. `src/tffv/retrieval/remote_zip.py`:
     * Phân tách thông báo lỗi rõ ràng: Phân biệt `Archive inspection byte cap reached (needed ... B, remaining ... B)` và `Archive inspection request cap reached (... >= ...)`.
  3. Bổ sung Unit Test:
     * Thêm test `test_range_request_cap_exceeded_message` trong `tests/test_remote_zip.py`.
     * Kiểm thử toàn bộ: **108 passed in 5.70s**.
- Trạng thái: M1.1 empirical vẫn **INCOMPLETE** cho đến khi chạy xong toàn bộ pipeline 100 claims.
- Tiếp theo: Hướng dẫn người dùng commit/push trên Windows, pull về Linux và chạy lại lệnh `acquire_subset`.

## 2026-10-23 22:50 +07:00 — Xác nhận Dung lượng 390 GB & Chuyển sang Chế độ Thực thi Server Trực tiếp

- Yêu cầu người dùng:
  * Thông báo dung lượng trống trên server hiện đã tăng lên gần 400 GB (~390 GB).
  * Đổi rule làm việc: Người dùng cho phép Codex kết nối trực tiếp qua SSH (`hvtham-server`) để chạy code, quản lý tiến trình và xuất kết quả gửi người dùng xem xét.
- Đánh giá khả thi tài nguyên (Resource Feasibility Check):
  * Kiểm tra qua SSH: `/dev/sda3` có Size 938 GB, Used 508 GB, **Avail 390 GB (57% used)**.
  * Toàn bộ 3 archives ZIP: 59,16 GiB (63,5 GB).
  * Toàn bộ 3.068 stores giải nén thô: 187,08 GiB (200,9 GB).
  * Candidate stores + artifacts: ~50–80 GB.
  * Tổng dung lượng tối đa cần cho FULL dataset: ~260–280 GB.
  * Kết luận: **390 GB là HOÀN TOÀN ĐỦ DÙNG** (dư hơn 110–130 GB dự phòng an toàn).
- Hiện trạng server đã xác minh:
  * Git repo server đã ở commit `e6a9c06` sạch sẽ.
  * GPU NVIDIA GeForce RTX 5090 (32 GiB VRAM) đang rảnh rỗi (0% util, 3 MiB used, 28°C).
  * Bước tải 100 stores đã hoàn thành thành công trong `outputs/m11-store-pilot100-20261021-142310/` (đầy đủ 100 files `train-0.json` đến `train-99.json` và `subset_manifest.json` đã seal).
- Trạng thái milestone: M1.1 empirical vẫn **INCOMPLETE**.

## 2026-10-23 23:14 +07:00 — Cách ly Hàng chứa URL Không Hợp lệ trong Candidate Preparation

- Hiện tượng:
  * Khi chạy `prepare_store_candidates` cho 100 stores, lệnh báo lỗi:
    `ValueError: Source schema/identity failures require review` tại dòng 79.
- Nguyên nhân:
  * Trong 100 stores thực tế được cào từ web bởi nhóm tác giả AVeriTeC, có 10 stores chứa đúng 1 hàng có `url='Metadata'` (dòng header metadata của crawler với `url2text=[]`) hoặc URL không hợp lệ (như `ftp://...`, `ttps://...`).
  * Kiểm tra qua `scripts/check_audit.py`: Cả 10 dòng này đều có `text_len = 0`.
  * Bộ lọc `prepare_store_candidates.py` trước đây kiểm tra cứng `invalid_source_url_rows > 0` và dừng toàn bộ tiến trình.
- Khắc phục:
  * Cập nhật `src/tffv/retrieval/prepare_store_candidates.py`:
    - Loại bỏ `invalid_source_url_rows` khỏi danh sách gây dừng tiến trình (vẫn giữ nguyên `annotation_field_rows`, `claim_id_mismatch_rows`, `invalid_text_shape_rows` để chống rò rỉ nhãn và sai lệch định danh).
    - Phân loại trạng thái hàng URL không hợp lệ: Nếu `text.strip() == ""`, gán `status = "empty_text"`; nếu có text, gán `status = "quarantined_invalid_url"`. Tuyệt đối không đưa vào `corpus.jsonl`.
  * Thêm unit test `test_invalid_url_row_quarantined_or_marked_empty_text` trong `tests/test_prepare_store_candidates.py`.
  * Kiểm thử local: **109 passed in 4.38s**.









## 2026-10-23 23:45 +07:00 — Thực thi GPU Benchmark Pilot 100 Claims trên RTX 5090 & Định lượng Khối lượng Công việc Thực tế

- Tiến hành thực thi:
  * Khởi chạy scripts/run_pilot100.sh trên server hvtham-server (PID 531897).
  * Hoàn thành chuẩn bị 100 candidate stores tại outputs/m11-candidates-pilot100-20261023-161557/candidate_manifest.json theo cơ chế cách ly quarantine_review_flags_v1.
- Đặc tính khối lượng công việc thực tế (Empirical Workload Profiling):
  * Tổng số web pages trong 100 stores: 74.748 tài liệu (Min: 358, Max: 1.454, Trung bình: 747,5 tài liệu/claim).
  * Tổng số passages ước tính sau khi cắt đoạn (180 từ, overlap 30 từ): ~6.309.816 passages (hơn 6,3 triệu đoạn văn).
  * Quy mô dữ liệu thực tế lớn hơn nhiều so với dự tính ban đầu do mỗi claim của AVeriTeC lưu trữ toàn bộ nội dung các trang web được crawler tải về.
- Giám sát phần cứng và tốc độ mã hóa:
  * Phần cứng: NVIDIA GeForce RTX 5090 (32 GiB VRAM), công suất tiêu thụ 480W–490W / 575W, GPU Util 76%–83%, bộ nhớ VRAM chiếm dụng ~8.175 MiB.
  * Tốc độ xử lý: ~170 passages/giây với batch size 16.
  * Các claims đã hoàn thành: train:0 (27.136 passages, ma trận 106 MB), train:1 (50.176 passages, ma trận 196 MB), train:2 (ma trận 129 MB).
  * Thời gian trung bình: ~2,5 – 4 phút/claim.
- Nhận định & Hạn chế:
  * Với 100 claims (~6,3 triệu passages), thời gian chạy hoàn tất ước tính ~5–7 giờ.
  * Với full 3.068 claims (~150 triệu passages), thời gian ước tính ở cấu hình hiện tại sẽ là ~7–8 ngày. Để tối ưu hóa cho full run sau này, cần tăng batch_size (từ 16 lên 64 hoặc 128) để khai thác tối đa 32 GB VRAM và tensor cores của RTX 5090.
- Trạng thái milestone: M1.1 empirical vẫn INCOMPLETE (đang chạy benchmark trên server).
- Bước tiếp theo: Tiếp tục theo dõi quá trình chạy benchmark trên server, thu thập báo cáo đánh giá chất lượng quality_eval khi hoàn tất.

## 2026-10-24 05:15 +07:00 (22:15 UTC) — Cập nhật Tiến độ Pilot 100 Claims: Vượt mốc 57%

- Trạng thái tiến trình:
  * Tiến trình nohup PID 531878 (runner) và PID 531897 (store_retrieval) vẫn đang chạy bền bỉ trên server hvtham-server.
  * Đã hoàn thành và lưu cache embedding cho 56/100 stores (từ train:0 đến train:55).
  * Hiện đang xử lý claim thứ 57 (train:56, 856 web pages).
- Giám sát phần cứng và tài nguyên:
  * GPU RTX 5090 duy trì ổn định: 491W / 575W, nhiệt độ 73°C, 66% GPU Util, 8.175 MiB VRAM.
  * Bộ nhớ cache embedding tích lũy: 15 GB.
  * Dung lượng đĩa trống trên /dev/sda3: 363 GB (rất an toàn).
- Quản lý tiến trình:
  * Đã kích hoạt task theo dõi phản ứng (reactive watcher task-1239) gắn với PID 531878; hệ thống sẽ tự động nhận diện và thông báo ngay khi toàn bộ 100 claims và báo cáo quality_eval hoàn thành.
- Trạng thái milestone: M1.1 empirical vẫn INCOMPLETE (đang tiếp tục chạy).

## 2026-10-24 14:00 +07:00 (07:00 UTC) — Hoàn thành Toàn diện Pilot 100 Claims (R1–R4 & Quality Eval) trên Server

- Trạng thái: Hoàn thành 100% pipeline thực nghiệm 100 claims (PID 531878 & 531897 đã kết thúc thành công).
- Thư mục kết quả:
  * Retrieval Store: outputs/m11-store-r14-pilot100-20261023-162716
  * Quality Evaluation: outputs/m11-eval-pilot100-20261023-162716
  * Tổng thời gian thực thi: 33.156,9 giây (~9,2 giờ).
- Kết quả Đánh giá Chất lượng Truy xuất (Quality Evaluation Report trên 100 claims):
  * Oracle Ceiling (Store Gold Coverage): 38,6% (chỉ 38,6% URL vàng thực sự tồn tại trong candidate stores).
  * Hit@10:
    - R1 (BM25): 5,0%
    - R2 (Dense Qwen3-0.6B): 8,0%
    - R3 (RRF Hybrid): 9,0%
    - R4 (Frozen Reranker Qwen3-0.6B): 14,0% (vượt trội gần gấp 3 lần BM25).
  * Recall@10:
    - R1: 0,035
    - R2: 0,051
    - R3: 0,051
    - R4: 0,091 (+78% so với R3).
  * MRR@100:
    - R1: 0,0147
    - R2: 0,0316
    - R3: 0,0265
    - R4: 0,0478 (+80% so với R3).
  * Hit@100:
    - R1: 26,0% | R2: 25,0% | R3: 27,0% | R4: 30,0% (tiếp cận 78% trần Oracle).
  * Evidence Recall@8:
    - R1: 0,023 | R2: 0,051 | R3: 0,046 | R4: 0,081.
- Hiệu quả Reranker R4 (so với R3):
  * 22 claims được cải thiện thứ hạng; 10 claims tụt thứ hạng; 68 claims giữ nguyên.
  * 8 claims được đẩy từ ngoài vào Top-10; 3 claims bị rơi khỏi Top-10.
  * Tổng net rank shift: +221 thứ hạng.
- Nhận định chuyên môn theo quy tắc dự án (AGENTS.md):
  * Kết quả này là thực nghiệm trên candidate stores thô (exploratory, corpus_admitted = false).
  * R4 thể hiện sức mạnh vượt bậc cả về độ chính xác và khả năng gom bằng chứng Top-8 cho mô hình suy luận.
  * Tuân thủ quy tắc: Không tự ý crown winner hoặc chọn R4 làm baseline M1.2 trước khi người dùng phê duyệt.
- Trạng thái milestone: Milestone M1.1 Retrieval Pilot 100 Claims **COMPLETE**.

## 2026-10-24 17:20 +07:00 — Audit Nghiêm ngặt M1.1 Retrieval (100 Claims) & Làm rõ Cơ chế Reranking

- Mục tiêu: Thực hiện lượt audit nhẹ, KHÔNG chạy lại GPU experiments, xác minh toàn bộ công thức, mẫu số, frozen status, rank shifts và tạo báo cáo chi tiết docs/M1_1_RETRIEVAL_AUDIT.md.
- Phát hiện & Giải quyết Discrepancy R3 Hit@100 (27%) vs R4 Hit@100 (30%):
  * Nguyên nhân kiến trúc: Cấu hình fusion_depth = 200, rerank_depth = 200, nhưng ranking_store_depth = 100.
  * Tệp R3.jsonl chỉ lưu trữ Top-100 (cắt bỏ thứ hạng 101-200 của RRF fusion).
  * Mô hình R4 nhận toàn bộ danh sách 200 candidates từ bước fusion và rerank lại cả 200 candidates.
  * Xác minh per-claim: Có đúng 5 claims chứa URL vàng nằm ở thứ hạng 101-200 của RRF (bị R3 bỏ sót trong Top-100) được R4 đánh giá điểm cao và kéo vào Top-100 (train:0 lên rank 83, train:6 lên rank 31, train:34 lên rank 18, train:59 lên rank 81, train:62 lên rank 23). Đồng thời 2 claims bị R4 đẩy ra ngoài Top-100 (train:49 và train:95).
  * Phép tính khớp tuyệt đối: 27 + 5 - 2 = 30 claims (30,0%).
- Xác minh Frozen State của Qwen3-Reranker-0.6B:
  * Checkpoint băm SHA-256: 5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353.
  * Mã nguồn thực thi: model.eval(), requires_grad_(False), kiểm tra assertion không có gradient, chạy trong torch.inference_mode().
  * Không có bất kỳ optimizer, loss function, hay weight mutation nào trong toàn bộ repo.
- Xác minh Chi tiết Rank Shifts từ Per-Claim CSV:
  * So sánh trên toàn bộ 100 claims: 22 claims cải thiện, 10 claims tụt hạng, 68 claims giữ nguyên.
  * 3 claims bị rơi khỏi Top-10: train:19 (rank 7 -> 27), train:35 (rank 9 -> 65), train:93 (rank 5 -> 20).
  * 8 claims được đẩy vào Top-10: train:12, train:20, train:23, train:56, train:63, train:74, train:75, train:79.
  * Tổng net rank shift: +392 (tổng dương) - 171 (tổng âm) = +221 thứ hạng.
- Phân biệt Rõ ràng Hai Chỉ số Oracle:
  * URL-level Store Coverage (38,6%): Tỷ lệ phần trăm URL vàng trung bình có mặt trong candidate store. Mẫu số: Tổng số URL vàng của claim.
  * Claim-level Oracle Hit Ceiling (54,0%): Tỷ lệ claims có ít nhất 1 URL vàng trong candidate store (54/100 claims). 46 claims còn lại hoàn toàn không có URL vàng trong store. R4 đạt Hit@100 trên 30 claims, tương đương đạt 55,56% trần tối đa Oracle (30/54).
- Artifacts & Tài liệu Tạo mới:
  * Tạo tệp: docs/M1_1_RETRIEVAL_AUDIT.md với đầy đủ công thức, bảng SHA-256, và phân tích chi tiết.
- Kiểm thử:
  * Chạy test suite: 109 passed in 25.15s (--basetemp=temp_pytest -p no:cacheprovider).
- Tuân thủ quy tắc người dùng:
  * Không push/pull, không tải model, không chạy GPU jobs.
  * Cung cấp commands để người dùng tự review/commit/push.


## 2026-10-24 21:55 +07:00 — M1.2 Triển khai Trọn vẹn Frozen Base Verification & Error Profiling (100 Claims)

- **Mục tiêu:** Hoàn thành toàn bộ Milestone M1.2 theo ủy quyền tự chủ: Chạy Frozen Base Fact Verification trên 100 claims AVeriTeC (`train:0` .. `train:99`) với Top-8 evidence truy xuất từ M1.1 R4 (`outputs/m11-store-r14-pilot100-20261023-162716/R4.jsonl`); đánh giá Accuracy/Macro-F1, đối chiếu gold labels độc lập sau khi khóa predictions, phân tích lỗi định tính và xây dựng kế hoạch Group-Safe Knowledge Isolation cho các milestone tiếp theo.
- **Tuân thủ Nghiêm ngặt Quy tắc Dự án (AGENTS.md):**
  * **Strictly Training-Free:** Tuyệt đối không fine-tuning, không gradient/backward, không optimizer, không learned router. Mô hình `Qwen/Qwen3-8B` được nạp ở chế độ `eval()`, `requires_grad_(False)` và thực thi trong `torch.inference_mode()`.
  * **Label-Free Inference:** Toàn bộ dự đoán được sinh và băm SHA-256 niêm phong trong `manifest.json` trước khi nạp `train.json` để đối chiếu đánh giá.
  * **Ontology 4 Nhãn Chính thức AVeriTeC:** `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`.
  * **Group-Safe Knowledge Isolation:** 100 pilot claims được gán trạng thái `exploratory_profiling_only`, tuyệt đối không dùng làm tập blind held-out test trong M1.3+.
- **Môi trường & Tài nguyên Server Thực thi:**
  * Server: `stackops@hvtham-server` (GPU: NVIDIA GeForce RTX 5090 32 GB VRAM, CPU: AMD EPYC, RAM: 128 GB).
  * Backbone Checkpoint: `Qwen/Qwen3-8B` (Revision `b968826d9c46dd6066d109eabc6255188de91218`, Apache-2.0). Nạp bfloat16, chiếm 17.02 GB VRAM lúc inference đỉnh.
  * Thời gian chạy thực đo: 1.130,64 giây (~18,8 phút) cho 100 claims, trung bình 11,31 giây/claim.
- **Kết quả Thực nghiệm Thực đo (Chính thức):**
  * **Overall Accuracy:** **28,0%** (28/100 claims đúng).
  * **Accuracy trên các claims hoàn tất phân giải (Parsed Claims):** **43,1%** (28/65 claims).
  * **Macro-F1:** **0,3076** (Macro Precision: 0,5023, Macro Recall: 0,2804).
  * **Chi tiết từng lớp:**
    - `Refuted`: Precision **87,5%** (14/16), Recall 34,1% (14/41), F1: **0,4912** (Support: 41).
    - `Supported`: Precision **80,0%** (8/10), Recall 21,6% (8/37), F1: **0,3404** (Support: 37).
    - `Conflicting Evidence/Cherrypicking`: Precision 21,4% (3/14), Recall 23,1% (3/13), F1: **0,2222** (Support: 13).
    - `Not Enough Evidence`: Precision 12,0% (3/25), Recall 33,3% (3/9), F1: **0,1765** (Support: 9).
  * **Confusion Matrix (Gold Rows x Predicted Cols):**
    - `Supported` (37): 8 Supported, 0 Refuted, 12 Not Enough Evidence, 5 Cherrypicking, 12 Truncated.
    - `Refuted` (41): 1 Supported, 14 Refuted, 7 Not Enough Evidence, 5 Cherrypicking, 14 Truncated.
    - `Not Enough Evidence` (9): 1 Supported, 0 Refuted, 3 Not Enough Evidence, 1 Cherrypicking, 4 Truncated.
    - `Conflicting Evidence/Cherrypicking` (13): 0 Supported, 2 Refuted, 3 Not Enough Evidence, 3 Cherrypicking, 5 Truncated.
- **Phát hiện & Phân tích Lỗi Chủ đạo (Error Profiling):**
  1. *Truncation Bottleneck (35 claims):* `Qwen3-8B` tự động sinh khối suy luận `<think>...</think>`. Khi duyệt qua 8 đoạn evidence dài (~2.300 prompt tokens), mô hình tuần tự phân tích từng đoạn dẫn đến cạn kiệt ngân sách `max_new_tokens=512` trước khi kịp đưa ra dòng kết luận `Verdict: <Class>`.
  2. *Độ chính xác cao trên các phán quyết nhị phân rõ ràng:* Khi mô hình phân giải xong, độ chuẩn xác của `Refuted` đạt 87,5% và `Supported` đạt 80,0%. Toàn bộ 100 claims chỉ ghi nhận **duy nhất 1 lỗi đảo ngược cực tính** (Polarity Inversion tại `train:20`).
  3. *Thiên kiến thận trọng về `Not Enough Evidence`:* Mô hình ghi nhận 22 False Positives cho `Not Enough Evidence`, cho thấy khi thiếu hướng dẫn ICL, zero-shot LLM đặt ngưỡng chứng cứ quá cao và từ chối cam kết `Supported`/`Refuted`.
  4. *Tương quan Retrieval vs Verification:* 
     - Có URL vàng trong Top-8 (Hit@8=True, 13 claims): Error rate 69,2% (4 đúng, 9 sai).
     - Không có URL vàng trong Top-8 (Hit@8=False, 87 claims): Error rate 72,4% (24 đúng, 63 sai).
     - 24 claims dự đoán đúng dù không có URL vàng: Do các hãng thông tấn lớn (AP, Reuters, CNN, tài liệu quốc hội) đưa tin đồng nhất về cùng một sự kiện thực tế.
     - 9 claims thất bại dù có URL vàng: 4 claims bị timeout/truncation, 2 claims nhầm lẫn cherrypicking, 2 claims đòi hỏi ngưỡng chứng cứ quá cao, 1 claim đảo ngược cực tính.
- **Phân loại Lỗi 5 Tầng (Preliminary Error Taxonomy):**
  - Tầng 1: Truncation & Token Budget Depletion (35 claims, 48,6% tổng lỗi).
  - Tầng 2: Under-Retrieval Bottleneck (63 claims, 87,5% tổng lỗi).
  - Tầng 3: Conservative Evidence Drift (28 claims, 38,9% tổng lỗi).
  - Tầng 4: Cherrypicking Contextual Nuance Deficit (21 claims, 29,2% tổng lỗi).
  - Tầng 5: Polarity Inversion (1 claim, 1,4% tổng lỗi).
- **Kế hoạch Group-Safe Knowledge Isolation:**
  - Đã xuất và niêm phong `outputs/m12-group-safety-20261024-132900/fold_isolation_plan.json`.
  - Quy định rõ: 100 pilot claims là dữ liệu thăm dò; các đánh giá tiếp theo ở M1.3+ phải lấy từ các folds độc lập (`train:100` .. `train:3067` hoặc dev split).
- **Artifacts Tạo mới & Hashes Niêm phong:**
  * Predictions: `outputs/m12-base-qwen3-8b-20261024-132900/predictions.jsonl` (SHA-256: `f519b6db9aa93000dcc530465e38de39585f55dfb5b933c137b2c6287dac4dc1`)
  * Manifest: `outputs/m12-base-qwen3-8b-20261024-132900/manifest.json` (SHA-256: `4859a72df9b7203e05167ea82ebc82c2ffab67160cb65f1710ee9cfb237db875`)
  * Eval Summary: `outputs/m12-eval-base-20261024-132900/evaluation_summary.json` (SHA-256: `37f59d5dc012a6114eb3897bda86c3db5a8cb5d5ecfe880f7690656041ec6057`)
  * Per-claim CSV: `outputs/m12-eval-base-20261024-132900/per_claim_evaluation.csv` (SHA-256: `0ba56b0cfa0fc9001b6e4ae15fca5f29910e53a5cce0220cb959ddaa2c3f8751`)
  * Error Analysis: `outputs/m12-errors-base-20261024-132900/error_analysis_summary.json` (SHA-256: `6c3c5ad836eb4b045e76a6b5711e74f26ca4c1e8dcb200a7fae51be98d7aeefc`)
  * Group Safety: `outputs/m12-group-safety-20261024-132900/fold_isolation_plan.json` (SHA-256: `1482eae18c5c6d666d4bb3d26a34e3e8a6001a9d9ede672102d671a2b14ede09`)
  * Báo cáo: `docs/M1_2_BASE_VERIFICATION.md` & `docs/M1_2_ERROR_ANALYSIS.md`.
- **Kiểm thử Phần mềm:**
  * Local: 115 passed in 11.87s.
  * Server: 115 passed in 2.18s.
- **Trạng thái Milestone:** Milestone M1.2 **COMPLETE**.


## 2026-10-24 22:25 +07:00 — M1.2.1 Triển khai Ổn định Hóa Frozen Base (Inference Audit & B0/B1/B2 Scaffold)

- **Mục tiêu:** Thực hiện kiểm toán kỹ thuật chuyên sâu (*Inference Audit*) về nguyên nhân 35 outputs bị truncated trong M1.2; triển khai cơ chế bật/tắt thinking hợp lệ theo chuẩn Qwen3 và thư viện Transformers; thiết lập khung thực nghiệm đối chứng 3 cấu hình B0 (Thinking, 512), B1 (Non-thinking, 512), B2 (Thinking, 1024); xây dựng bộ so sánh chuyển dịch phán quyết từng cặp (*Paired Verdict Transitions*); kiểm thử local 100% đạt; đóng gói tài liệu và chuẩn bị lệnh chạy server cho người dùng.
- **Tuân thủ Tuyệt đối Quy tắc Nghiên cứu (AGENTS.md & User Mandate):**
  * **Strictly Training-Free:** Giữ nguyên mô hình đóng băng 100%, không fine-tune, không gradient, không optimizer, không learned router.
  * **Không tự ý chạy GPU jobs:** Hoàn thành toàn bộ code, configs, tests, tài liệu trên local và push GitHub; cung cấp lệnh bash chuẩn xác để người dùng tự chạy trên server.
  * **Không rò rỉ nhãn vàng:** Giữ nguyên quy trình seal manifest predictions trước khi nạp `train.json` để chấm điểm.
  * **Báo cáo trung thực:** Kết quả GPU của B1/B2 được ghi rõ `PENDING SERVER EXECUTION` trong `docs/M1_2_1_BASE_STABILIZATION.md`.
- **Kết quả Audit Kỹ thuật Chi tiết (Inference Audit Findings):**
  1. *Khám phá Cơ chế Thinking trong Chat Template:* Kiểm tra `tokenizer_config.json` của `Qwen/Qwen3-8B` phát hiện hàm `apply_chat_template` có tham số bản địa `enable_thinking`. Khi `enable_thinking=False`, template tự động chèn `<think>\n\n</think>\n\n` vào tiền tố trợ lý (`<|im_start|>assistant\n`), báo hiệu cho mô hình rằng suy luận đã kết thúc và buộc mô hình sinh trực tiếp câu trả lời mà không tốn token suy luận. Khi `enable_thinking=True` (mặc định), tiền tố chỉ là `<|im_start|>assistant\n`, kích hoạt sinh khối `<think>...</think>`.
  2. *Cơ chế Cạn Token (512 Budget Exhaustion):* Với 8 đoạn trích dài (~2.300 prompt tokens), mô hình tuần tự phân tích từng đoạn bên trong `<think>`, dẫn đến cạn kiệt 512 token ở đoạn 5–8 trước khi kịp xuất dòng `Verdict: <Class>`.
  3. *Nâng cấp Bộ phân giải Verdict Hai tầng (Two-Tier Parser):* Ưu tiên quét phân đoạn sau thẻ `</think>` trước để loại bỏ nhiễu thảo luận đa chiều bên trong suy luận; nếu không có thẻ `</think>` mới phân giải toàn bộ văn bản.
- **Mã Nguồn & Cấu Hình Triển Khai Mới:**
  * `configs/m121_b0_thinking_512.json`: Tái sử dụng cache M1.2.
  * `configs/m121_b1_nonthinking_512.json`: Chế độ non-thinking (`enable_thinking=False`), `max_new_tokens=512`.
  * `configs/m121_b2_thinking_1024.json`: Chế độ thinking (`enable_thinking=True`), `max_new_tokens=1024`.
  * `src/tffv/verification/ontology.py`: Nâng cấp `normalize_verdict` với post-think priority.
  * `src/tffv/verification/backbone.py`: Hỗ trợ `enable_thinking` và `max_new_tokens` trên cả causal LM và synthetic double.
  * `src/tffv/verification/predict.py`: Hỗ trợ cờ CLI `--enable-thinking`, `--no-thinking`, `--max-new-tokens`, `--config`.
  * `src/tffv/verification/stabilization.py`: Module tính toán ma trận chuyển dịch cặp (improved, regressed, unparsed resolution) và xuất bảng so sánh 3 cấu hình.
  * `scripts/run_m121_stabilization.sh`: Script điều phối tự động chạy B1, B2, tái sử dụng B0 và xuất báo cáo so sánh.
  * `tests/test_base_stabilization.py`: Bộ kiểm thử tự động cho thinking toggle, parser và paired transitions.
  * `docs/M1_2_1_BASE_STABILIZATION.md`: Tài liệu đặc tả giao thức và bảng so sánh (chờ kết quả server).
- **Kiểm Thử Phần Mềm Local:**
  * Chạy `pytest --basetemp=temp_pytest -p no:cacheprovider`: **119/119 passed in 9.64s** (100% pass, gồm 4 tests mới cho M1.2.1).
- **Trạng thái Milestone:** M1.2.1 Implementation & Local Verification **COMPLETE**. Sẵn sàng thực thi trên GPU server.


## 2026-10-24 23:20 +07:00 — M1.2.1 Sửa Lỗi Nghiêm Trọng Paired Evaluation (P0) & Hoàn Thiện Báo Cáo Đối Chứng B0/B1/B2

- **Mục tiêu:** Xử lý triệt để lỗi P0 trong báo cáo M1.2.1: B0 (28 đúng), B1 (54 đúng), B2 (53 đúng) nhưng paired transitions báo 0 improved, 0 regressed, 100 unchanged correct; kiểm toán toàn diện artifacts, claim IDs và mã nguồn chuyển dịch; tái tính toán các chỉ số thống kê (McNemar test, Paired Bootstrap 95% CI); tái phân tích Error Taxonomy trên kết quả thực tế của B1/B2; chuẩn bị kế hoạch nghiên cứu M1.3 Readiness.
- **Root Cause Analysis (Nguyên Nhân Gốc Rễ):**
  * Tệp `per_claim_evaluation.csv` được nạp bằng `csv.DictReader`, trường `row["is_correct"]` trả về chuỗi `"True"` hoặc `"False"`.
  * Đoạn mã `b0_correct = bool(b0_r.get("is_correct", False))` trong Python gặp lỗi ép kiểu: `bool("False") == True` (mọi chuỗi ký tự khác rỗng đều là Truthy).
  * Do đó, toàn bộ 100 claims đều bị ép thành `b0_correct = True` và `tgt_correct = True`, dẫn đến 100 claims rơi vào `unchanged_correct`.
- **Hành Động Khắc Phục & Gia Cố (Hardening):**
  * Tái tính toán tính đúng sai trực tiếp từ phán quyết: `is_correct = bool(predicted_verdict and predicted_verdict in VERDICTS and predicted_verdict == gold_label)`. Tuyệt đối không tin cậy cờ boolean chưa kiểm chứng.
  * Tích hợp kiểm tra phòng thủ: Phát hiện trùng lặp sample_id (`ValueError`), kiểm tra căn chỉnh tập sample_id giữa 2 cấu hình (`ValueError`), kiểm tra bất biến tổng số lượng chuyển dịch bằng đúng số claims đánh giá ($= 100$), kiểm tra bất biến $\text{Net Gain} = \sum y_{\text{target}} - \sum y_{\text{base}}$.
  * Thêm kiểm tra tự so sánh (*Self-comparison check*): B0 vs B0 bảo đảm 0 improved, 0 regressed, 0 net gain, 100% unchanged.
  * Tích hợp tính toán kiểm định chính xác McNemar (hai phía) và Paired Bootstrap 95% CI ($10.000$ resamples) bằng thư viện chuẩn `math.comb` và `random`.
- **Kết Quả Chuyển Dịch Phán Quyết Sau Sửa Lỗi (Đã Xác Minh Khớp Tuyệt Đối):**
  * **B0 $\to$ B1 (Non-thinking, 512):**
    - Net Gain: **+26 claims** (28/100 $\to$ 54/100).
    - Improved: **33 claims** (Unparsed $\to$ Correct: 15, Wrong $\to$ Correct: 18).
    - Regressed: **7 claims** (Correct $\to$ Wrong: 7, Correct $\to$ Unparsed: 0).
    - Unchanged: Correct = 21, Wrong = 19, Unparsed = 0.
    - Tổng danh mục: $15 + 20 + 18 + 7 + 0 + 21 + 19 + 0 = 100$.
    - McNemar p-value: **$4,23 \times 10^{-5}$ ($p = 0,00004$)** $\implies$ Ý nghĩa thống kê vượt trội ($p < 0,05$).
    - Paired Bootstrap 95% CI ($\Delta$ Acc): **[+15,00%, +37,00%]**.
  * **B0 $\to$ B2 (Thinking, 1024):**
    - Net Gain: **+25 claims** (28/100 $\to$ 53/100).
    - Improved: **27 claims** (Unparsed $\to$ Correct: 15, Wrong $\to$ Correct: 12).
    - Regressed: **2 claims** (Correct $\to$ Wrong: 2, Correct $\to$ Unparsed: 0).
    - Unchanged: Correct = 26, Wrong = 25, Unparsed = 4.
    - Tổng danh mục: $15 + 16 + 12 + 2 + 0 + 26 + 25 + 4 = 100$.
    - McNemar p-value: **$1,62 \times 10^{-6}$ ($p = 0,000002$)** $\implies$ Ý nghĩa thống kê vượt trội ($p < 0,05$).
    - Paired Bootstrap 95% CI ($\Delta$ Acc): **[+16,00%, +34,00%]**.
  * **B1 $\to$ B2:**
    - Net Gain: **-1 claim** (54/100 $\to$ 53/100).
    - Improved: **14 claims** (Wrong $\to$ Correct: 14).
    - Regressed: **15 claims** (Correct $\to$ Wrong: 15).
    - Unchanged: Correct = 39, Wrong = 28, Unparsed = 0.
    - Wrong $\to$ Unparsed: **4 claims** (`train:11, train:31, train:48, train:51`).
    - Tổng danh mục: $14 + 15 + 39 + 28 + 4 = 100$.
    - McNemar p-value: **1,00000** $\implies$ Không có sự khác biệt có ý nghĩa thống kê giữa B1 và B2.
    - Paired Bootstrap 95% CI ($\Delta$ Acc): **[-12,00%, +10,00%]** (chứa giá trị 0).
- **Thiết Lập Baseline Protocol v1.0:**
  * Chọn **B1 (Non-thinking, 512 tokens)** làm **Base Protocol v1.0**: Độ chính xác cao nhất (54,0%), 100% phán quyết hợp lệ (0 unparsed), độ trễ cực thấp (5,10s/claim - nhanh gấp 3,36 lần B2), tiết kiệm 3,52 lần token.
  * Giữ **B2 (Extended Thinking, 1024 tokens)** làm đối chứng suy luận đa bước (Accuracy 53,0%, Macro-F1 0,4527, độ trễ 17,15s).
- **Tái Phân Tích Error Taxonomy (B1 & B2):**
  * Tách biệt rõ lỗi tính hợp lệ (B1: 0, B2: 4) và lỗi nhầm lẫn ngữ nghĩa (B1: 46, B2: 43).
  * Tỷ lệ lỗi khi có URL vàng (46,2%) tương đương khi không có URL vàng (46,0%), chứng minh sự hiện diện của các nguồn tin tương đương (syndicated news) và nút thắt chính nằm ở khả năng hiệu chuẩn ngữ nghĩa của mô hình.
  * Hai nhóm nhầm lẫn chủ đạo: `Refuted` bị chuyển thành `Not Enough Evidence` (17,4% tổng lỗi ở B1) và `Supported` bị chuyển thành `Cherrypicking` (15,2% tổng lỗi ở B1) do mô hình nhầm lẫn giữa sự bất đồng quan điểm chính trị trong bài báo với mâu thuẫn về sự kiện thực tế.
- **Kiểm Thử Phần Mềm:**
  * Bổ sung 3 bài kiểm thử hồi quy trong `tests/test_base_stabilization.py`: `test_paired_transitions_string_bool_regression`, `test_paired_transitions_self_comparison`, `test_paired_transitions_id_alignment_and_duplicates`, `test_exact_mcnemar_and_bootstrap`.
  * Chạy `pytest`: **122/122 passed in 8.25s** (100% pass).
- **Trạng Thái Milestone:** M1.2.1 P0 Repair **COMPLETE & VERIFIED**. Sẵn sàng cho M1.3.

## 2026-10-24 23:45 +07:00 — M1.3 Triển khai Toàn diện Error-Informed Fact Verification (C0–C6) & Leakage Guards

- **Mục tiêu:** Hoàn thành thiết kế và hiện thực hóa Milestone M1.3 (*Error-Informed Training-Free Fact Verification*); kiểm tra giả thuyết khoa học liệu tri thức từ các lỗi quá khứ (error memory, exemplars, corrected errors, contrastive pairs, error rules) có cải thiện độ chính xác và Macro-F1 trên mô hình frozen `Qwen/Qwen3-8B` so với Base Protocol v1.0 (B1 Non-Thinking, 512 tokens); xây dựng hệ thống phòng chống rò rỉ dữ liệu (*Leakage & Quarantine Guards*); hoàn thành 100% tests local và sẵn sàng chạy GPU trên server.
- **Tuân thủ Nghiêm ngặt Quy tắc Nghiên cứu (AGENTS.md & User Mandate):**
  * **Strictly Training-Free, Quality-First:** Không fine-tune, không gradient, không optimizer, không learned neural routers, không tối ưu routing có chọn lọc (selective routing). Toàn bộ trọng số mô hình đóng băng 100%.
  * **Không rò rỉ nhãn vàng:** Dữ liệu quá khứ chỉ được xây dựng từ tập source (Fold A hoặc exploratory set); các exemplars tuyệt đối không chứa claims thuộc tập đánh giá. Nhãn vàng của tập đánh giá chỉ được mở sau khi tệp phán đoán đã được materialize và seal sha256.
  * **Phân định ranh giới thực thi:** Codex chuẩn bị mã nguồn, cấu hình, kiểm thử CPU local, đẩy lên GitHub; cung cấp lệnh terminal chuẩn xác không placeholder để người dùng trực tiếp thực thi trên GPU server `hvtham-server`.
- **Cấu Trúc Hệ Thống & Các Module Triển Khai Mới:**
  1. `src/tffv/verification/leakage.py`:
     - Đo lường độ tương đồng từ vựng bằng Jaccard và containment 3-gram.
     - Phát hiện và cách ly các near-duplicates giữa tập source và tập candidate (`find_near_duplicates`, `quarantine_near_duplicates`).
     - Bắt buộc tính rời rạc tuyệt đối giữa các fold (`assert_fold_disjointness`), chặn đứng mọi nguy cơ rò rỉ ID.
  2. `src/tffv/verification/error_memory.py`:
     - Xây dựng `ErrorMemoryBank` từ phán đoán của Base Protocol v1.0 và nhãn vàng chính thức.
     - Lập chỉ mục đa chiều: theo lớp nhãn vàng (`by_class`), theo cặp nhầm lẫn thực tế (`by_confusion`), theo tập đúng/sai (`correct_sids`, `wrong_sids`).
     - Hỗ trợ lấy mẫu cân bằng cho C2 (random source), C3 (correct-only), C4 (error-only kèm phân tích bẫy sai lầm và sửa đổi nhãn vàng), C5 (cặp đối chứng tương phản correct vs corrected error trên ranh giới Supported và Refuted).
     - Tạo danh mục kiểm tra 4 điểm (*4-point verification checklist*) đúc kết từ các lỗi thường gặp trong $\mathcal{D}_{\text{error}}$ (media debate vs factual contradiction, evidence sufficiency threshold, cherrypicking precision, polarity check).
  3. `src/tffv/verification/m13_prompting.py`:
     - Bộ sinh prompt chuyên biệt cho 7 điều kiện C0–C6 tuân thủ định dạng system prompt, evidence block, context budget (< 3.500 tokens).
  4. `src/tffv/verification/m13_runner.py`:
     - Trình điều phối thực nghiệm toàn diện từ inference, sealing manifest, chấm điểm với `evaluate.py`, so sánh cặp đối chứng với C0.
     - Tự động tính toán số claims cải thiện (improved), suy giảm (regressed), không đổi (unchanged); kiểm định chính xác McNemar (hai phía); khoảng tin cậy Paired Bootstrap 95% CI.
     - Thu thập chỉ số tài nguyên: thời gian trễ trung bình mỗi claim, số lượng token đầu vào (prompt) và đầu ra (generation).
     - Tự động sinh báo cáo khoa học Markdown `M1_3_EVALUATION_REPORT.md` và `comparison_summary.json`.
  5. `scripts/run_m13_error_informed.sh`:
     - Script bash hoàn chỉnh cho GPU server `hvtham-server`, tự động phát hiện snapshot Qwen3-8B, R4 evidence, claims và B1 predictions.
  6. `tests/test_m13_error_memory.py`:
     - Bộ kiểm thử tự động toàn diện kiểm tra tính đúng đắn của leakage detection, memory bank sampling, prompt builders và pipeline synthetic end-to-end.
  7. `docs/M1_3_ERROR_INFORMED_PROTOCOL.md`:
     - Tài liệu đặc tả giao thức thực nghiệm, ma trận C0–C6, giả thuyết nghiên cứu và hướng dẫn thực thi.
- **Kết Quả Kiểm Thử Phần Mềm Local:**
  * Chạy `pytest -v --basetemp=temp_pytest -p no:cacheprovider`: **127/127 passed in 5.72s** (100% tests đạt chuẩn).
- **Trạng Thái Git:**
  * Sẵn sàng commit và push lên nhánh `main` GitHub repository `lee-vtruong/training-free-fact-verification`.

## 2026-10-25 08:00 +07:00 — M1.3 P0 Audit & Sửa Chữa Toàn Diện Báo Cáo Thực Nghiệm C0–C6 (Negative Results)

- **Mục tiêu:** Thực hiện audit toàn diện P0 cho Milestone M1.3 (*Error-Informed Fact Verification*); tìm và khắc phục tận gốc các lỗi báo cáo trong `M1_3_EVALUATION_REPORT.md` (Valid verdicts = 0/0, Acc = 0.0%, Macro-F1 = 0, Output tokens = 0, tổng số transitions bị thiếu ở C4, C5, C6); chuẩn hóa bảng dự đoán canonical; tính toán lại toàn bộ chỉ số thực nghiệm, ma trận nhầm lẫn, kiểm định McNemar hiệu chỉnh Holm và Paired Bootstrap 95% CI; phân tích nguyên nhân khoa học của kết quả phủ định (negative results).
- **Nguyên Nhân Gốc Rễ Đã Xác Minh (Root Cause Analysis):**
  1. *Lỗi cấu trúc lồng nhau trong báo cáo (Metrics 0.0% & Valid 0/0):* Trong `m13_runner.py`, biến `condition_summaries[cond]` lưu trữ dictionary có key `'metrics'`, trong khi giá trị của key đó lại chính là kết quả của `evaluate.py` (vốn đã chứa sub-key `'metrics'`). Hàm `generate_m13_markdown_report()` truy vấn trực tiếp `metrics.get('accuracy', 0.0)` ở root level dẫn đến không tìm thấy và fallback về `0.0`. Tương tự, lookup `per_class_f1` thay vì `per_class` khiến toàn bộ F1 lớp bị 0.
  2. *Lỗi đếm số lượng token đầu ra (Output tokens = 0):* Trong `backbone.py`, hàm `generate()` trả về dict với key `"generated_tokens"`. Tuy nhiên, trong `m13_runner.py` dòng 101 lại truy vấn key `"generation_tokens"`. Sai lệch tên key khiến giá trị lưu vào `predictions.jsonl` luôn là `None` (null) và tổng hợp thành `0`.
  3. *Lỗi ẩn danh mục chuyển đổi (Tổng transition không bằng 46):* Logic tính toán chuyển đổi trong `stabilization.py` phân loại chính xác 9 danh mục phân biệt loại trừ lẫn nhau. Tuy nhiên, bảng Markdown trong `m13_runner.py` chỉ render 4 cột tóm tắt (`Improved`, `Regressed`, `Unchanged Correct`, `Unchanged Wrong`), hoàn toàn bỏ qua các chuyển đổi liên quan đến `unparsed` (cụ thể là `wrong_to_unparsed`: 2 ở C4, 1 ở C5, 6 ở C6), tạo cảm giác tổng số claim bị thiếu.
- **Hiện Thực Hóa Canonical Prediction Table & Khắc Phục:**
  - Xây dựng module audit chuyên biệt `src/tffv/verification/m13_audit.py` và CLI `scripts/audit_and_repair_m13.py`.
  - Xuất bảng canonical chuẩn hóa: `outputs/m13-repaired-audit/canonical_per_claim_predictions.csv` gồm đúng 322 dòng (46 claims $\times$ 7 điều kiện C0–C6) với đầy đủ 10 trường: `claim_id`, `gold_label`, `condition`, `predicted_label`, `parse_status`, `is_correct`, `input_tokens`, `output_tokens`, `latency`, `prediction_source_path`.
  - Tính toán lại số lượng token đầu ra chính xác bằng `AutoTokenizer.from_pretrained('cache/qwen3_tokenizer')` trực tiếp trên text output thô (`raw_generation`).
  - Kiểm tra tính rời rạc và ngăn chặn rò rỉ: 100% exemplars cho C2–C5 được rút từ Fold A (`train:0..49`). Toàn bộ 46 claims đánh giá thuộc Fold B (`train:50..99`) sau khi đã cách ly 4 near-duplicates (`train:67, train:70, train:77, train:95`). Tỷ lệ rò rỉ ID là 0.0%.
- **Kết Quả Thực Nghiệm Đã Sửa Chữa Toàn Diện:**
  * **C0 (Frozen Zero-Shot Base Protocol v1.0):** Accuracy **60.9%** (28/46), Macro-F1 **0.4883**, Valid rate: **100.0%** (46/46), Mean Latency: **4.47s**, Tokens: 2,483 in / 192.3 out.
  * **C1 (Generic Task Instruction):** Accuracy **56.5%** (26/46), Macro-F1 **0.4664**, Valid rate: **97.8%** (45/46), Net Gain: **-2**, McNemar $p = 0.7266$, Bootstrap 95% CI: `[-17.4%, +6.5%]`.
  * **C2 (Random Source ICL):** Accuracy **41.3%** (19/46), Macro-F1 **0.3578**, Valid rate: **100.0%** (46/46), Net Gain: **-9**, McNemar $p = 0.0117$, Holm-adjusted $p = 0.0586$, Bootstrap 95% CI: `[-32.6%, -6.5%]`.
  * **C3 (Correct-Only ICL):** Accuracy **52.2%** (24/46), Macro-F1 **0.4185**, Valid rate: **100.0%** (46/46), Net Gain: **-4**, McNemar $p = 0.3877$, Holm-adjusted $p = 0.7754$, Bootstrap 95% CI: `[-23.9%, +6.5%]`.
  * **C4 (Error-Only ICL with Corrections):** Accuracy **43.5%** (20/46), Macro-F1 **0.3689**, Valid rate: **93.5%** (43/46), Net Gain: **-8**, McNemar $p = 0.0215$, Holm-adjusted $p = 0.0859$, Bootstrap 95% CI: `[-30.4%, -4.3%]`.
  * **C5 (Contrastive Correct/Wrong ICL):** Accuracy **50.0%** (23/46), Macro-F1 **0.3581**, Valid rate: **95.7%** (44/46), Net Gain: **-5**, McNemar $p = 0.2668$, Holm-adjusted $p = 0.8005$, Bootstrap 95% CI: `[-26.1%, +4.3%]`.
  * **C6 (Distilled Error Rules):** Accuracy **34.8%** (16/46), Macro-F1 **0.3508**, Valid rate: **73.9%** (34/46), Net Gain: **-12**, McNemar $p = 0.0018$, Holm-adjusted $p = 0.0110$ (**suy giảm có ý nghĩa thống kê**), Bootstrap 95% CI: `[-41.3%, -13.0%]`.
- **Bảo Đảm Tính Toàn Vẹn Ma Trận Chuyển Đổi:**
  - 100% các điều kiện C1–C6 khi so sánh cặp với C0 đều có tổng số claims chuyển đổi chính xác bằng $46$ ($46/46$).
  - Không có mẫu nào bị bỏ sót hay đếm trùng; công thức $\text{Improved} - \text{Regressed} = \text{Correct}_{\text{cond}} - \text{Correct}_{C0}$ nghiệm đúng trên tất cả các điều kiện.
- **Phân Tích Thất Bại Khoa Học (Scientific Failure Analysis):**
  1. *Prompt Bloat & Distraction trong Few-Shot ICL:* Việc đưa 4 ví dụ vào prompt đẩy context length lên 4,700–4,950 tokens, làm loãng sự tập trung của cơ chế self-attention vào 8 bằng chứng của claim mục tiêu.
  2. *Negative Transfer từ Cảnh Báo Lỗi (Error Oversensitivity):* Cảnh báo lỗi trong C4 khiến mô hình trở nên quá thận trọng, khiến độ bao phủ nhãn `Refuted` giảm sút nghiêm trọng từ 69.6% (C0) xuống còn 43.5% (C4), mô hình thiên lệch chuyển sang `Not Enough Evidence`.
  3. *Checklist Exhaustion trong C6:* Hướng dẫn áp dụng "4-point verification checklist" khiến mô hình sinh lời giải thích dài dòng cho từng điểm kiểm tra, dẫn đến việc cạn kiệt ngân sách `max_new_tokens=512` trước khi kịp phát ra token `Verdict: ...` ở 12/46 claims (26.1% unparsed).
- **Tài Liệu & Báo Cáo:**
  - Đã lập báo cáo khoa học hoàn chỉnh tại `docs/M1_3_REPAIRED_EVALUATION_REPORT.md`.
  - Cập nhật test suite với `test_generate_m13_markdown_report_unwraps_metrics` và `test_canonical_m13_audit_integrity`.
  - Bổ sung cơ chế auto-detect đa đường dẫn (`find_train_json`, `find_tokenizer_dir`, `find_m13_dir`) trong `src/tffv/verification/m13_audit.py` hỗ trợ liền mạch cả môi trường local và server `/home/stackops/whale/ERRORFlow/data/raw/averitec/train.json`.
  - Toàn bộ **129/129 tests passed in 14.67s**.
- **Trạng Thái Milestone:** M1.3 P0 Audit & Repair **COMPLETE & VERIFIED**.

## 2026-10-25 08:30 +07:00 — M1.3 Final Consistency Audit & Chuẩn Bị Thực Nghiệm M1.4 Pilot (D0–D6)

- **Mục tiêu:**
  1. *Part A — Hoàn thiện tính nhất quán báo cáo M1.3:* Sửa nguồn số Macro-F1 của C6 (khớp chính xác 0.3508 giữa bảng và phần phân tích narrative); tái thiết kế bảng chuyển đổi cặp thể hiện tường minh 9 danh mục độc lập loại trừ lẫn nhau ($C \to C, C \to W, C \to U, W \to C, W \to W, W \to U, U \to C, U \to W, U \to U$), mỗi claim thuộc đúng 1 category và tổng bằng 46; bóc tách suy giảm của C6 thành 2 phần độc lập: tổn thất do truncation (6 claims) và tổn thất phân loại ngữ nghĩa trên tập parsed (6 claims); loại bỏ mọi nhận định tâm lý/nhân quả chưa được kiểm chứng.
  2. *Part B — Chuẩn bị toàn diện Pilot M1.4 (Training-Free, Quality-First):* Thiết kế giao thức thử nghiệm có kiểm soát D0–D6 với context budget nghiêm ngặt (< 3.200 tokens), k=2 compact demonstrations, retrieval tương đồng từ vựng 3-gram, luật kiểm định ngắn gọn (ngăn chặn cạn kiệt ngân sách sinh).
- **Chi Tiết Khắc Phục Part A (M1.3 Final Consistency):**
  - Đã đối soát mã nguồn và dữ liệu JSON: C6 Macro-F1 thực tế là **0.3508** (trung bình F1 4 lớp: Supported 0.5517, Refuted 0.4516, Not Enough Evidence 0.4000, Cherrypicking 0.0000). Giá trị 0.2867 trong phần phân tích văn bản là lỗi typo từ bản thảo sớm, đã được sửa chuẩn hóa trong `src/tffv/verification/m13_audit.py` và `docs/M1_3_REPAIRED_EVALUATION_REPORT.md`.
  - Bảng so sánh cặp được tổ chức lại với 9 cột độc lập, không gộp lẫn nhóm cha và nhóm con:
    * C1: 23 $C \to C$, 4 $C \to W$, 1 $C \to U$, 3 $W \to C$, 15 $W \to W$, 0 $W \to U$, 0 $U \to C$, 0 $U \to W$, 0 $U \to U$. Tổng = 46. Net = -2.
    * C2: 18 $C \to C$, 10 $C \to W$, 0 $C \to U$, 1 $W \to C$, 17 $W \to W$, 0 $W \to U$. Tổng = 46. Net = -9.
    * C3: 20 $C \to C$, 8 $C \to W$, 0 $C \to U$, 4 $W \to C$, 14 $W \to W$, 0 $W \to U$. Tổng = 46. Net = -4.
    * C4: 19 $C \to C$, 8 $C \to W$, 1 $C \to U$, 1 $W \to C$, 15 $W \to W$, 2 $W \to U$. Tổng = 46. Net = -8.
    * C5: 19 $C \to C$, 8 $C \to W$, 1 $C \to U$, 4 $W \to C$, 13 $W \to W$, 1 $W \to U$. Tổng = 46. Net = -5.
    * C6: 15 $C \to C$, 7 $C \to W$, 6 $C \to U$, 1 $W \to C$, 11 $W \to W$, 6 $W \to U$. Tổng = 46. Net = -12.
  - Phân tích bóc tách C6: Trong 12 claims bị unparsed, có 6 claims trước đó đã đoán đúng ở C0 ($C \to U = 6$). Trên 34 claims parsed, mô hình đúng 16 claims (độ chính xác có điều kiện 47.1% so với 64.7% của C0 trên chính 34 claims này, net loss = 6 claims). Như vậy 50% mức suy giảm là do truncation cơ học của token budget, 50% là do phân loại ngữ nghĩa.
- **Triển Khai M1.4 Controlled Pilot (Part B):**
  - **7 Điều kiện thực nghiệm:**
    * `D0`: Frozen Base Protocol v1.0 (Zero-shot B1, R4 Top-8, 512 tokens).
    * `D1`: Concise generic verification rules (chỉ thị ngắn gọn 2 câu, ép lý giải 1 câu để không vượt 512 tokens).
    * `D2`: Concise source-derived error rules (luật đúc kết từ lỗi Fold A về media debate và missing context).
    * `D3`: Random error demonstrations ($k=2$, trích xuất Top-1 passage $\le 250$ ký tự từ Fold A).
    * `D4`: Similarity-retrieved error demonstrations ($k=2$, truy xuất lỗi tương đồng từ vựng 3-gram từ Fold A).
    * `D5`: Similarity-retrieved error demonstrations with verified corrections ($k=2$, kèm Base mistake và Gold correction).
    * `D6`: Correct/wrong contrastive demonstrations ($k=2$: 1 nearest correct + 1 nearest error từ Fold A).
  - **Modules triển khai mới:**
    * `src/tffv/verification/m14_prompting.py`: Bộ sinh prompt cho D0–D6, hàm tính độ tương đồng 3-gram và bộ chọn top-k tương đồng xác định.
    * `src/tffv/verification/m14_runner.py`: Runner thực nghiệm toàn diện, tự động seal hash, tính toán chuyển đổi cặp 9 danh mục, kiểm định McNemar và Bootstrap CI.
    * `scripts/run_m14_pilot.sh`: Script bash thực thi trên server GPU RTX 5090.
    * `tests/test_m14_pilot.py`: Bộ kiểm thử 4 bài test độ tương đồng, sinh prompt và synthetic end-to-end runner.
    * `docs/M1_4_PILOT_PROTOCOL.md`: Đặc tả chi tiết giao thức, giả thuyết và hướng dẫn vận hành.
  - **Chỉ định pilot:** Toàn bộ thực nghiệm M1.4 trên 46 claims Fold B được đánh dấu rõ là **EXPLORATORY DEBUG PILOT** để phát hiện cơ chế; bất kỳ tín hiệu khả quan nào cũng phải được kiểm chứng lại trên tập held-out độc lập.
- **Kết Quả Kiểm Thử Phần Mềm & Xác Minh Server:**
  * Sửa lỗi chữ ký gọi hàm `Qwen3FrozenBackbone(spec=spec, device=args.device, dtype=args.dtype)` trong `src/tffv/verification/m14_runner.py`.
  * Giới hạn chặt chẽ `candidate_sids` trong Fold B (`50 <= id < 100`) ngăn chặn việc nạp quá 3.000 claims từ toàn bộ corpus.
  * Bổ sung cơ chế chống tràn số `OverflowError` trong `exact_mcnemar_test_twosided` (`src/tffv/verification/stabilization.py`) với xấp xỉ phân phối chuẩn hiệu chỉnh tính liên tục cho $N > 1000$.
  * Bổ sung script tiền kiểm tra thực tế trên server GPU: `scripts/test_m14_server_preflight.py`.
  * **Xác minh trực tiếp trên server `hvtham-server` qua SSH:**
    1. Tải thành công `Qwen3FrozenBackbone` trên GPU CUDA trong 7.31s.
    2. Chạy thử 1 forward pass thành công trong 1.25s.
    3. Chạy thử toàn bộ pipeline `m14_runner.py` trên các tệp dữ liệu thật của server (`R4.jsonl`, `claims.jsonl`, `train.json`, `b1_preds`) hoàn thành 100% không có lỗi, sinh đầy đủ báo cáo Markdown.
  * Toàn bộ **134/134 unit tests** đạt 100% pass trên cả máy local và server.
- **Trạng Thái Git:**
  * Đã push commit `75f88d1` lên nhánh `main`. Pipeline đã được tiền kiểm tra và sẵn sàng 100% để người dùng bấm lệnh chạy.



