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

