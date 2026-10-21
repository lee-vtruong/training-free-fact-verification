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
