# M1.1 — Lệnh để người dùng chạy trên terminal server

Codex chỉ code, test local, hướng dẫn và phân tích output người dùng gửi lại.
**Không có lệnh SSH, push/pull hay GPU job nào được Codex tự chạy.**
Các lệnh dưới đây do người dùng thực thi, sau khi đã chuyển code mới lên server.
Không sửa repo ERRORFlow. Không dùng dữ liệu QA/gold làm corpus độc lập.

## 0. Đồng bộ code: local → GitHub → server

Luôn gửi kèm lệnh push và pull mỗi lần bàn giao code. Người dùng tự thực hiện.
Tại lần kiểm tra 2026-10-20, local chưa có commit, branch `master`, chưa có
remote. Ba lệnh setup đầu dưới đây chỉ dành cho trạng thái ban đầu này.

**Windows PowerShell — máy local:**

```powershell
cd C:\Users\ASUS\Downloads\training-free-fact-verification
git branch -m main
git remote add origin https://github.com/lee-vtruong/training-free-fact-verification.git
git status --short --branch
git remote -v
python scripts/run_tests.py
git add .gitignore .env.example AGENTS.md README.md pyproject.toml configs src scripts tests docs reports outputs/.gitkeep
git diff --cached --stat
git diff --cached --check
git commit -m "Implement M1.1 retrieval pipelines and reproducible server commands"
git push -u origin main
git rev-parse --short HEAD
```

Chỉ tiếp tục commit/push khi tests pass và staged files đúng phạm vi. Không
stage datasets, checkpoints, `.env`, caches hoặc generated outputs. GitHub repo
phải tồn tại và tài khoản có quyền push. Chưa kiểm tra lịch sử GitHub trong
phiên này; nếu push bị từ chối vì remote có commits khác, gửi lỗi để xử lý theo
lịch sử thực tế, không force-push.

Nếu Windows báo WinError 5 với `pytest-of-ASUS` hoặc `.pytest_cache`, dùng
`python scripts/run_tests.py` như trên. Runner tạo temporary directory riêng
cho mỗi lần chạy và tắt disk cache; không cần Administrator, sửa ACL hoặc xóa
temporary/cache directories cũ. Đây khác với Git `safe.directory`, vốn chỉ giải
quyết việc Git tin cậy repository, không thay đổi quyền filesystem.

**Linux terminal — server đã clone repo:**

```bash
cd ~/whale/training-free-fact-verification
git status --short --branch
git remote -v
git switch main
git pull --ff-only origin main
git rev-parse --short HEAD
```

Nếu working tree server có thay đổi hoặc lệnh Git báo lỗi, dừng trước khi chạy
experiment và gửi output; không reset/stash tự động. Đối chiếu commit local và
server sau pull.

**Nếu server chưa clone repo**, dùng thay cho block pull:

```bash
mkdir -p ~/whale
cd ~/whale
git clone https://github.com/lee-vtruong/training-free-fact-verification.git
cd training-free-fact-verification
git status --short --branch
git remote -v
git rev-parse --short HEAD
```

Sau khi đồng bộ thành công mới chạy bước 1 bên dưới. Những lần sau bỏ setup
branch/remote; vẫn kiểm tra branch/remote, stage files cụ thể, commit/push rồi
pull trên server. Các lệnh này không thao tác vào ERRORFlow.

## 1. Chạy trước: kiểm tra hạ tầng, chưa inference

Trong terminal Linux của server:

```bash
cd ~/whale/training-free-fact-verification
git status --short --branch
git remote -v
python3 scripts/m11_resources.py \
  --old-root ~/whale/ERRORFlow \
  --model-root ~/whale/cache/models \
  --model-root ~/.cache/huggingface/hub \
  --output outputs/m11-resources-001
cat outputs/m11-resources-001/resources.json
```

`~/whale/...` là đường dẫn quy ước, chưa được xác minh. Nếu repo hoặc model
cache nằm nơi khác, thay bằng path thực tế. Nếu thư mục output đã tồn tại, dùng
tên mới; không xóa/overwrite output cũ. Scan giới hạn depth/file count, bỏ qua
old outputs và thư mục ẩn; thiếu path trong scan không chứng minh nó không tồn tại.

**Gửi lại `resources.json` để xác nhận paths và dependencies trước.** Nếu cần
kiểm tra corpus/schema riêng, gửi tên file, field names, kích thước và provenance;
không đưa secrets hoặc labels vào input retrieval. Không tự nhận model đã có
chỉ vì tồn tại một thư mục tên tương tự.

### Follow-up khi inventory chỉ có annotations và thiếu model retrieval

Sau khi push/pull script mới theo bước 0, chạy probe sau. Script chỉ đọc schema
source train (không in primitive values), tìm candidate paths cả archive formats,
và lấy package metadata từ các venv cũ bằng Python `-I -B`. Nó không import
torch/models, không sửa venv cũ, không đọc dev gold để thiết kế retrieval, và
không chuyển QA thành corpus. Paths tìm được vẫn cần review provenance.

```bash
PROBE_OUT="outputs/m11-input-probe-$(date +%Y%m%d-%H%M%S)"
python scripts/m11_probe_inputs.py \
  --old-root "$HOME/whale/ERRORFlow" \
  --search-root "$HOME/whale/ERRORFlow/data" \
  --search-root "$HOME/whale/cache" \
  --env-python "$HOME/whale/ERRORFlow/.venv/bin/python" \
  --env-python "$HOME/whale/GraphCURE/.venv/bin/python" \
  --output "$PROBE_OUT" &&
cat "$PROBE_OUT/input_probe.json"
```

Gửi `input_probe.json`. Chưa tải checkpoint/corpus, cài GPU stack hoặc thay
Qwen retrieval bằng model khác chỉ vì một model khác đã có sẵn.

### Sau probe ngày 2026-10-21: chuẩn hóa source claims

User probe đã xác nhận raw source train có 3068 records và SHA256 bên dưới.
Sau khi đồng bộ code mới, trong project venv hiện tại chạy:

```bash
CLAIMS_OUT="outputs/m11-source-claims-$(date +%Y%m%d-%H%M%S)"
python -m tffv.retrieval.prepare_claims \
  --train-json "$HOME/whale/ERRORFlow/data/raw/averitec/train.json" \
  --expected-sha256 ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c \
  --expected-records 3068 \
  --output "$CLAIMS_OUT" &&
cat "$CLAIMS_OUT/claims_manifest.json"
```

Exporter chỉ dùng `claim`, giữ toàn bộ source rows theo đúng thứ tự, dùng ID
`train:<array ordinal>` và lưu mapping/hash. Không dùng nhãn/QA/justification/
annotation URLs, không deduplicate claims và không export dev/test. Đây là
claims manifest riêng, chưa phải full retrieval data manifest và chưa có corpus.

### Kiểm tra runtime cũ read-only, chưa load checkpoint

Probe tìm thấy GraphCURE venv: Python 3.11.15, torch 2.13.0, transformers 5.15.0.
Transformers này ngoài dải `<5` đang khai báo trong project; chưa coi là runtime
đã hỗ trợ, không cài/downgrade vào môi trường cũ. Lệnh sau chỉ import libraries
và nhân hai ma trận 8×8 trên GPU, không model inference hoặc optimizer:

```bash
OLD_PY="$HOME/whale/GraphCURE/.venv/bin/python"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$OLD_PY" -B - <<'PY'
import json
import torch
import transformers
print(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__,
                  "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in the inspected old environment")
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}))
with torch.inference_mode():
    x = torch.ones((8, 8), device="cuda")
    print(json.dumps({"small_matmul_sum": (x @ x).sum().item()}))
PY
```

Gửi cả claims manifest và runtime output. CUDA health thành công vẫn không chứng
minh Qwen adapter tương thích transformers 5. Cần model-load smoke riêng trước
khi thay đổi range khai báo hay dùng môi trường đó cho benchmark.

### Knowledge store chưa tải: chỉ lấy metadata và dung lượng

Người dùng đã xác nhận chưa tải knowledge store. Sau khi đồng bộ code mới, chạy
lệnh sau trong project venv; không cần cài huggingface-hub, torch hay transformer:

```bash
PLAN_OUT="outputs/m11-corpus-plan-$(date +%Y%m%d-%H%M%S)"
python scripts/m11_corpus_plan.py --output "$PLAN_OUT" &&
cat "$PLAN_OUT/acquisition_plan.json"
```

Script chỉ đọc public API JSON tối đa 4 MiB để pin revision/size/SHA256 của ba
archive train. Không tải corpus/model hoặc giải nén. Nếu API timeout/error hoặc
không có LFS hashes, gửi nguyên lỗi; không tự thay endpoint hoặc tải cả repo.
Gửi acquisition_plan.json để lập phương án bounded archive inspection/storage.
`compressed_only_fits=true` chưa nghĩa là đủ chỗ cho archive + extraction +
passage/vector cache. Các chi phí chưa đo giữ null và plan chưa cho bulk download.

### Đã có acquisition plan: kiểm tra ZIP directory, không tải toàn archive

Sau push/pull code mới, dùng đúng plan người dùng đã tạo. Chỉ lấy range payload
tối đa 4 MiB/archive, tổng tối đa 12 MiB cho ba archive train; không giải nén ra
disk. Tùy chọn sample chỉ xem field/type của dòng đầu một source member, không
in nội dung text/annotations. HTTP server không hỗ trợ Range sẽ báo lỗi, không
fallback tải 20 GB.

```bash
ZIP_OUT="outputs/m11-zip-inspection-$(date +%Y%m%d-%H%M%S)"
python -m tffv.retrieval.remote_zip \
  --plan outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json \
  --sample-first-record \
  --output "$ZIP_OUT" &&
cat "$ZIP_OUT/zip_summary.json"
```

Gửi `zip_summary.json`. Nếu lỗi, gửi failure.json/traceback; không tăng cap hoặc
chạy download thay thế. Các `*.inventory.json` lưu full member lists nếu cần
phân tích tiếp. Tổng declared extraction bytes không bao gồm filesystem overhead,
corpus/vector cache hoặc headroom. Đây chưa phải chứng nhận đủ chỗ/không leakage.

### Với 125.03 GiB trống: lấy complete store của claims 0/1/2 để smoke

Không tải/giải nén toàn bộ: ZIP directory đã báo 187.08 GiB giải nén, hoặc
246.25 GiB khi giữ cả ZIP. Bước sau chỉ lấy đủ raw store của 3 source claims,
không chọn method trên subset này. Giới hạn **128 MiB range payload tổng**,
**512 MiB raw tổng**, **256 MiB/member**; không model download/inference/GPU.

Sau push/pull code mới:

```bash
SUBSET_OUT="outputs/m11-source-store-smoke-$(date +%Y%m%d-%H%M%S)"
python -m tffv.retrieval.acquire_subset \
  --plan outputs/m11-corpus-plan-20261021-042129/acquisition_plan.json \
  --claim-ids 0 1 2 \
  --output "$SUBSET_OUT" --execute &&
cat "$SUBSET_OUT/subset_manifest.json"
```

Bỏ `--execute` nếu chỉ muốn xem preflight; dùng output name khác cho lần execute
tiếp theo. Script kiểm tra metadata/caps trước transfer, ghi raw member mới,
kiểm CRC đầy đủ và SHA256 từng file. Không xác minh SHA256 toàn ZIP vì không tải
hết ZIP. Profile không in raw text; corpus_admitted vẫn false đến khi review
lineage/filtering/mapping. `*.part` khi failure không phải complete artifact.
Gửi manifest hoặc failure.json; không tự tăng cap hay xóa dữ liệu cũ.

### Complete stores đã tải: audit offline, không tải lại

Acquisition `outputs/m11-source-store-smoke-20261021-053505` đã thành công.
Initial profile của store 1 vượt 8 MiB/record; không cần tải lại. Chạy audit mới
với encoded-record cap 64 MiB được ghi trong config. Không tăng model context,
không truncate/skip data; nếu vẫn vượt cap, lệnh dừng và lưu failure.

```bash
AUDIT_OUT="outputs/m11-source-store-audit-$(date +%Y%m%d-%H%M%S)"
python -m tffv.retrieval.audit_subset \
  --subset-manifest outputs/m11-source-store-smoke-20261021-053505/subset_manifest.json \
  --max-record-mib 64 \
  --output "$AUDIT_OUT" &&
cat "$AUDIT_OUT/audit_summary.json"
```

Sau khi audit thành công, xem review queue:

```bash
cat "$AUDIT_OUT/review_queue.json"
```

Gửi summary và review queue; không cần paste lại range_reads hoặc raw stores.
Queue chứa nguồn/excerpts của source train, không phải target gold labels.
Flags chỉ là nghi vấn cần xem evidence. Chưa tự drop các nguồn đó hoặc admit
corpus, chưa model inference. Các raw file acquisition ban đầu không bị sửa.

## 2. Environment và synthetic smoke, sau khi kiểm tra tài nguyên

Nếu cần môi trường mới, chỉ tạo trong repo mới. Không sửa ERRORFlow/GraphCURE venv.
Việc cài package có thể tải dependency lớn: kiểm tra dung lượng trước.

```bash
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -e '.[dev]'
python scripts/run_tests.py
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/m11_smoke.py --output outputs/m11-synthetic-smoke-001
cat outputs/m11-synthetic-smoke-001/evaluation/comparison.md
```

Smoke này dùng test doubles, **không phải Qwen hay AVeriTeC benchmark**. Không
có checkpoint download. Không dùng synthetic metrics để chọn pipeline.
Nếu môi trường độc lập đã tương thích, có thể dùng nó thay vì tạo lại.

Synthetic smoke/tests chỉ cần standard library, numpy và pytest; chưa cần
PyTorch/CUDA/transformers. **Không cài `.[dev,retrieval]` trong bước này.**
Dependencies GPU chỉ cài riêng sau khi xác minh dung lượng, GPU/driver và các
environment có sẵn. Không tự chọn bản torch/CUDA mới nhất hoặc cài lại một
stack đã có trước khi kiểm tra khả năng tái sử dụng read-only.

### Nếu cài đặt báo `No space left on device`

Dừng chuỗi cài/test/smoke. Thiếu pytest sau đó là hậu quả của việc cài bị ngắt;
file comparison không tồn tại vì chưa có smoke thành công. Kiểm tra trên terminal
server, không xóa gì:

```bash
df -h . /tmp "$HOME"
df -i . /tmp "$HOME"
du -sh .venv "$HOME/.cache/pip" 2>/dev/null
for env_python in \
  "$HOME/whale/ERRORFlow/.venv/bin/python" \
  "$HOME/whale/GraphCURE/.venv/bin/python"
do
  if [ -x "$env_python" ]; then
    "$env_python" -c 'import sys, importlib.metadata as m; print(sys.executable); print({n: next((d.version for d in m.distributions() if d.metadata["Name"].lower() == n), None) for n in ["torch", "transformers", "numpy", "pytest"]})'
  fi
done
```

Gửi output để quyết định môi trường/path phù hợp. `--no-cache-dir` tránh tạo
pip cache mới nhưng không giải quyết filesystem đã đầy. Không xóa checkpoints,
old environments/caches hoặc `.venv` đang có để giải phóng chỗ khi chưa xác định
phạm vi và được người dùng đồng ý. `.venv` mới cài dở chưa được coi là hợp lệ.

Chỉ sau khi đủ dung lượng hoặc đã chọn environment tương thích mới chạy block
cài nhẹ bên trên. Không tiếp tục benchmark khi tests/smoke còn lỗi.

## 3. Chuẩn bị inputs và snapshots đã được kiểm chứng

Xem `docs/M11_RETRIEVAL_PROTOCOL.md` cho schema. Chỉ cấu hình các paths thực tế:

```bash
export TFFV_RETRIEVAL_CLAIMS=/absolute/path/to/label_free_claims.jsonl
export TFFV_RETRIEVAL_CORPUS=/absolute/path/to/independent_documents.jsonl
export TFFV_RETRIEVAL_DATA_MANIFEST=/absolute/path/to/reviewed_data_manifest.json
export TFFV_EMBED_MODEL=/absolute/path/to/Qwen3-Embedding-0.6B/snapshot
export TFFV_RERANK_MODEL=/absolute/path/to/Qwen3-Reranker-0.6B/snapshot
export TFFV_EMBED_REVISION=ACTUAL_40_CHARACTER_UPSTREAM_COMMIT
export TFFV_RERANK_REVISION=ACTUAL_40_CHARACTER_UPSTREAM_COMMIT
```

Các giá trị trên là placeholders cần thay; không tự đoán model revision.
Template data manifest tại `configs/retrieval_data_manifest.template.json`.
Điền hashes/counts và xác nhận từng review flag bằng evidence thực tế.
Có thể lấy SHA256 của hai file canonical bằng:

```bash
sha256sum "$TFFV_RETRIEVAL_CLAIMS" "$TFFV_RETRIEVAL_CORPUS"
```

Không có automatic adapter cho raw AVeriTeC/corpus chưa được xem schema.
Sau khi biết raw fields/provenance, mới viết conversion có version/hash;
không rename gold QA thành `web_document` để vượt validation.

Đọc model bytes để fingerprint, chưa inference:

```bash
python -m tffv.retrieval.cli fingerprint \
  --model-dir "$TFFV_EMBED_MODEL" --output outputs/m11-embed-fingerprint-001
python -m tffv.retrieval.cli fingerprint \
  --model-dir "$TFFV_RERANK_MODEL" --output outputs/m11-rerank-fingerprint-001
export TFFV_EMBED_SHA256="$(python -c 'import json; print(json.load(open("outputs/m11-embed-fingerprint-001/snapshot.json"))["snapshot_sha256"])')"
export TFFV_RERANK_SHA256="$(python -c 'import json; print(json.load(open("outputs/m11-rerank-fingerprint-001/snapshot.json"))["snapshot_sha256"])')"
```

Fingerprint đọc toàn bộ weights, có I/O nhưng không chạy GPU. Không tự tải model
nếu thiếu; báo lại paths và dung lượng để quyết định bước tiếp theo.

## 4. Preflight và real-model smoke có giới hạn

Template đọc các biến môi trường bên trên; không tự load `.env`.

```bash
python -m tffv.retrieval.cli benchmark \
  --config configs/retrieval_m11.template.json \
  --max-claims 3 --max-passages 256
```

Lệnh này chỉ validate/hiện workload, chưa load model hoặc inference. Smoke tối đa
256 passage embeddings + 3 query embeddings + 120 cặp reranking; corpus nhỏ
hơn sẽ dùng workload nhỏ hơn. Chưa có ước lượng runtime/VRAM thực nghiệm.
Chỉ chạy sau khi GPU/dung lượng và input đã được xác nhận:

```bash
mkdir -p outputs/logs
set -o pipefail
python -m tffv.retrieval.cli benchmark \
  --config configs/retrieval_m11.template.json \
  --max-claims 3 --max-passages 256 \
  --output outputs/m11-real-smoke-001 --execute \
  2>&1 | tee outputs/logs/m11-real-smoke-001.log
python -m tffv.retrieval.cli evaluate \
  --run outputs/m11-real-smoke-001 \
  --output outputs/m11-real-smoke-eval-001
cat outputs/m11-real-smoke-eval-001/comparison.md
```

Smoke subset không đại diện toàn bộ corpus và không được dùng để chọn method.
Gửi manifest, summary và log về trước khi benchmark lớn. Nếu có failure, gửi
`failure.json` và log; không xóa artifact hay tự giảm context để che lỗi.

## 5. Full benchmark — chỉ sau khi người dùng quyết định chạy workload

Trước hết in workload toàn bộ, không có `--execute`:

```bash
python -m tffv.retrieval.cli benchmark \
  --config configs/retrieval_m11.template.json
```

Với P passages, N claims: encode P documents và N queries; rerank tối đa
N × min(40,P) cặp. Vector cache float32 chiếm khoảng P × 1024 × 4 bytes, chưa
tính corpus, BM25 postings và output. Full corpus có index cache khác smoke
subset; các full rerun tương thích sẽ tái sử dụng cùng index đã seal.

Sau khi đã xem workload và xác nhận tài nguyên, người dùng có thể chạy trong tmux:

```bash
tmux new -s tf-retrieval-m11
cd ~/whale/training-free-fact-verification
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Export lại các biến inputs/models nếu đây là terminal mới.
mkdir -p outputs/logs
set -o pipefail
python -m tffv.retrieval.cli benchmark \
  --config configs/retrieval_m11.template.json \
  --output outputs/m11-full-001 --execute \
  2>&1 | tee outputs/logs/m11-full-001.log
```

## 6. Đánh giá cache đã seal

Không có relevance annotations hợp lệ:

```bash
python -m tffv.retrieval.cli evaluate \
  --run outputs/m11-full-001 --output outputs/m11-full-eval-001
cat outputs/m11-full-eval-001/comparison.md
```

Recall/MRR/nDCG sẽ là `not measured`, không phải zero. Nếu có qrels passage-level
đã được kiểm chứng và mapping đúng passage index, dùng một output mới:

```bash
python -m tffv.retrieval.cli evaluate \
  --run outputs/m11-full-001 --output outputs/m11-full-qrels-eval-001 \
  --qrels /absolute/path/to/validated_qrels.jsonl \
  --qrels-manifest /absolute/path/to/validated_qrels_manifest.json
```

Qrels JSONL: đúng ba fields `sample_id`, `passage_id`, `relevance` (grade 0–4).
Manifest yêu cầu `validated: true`, `no_inferred_labels: true`,
`annotation_provenance`, `relevance_unit: "passage"`, `synthetic: false`,
`unjudged_policy: "unjudged_as_nonrelevant"` và hashes `claims_sha256`,
`passages_sha256`, `qrels_sha256` trỏ tới đúng artifacts của run.
Không tự chuyển gold verdict, QA hoặc URL-match thành relevance grades.

## 7. Gửi lại để phân tích

- Git branch/commit, commands và model/corpus provenance.
- `resources.json`, run `manifest.json`, `workload.json`, `summary.json`.
- Evaluation `comparison.md` và `comparison.json`.
- `leakage_scan.json`, relevant logs/failures.
- Một số entries numerical/temporal/conflicting trong `manual_review_queue.jsonl`,
  kèm evidence sources và nhận xét thủ công nếu có.

Không cần gửi toàn bộ embedding matrix. Chưa chọn R4 hay bất kỳ method nào làm
baseline M1.2. Chưa chạy Qwen3-8B và chưa tạo Historical Error Memory.
# Source claim validation failure: read-only diagnostic

User inspection confirmed one empty string at train:1948. Export remains strict
by default. To explicitly quarantine empty strings only, add
`--invalid-claim-policy quarantine_empty` to the normal `--output` export command
(not `--inspect`). This writes claims, a full source ID mapping and a hashed
quarantine ledger. Expected counts: source 3068, exported 3067, quarantined 1.
Original IDs and split membership remain unchanged. This is a valid-claim subset,
not full official-split coverage. It does not admit evidence or clear leakage.

If hash/count pass but claim validation fails, inspect all affected original IDs
without exporting, skipping rows, reading annotation values, or repairing text:

```bash
python -m tffv.retrieval.prepare_claims \
  --train-json /home/stackops/whale/ERRORFlow/data/raw/averitec/train.json \
  --expected-sha256 ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c \
  --expected-records 3068 --inspect
```

Send the diagnostic JSON. Invalid claims are reported as IDs/reasons, never filled
from justification or QA. Ordinary export still rejects any invalid claim before
writing artifacts. The first three valid claim fields are printed for source
context review; invalid ones are null. Do not infer a missing claim from these.

