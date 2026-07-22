import assert from "node:assert/strict";


const base = "http://127.0.0.1:9000";
// Realtime Database emulator instances use the project's default RTDB name,
// not the bare project ID. Using `demo-localfit` here silently activated a
// second, rules-free namespace and made every authorization assertion
// meaningless.
const namespace = "demo-localfit-default-rtdb";

async function request(path, method, body) {
  const response = await fetch(`${base}/${path}.json?ns=${namespace}`, {
    method,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // A denied emulator response may not have a JSON body.
  }
  return { ok: response.ok, status: response.status, payload };
}

const valid = {
  ram_gb: 24,
  vram_gb: 24,
  unified_memory: true,
  model_installed: "model-7B-Q4.gguf",
  model_repo_id: "org/model-7B-GGUF",
  model_size_bytes: 4 * 1024 ** 3,
  engine: "ollama",
  benchmark_version: 4,
  recorded_at: "2026-07-20T00:00:00+00:00",
  tokens_per_sec: 20.5,
  sample_count: 3,
  tokens_per_sec_min: 19.5,
  tokens_per_sec_max: 21.5,
  runtime_profile: "balanced",
  context_length: 4096,
  gpu_offload_percent: 100,
  cpu_threads: 8,
  num_batch: 512,
};

const created = await request("telemetry", "POST", valid);
assert.equal(created.ok, true, `valid schema 4 event was rejected (${created.status})`);
assert.equal(typeof created.payload?.name, "string");

for (const benchmarkVersion of [1, 2, 3, 4]) {
  const legacy = {
    ...valid,
    benchmark_version: benchmarkVersion,
  };
  if (benchmarkVersion < 3) {
    legacy.os = "test-os";
    legacy.cpu = "test-cpu";
    legacy.gpu = "test-gpu";
  }
  const legacyCreated = await request("telemetry", "POST", legacy);
  assert.equal(
    legacyCreated.ok,
    true,
    `legacy schema ${benchmarkVersion} event was rejected (${legacyCreated.status})`,
  );
}

const validV5 = {
  ...valid,
  benchmark_version: 5,
  model_filename: "model-7B-Q4.gguf",
  model_digest: "a".repeat(64),
  parameter_count_b: 7,
  active_parameter_count_b: 7,
  quant_bits: 4,
  engine_version: "0.12.0",
  client_version: "0.1.0",
  runtime_profile: "explicit_ollama_options",
  context_length: 4096,
  gpu_offload_percent: 100,
  cpu_threads: 8,
  num_batch: 512,
  sample_count: 3,
  tokens_per_sec_min: 19.5,
  tokens_per_sec_max: 21.5,
  quality_pack_id: "localfit-smoke",
  quality_pack_version: "1",
  quality_correct: 4,
  quality_total: 5,
  quality_accuracy: 0.8,
};
const v5Created = await request("telemetry", "POST", validV5);
assert.equal(v5Created.ok, true, `valid schema 5 event was rejected (${v5Created.status})`);

const taggedFilenameV5 = await request("telemetry", "POST", {
  ...validV5,
  model_filename: "model:latest",
});
assert.equal(taggedFilenameV5.ok, true, "schema 5 rejected a normal Ollama model tag");

const missingV5Metadata = await request("telemetry", "POST", {
  ...validV5,
  client_version: undefined,
});
assert.equal(missingV5Metadata.ok, false, "schema 5 accepted missing direct metadata");

const invalidV5Runtime = await request("telemetry", "POST", {
  ...validV5,
  cpu_threads: 0,
});
assert.equal(invalidV5Runtime.ok, false, "schema 5 accepted invalid runtime metadata");

const fractionalV5Runtime = await request("telemetry", "POST", {
  ...validV5,
  cpu_threads: 8.5,
});
assert.equal(fractionalV5Runtime.ok, false, "schema 5 accepted fractional runtime metadata");

const invalidV5Samples = await request("telemetry", "POST", {
  ...validV5,
  sample_count: 2,
});
assert.equal(invalidV5Samples.ok, false, "schema 5 accepted fewer than three samples");

const invalidV5Filename = await request("telemetry", "POST", {
  ...validV5,
  model_filename: "C:\\private\\model.gguf",
});
assert.equal(invalidV5Filename.ok, false, "schema 5 accepted a local model path");

const invalidV5Digest = await request("telemetry", "POST", {
  ...validV5,
  model_digest: "A".repeat(64),
});
assert.equal(invalidV5Digest.ok, false, "schema 5 accepted a non-normalized digest");

const nonHexV5Digest = await request("telemetry", "POST", {
  ...validV5,
  model_digest: "g".repeat(64),
});
assert.equal(nonHexV5Digest.ok, false, "schema 5 accepted a non-hex digest");

const invalidV5Quality = await request("telemetry", "POST", {
  ...validV5,
  quality_accuracy: 0.1,
});
assert.equal(invalidV5Quality.ok, false, "schema 5 accepted an inconsistent quality ratio");

const partialV5Quality = await request("telemetry", "POST", {
  ...validV5,
  quality_pack_id: undefined,
});
assert.equal(partialV5Quality.ok, false, "schema 5 accepted partial quality metadata");

const fractionalVersion = await request("telemetry", "POST", {
  ...valid,
  benchmark_version: 4.5,
});
assert.equal(fractionalVersion.ok, false, "schema accepted a fractional benchmark version");

const rawName = await request("telemetry", "POST", { ...valid, cpu: "Apple M5" });
assert.equal(rawName.ok, false, "schema 4 unexpectedly accepted a raw CPU name");

const unknown = await request("telemetry", "POST", { ...valid, unexpected: "value" });
assert.equal(unknown.ok, false, "telemetry unexpectedly accepted an unknown field");

const outOfRange = await request("telemetry", "POST", { ...valid, tokens_per_sec: 5000 });
assert.equal(outOfRange.ok, false, "telemetry unexpectedly accepted an out-of-range speed");

const overwrite = await request(`telemetry/${created.payload.name}`, "PUT", {
  ...valid,
  tokens_per_sec: 99,
});
assert.equal(overwrite.ok, false, "append-only telemetry unexpectedly allowed an overwrite");

const unrelated = await request("unrelated", "POST", { value: true });
assert.equal(unrelated.ok, false, "default-deny rule unexpectedly allowed another path");

const readable = await request("telemetry", "GET");
assert.equal(readable.ok, true, "public retraining read unexpectedly failed");

console.log("Firebase rules scenarios passed.");
