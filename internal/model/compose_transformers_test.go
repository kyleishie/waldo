// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openwaldo/waldo/internal/training"
)

func TestTransformersComposeRoundTrip(t *testing.T) {
	compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if compose.Base != nil || compose.Architecture.Transformers == nil {
		t.Fatal("expected explicit random Transformers initialization")
	}
	parameters, err := compose.Stages[0].ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if parameters.BatchSize != 2 || parameters.LearningRate != .001 || parameters.Trainer.Class != "Trainer" {
		t.Fatalf("incorrect resolution: %+v", parameters)
	}
	path, err := ArchiveCompose(t.TempDir(), compose, "smoke")
	if err != nil {
		t.Fatal(err)
	}
	loaded, _, err := LoadCompose(path)
	if err != nil || !reflect.DeepEqual(compose, loaded) {
		t.Fatalf("history changed config or trainer: %v", err)
	}
	before, _ := canonicalHash(compose.Architecture)
	compose.Architecture.Transformers.Package.Version = "5.16.2"
	after, _ := canonicalHash(compose.Architecture)
	if before == after {
		t.Fatal("package pin does not affect architecture interpretation identity")
	}
}

func TestTransformersComposeRejectsUnsafeOrAmbiguousInputs(t *testing.T) {
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{
		"schema":                     strings.Replace(string(data), "schema: 2", "schema: 1", 1),
		"provider":                   strings.Replace(string(data), "provider: huggingface-transformers", "provider: waldo-native", 1),
		"package version":            strings.Replace(string(data), "version: 5.16.1", "version: latest", 1),
		"hash":                       strings.Replace(string(data), "sha256: 2f2d", "sha256: 3g2d", 1),
		"unknown trainer":            strings.Replace(string(data), "logging_steps: 1", "logging_step_typo: 1", 1),
		"output override":            strings.Replace(string(data), "logging_steps: 1", "output_dir: /tmp/escape", 1),
		"publication":                strings.Replace(string(data), "logging_steps: 1", "push_to_hub: true", 1),
		"duplicate batch owner":      strings.Replace(string(data), "steps: 2", "steps: 2\n      batch_size: 8", 1),
		"unimplemented architecture": strings.Replace(string(data), "LlamaForCausalLM", "Qwen3_5ForCausalLM", 1),
		"extra document":             string(data) + "\n---\n{}",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.yaml")
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadCompose(path); err == nil {
				t.Fatal("invalid compose accepted")
			}
		})
	}
}

// Opt-in real end-to-end test: Go compose -> resolved corpus BOM -> Trainer ->
// verified artifacts and durable run evidence -> managed continuation.
func TestTransformersRealLifecycle(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real CPU training")
	}
	compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	builder := Builder{Root: t.TempDir()}
	prepared := preparedFixture(t, compose.Stages[0])
	inspection, err := builder.Compose(context.Background(), "smoke", compose, []PreparedStage{prepared})
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.Runs) != 1 || inspection.Runs[0].State != RunComplete || inspection.Runs[0].Observation.Simulated {
		t.Fatal("expected real complete Transformers run")
	}
	run := inspection.RunBOMs[0]
	if run.Execution.Backend.Name != training.BackendTransformers || !reflect.DeepEqual(run.Execution.Package, &compose.Architecture.Transformers.Package) || run.Parameters.Trainer == nil {
		t.Fatal("missing package/trainer provenance")
	}
	if len(inspection.Runs[0].Observation.Artifacts) != 5 || inspection.Runs[0].Observation.ConsumedTokens != 128 {
		t.Fatalf("unexpected observation: %+v", inspection.Runs[0].Observation)
	}
	for _, artifact := range inspection.Runs[0].Observation.Artifacts {
		if strings.HasSuffix(artifact.Path, "/runtime.json") {
			data, err := os.ReadFile(filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]), artifact.Path))
			if err != nil {
				t.Fatal(err)
			}
			var evidence struct {
				Parameters int `json:"parameters"`
			}
			if err := json.Unmarshal(data, &evidence); err != nil || evidence.Parameters <= 0 {
				t.Fatalf("missing measured parameter count: %v", err)
			}
		}
	}
	prepared.Stage.Name = "continuation"
	prepared.Stage.Parameters.Trainer.Arguments["gradient_accumulation_steps"] = float64(2)
	prepared.Stage.Parameters.Trainer.Arguments["per_device_train_batch_size"] = float64(1)
	continued, err := builder.Train(context.Background(), "smoke", prepared)
	if err != nil {
		t.Fatal(err)
	}
	if continued.RunBOMs[1].Initialization == nil {
		t.Fatal("continuation did not use previous verified weights")
	}
	if continued.Runs[1].Observation.ConsumedTokens != 128 {
		t.Fatalf("gradient accumulation observation: %+v; resolved: %+v", continued.Runs[1].Observation, continued.RunBOMs[1].Parameters)
	}
	if resolved := continued.RunBOMs[1].Parameters; resolved.BatchSize != 2 || resolved.GradientAccumulation != 2 || resolved.TrainerResolution == nil {
		t.Fatalf("accumulation was not recorded as an immutable run fact: %+v", resolved)
	}
	for _, family := range []struct{ name, modelClass, configClass string }{{"qwen2", "Qwen2ForCausalLM", "Qwen2Config"}, {"qwen3", "Qwen3ForCausalLM", "Qwen3Config"}} {
		t.Run(family.name, func(t *testing.T) {
			other, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
			if err != nil {
				t.Fatal(err)
			}
			other.Architecture.Transformers.ModelClass = family.modelClass
			other.Architecture.Transformers.ConfigClass = family.configClass
			other.Architecture.Transformers.Config["model_type"] = family.name
			if _, err := builder.Initialize(family.name, other.Architecture); err != nil {
				t.Fatal(err)
			}
			trained, err := builder.Train(context.Background(), family.name, preparedFixture(t, other.Stages[0]))
			if err != nil {
				t.Fatal(err)
			}
			if trained.Runs[0].Observation.Steps != 2 || trained.Runs[0].Observation.Simulated {
				t.Fatal("expected two real provider training steps")
			}
		})
	}
}

func TestTransformersComposeRejectsNativeExecutionKnobs(t *testing.T) {
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{
		"learning rate":          strings.Replace(string(data), "steps: 2", "steps: 2\n      learning_rate: 0.001", 1),
		"accumulation":           strings.Replace(string(data), "steps: 2", "steps: 2\n      gradient_accumulation_steps: 1", 1),
		"compute precision":      strings.Replace(string(data), "steps: 2", "steps: 2\n      compute_precision: bfloat16", 1),
		"activation checkpoints": strings.Replace(string(data), "steps: 2", "steps: 2\n      activation_checkpointing: true", 1),
		"compile":                strings.Replace(string(data), "steps: 2", "steps: 2\n      compile: true", 1),
		"optimizer":              strings.Replace(string(data), "steps: 2", "steps: 2\n      optimizer: adamw", 1),
		"schedule":               strings.Replace(string(data), "steps: 2", "steps: 2\n      schedule: cosine", 1),
		"warmdown":               strings.Replace(string(data), "steps: 2", "steps: 2\n      warmdown_steps: 1", 1),
		"minimum rate":           strings.Replace(string(data), "steps: 2", "steps: 2\n      minimum_learning_rate_ratio: 0", 1),
		"checkpoint cadence":     strings.Replace(string(data), "steps: 2", "steps: 2\n      checkpoint_every: 0", 1),
		"fp16 and bf16":          strings.Replace(string(data), "logging_steps: 1", "logging_steps: 1\n        fp16: true\n        bf16: true", 1),
		"fp16 without fp32":      strings.Replace(strings.Replace(string(data), "dtype: float32", "dtype: bfloat16", 1), "logging_steps: 1", "logging_steps: 1\n        fp16: true", 1),
		"bf16 with fp16 weights": strings.Replace(strings.Replace(string(data), "dtype: float32", "dtype: float16", 1), "logging_steps: 1", "logging_steps: 1\n        bf16: true", 1),
		"fractional batch":       strings.Replace(string(data), "per_device_train_batch_size: 2", "per_device_train_batch_size: 1.5", 1),
		"missing learning rate":  strings.Replace(string(data), "learning_rate: 0.001", "", 1),
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.yaml")
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadCompose(path); err == nil {
				t.Fatal("ambiguous or unsupported execution knob accepted")
			}
		})
	}
}

func transformersAccumulationCompose(t *testing.T) Compose {
	t.Helper()
	data, err := os.ReadFile("../../docs/examples/transformers-smoke.yaml")
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Replace(strings.Replace(string(data), "per_device_train_batch_size: 2", "per_device_train_batch_size: 1", 1), "gradient_accumulation_steps: 1", "gradient_accumulation_steps: 2\n        bf16: true\n        gradient_checkpointing: true", 1)
	input = strings.Replace(input, "warmup_steps: 0", "warmup_steps: 0.5", 1)
	path := filepath.Join(t.TempDir(), "accumulation.yaml")
	if err := os.WriteFile(path, []byte(input), 0600); err != nil {
		t.Fatal(err)
	}
	compose, _, err := LoadCompose(path)
	if err != nil {
		t.Fatal(err)
	}
	return compose
}

func TestTransformersAccumulationReachesResolvedFactsAndForecast(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	stage := compose.Stages[0]
	if stage.Parameters.BatchSize != 0 || stage.Parameters.GradientAccumulation != 0 || stage.Parameters.ComputePrecision != "" {
		t.Fatalf("normalized compose stores native duplicates: %+v", stage.Parameters)
	}
	resolved, err := stage.ResolveParameters()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.BatchSize != 2 || resolved.GradientAccumulation != 2 || resolved.ComputePrecision != "bfloat16" || !resolved.ActivationCheckpointing {
		t.Fatalf("resolved %+v", resolved)
	}
	forecast, err := ForecastCompose(compose)
	if err != nil {
		t.Fatal(err)
	}
	fitness := forecast.Fitness.Stages[0]
	if len(fitness.Topologies) != 1 || fitness.Topologies[0].GlobalBatchSequences != 2 || fitness.Topologies[0].GlobalMicroBatchSequences != 1 || fitness.Topologies[0].RankLocalMicroBatchSequences != 1 {
		t.Fatalf("forecast topology %+v", fitness.Topologies)
	}
	if fitness.GlobalTokensPerOptimizerStep != 64 || fitness.WarmupPercent != 50 {
		t.Fatalf("forecast stage %+v", fitness)
	}
	// One sequence per micro-batch with recomputation must forecast less
	// activation memory than two sequences without it.
	single, err := ForecastCompose(func() Compose {
		other, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
		if err != nil {
			t.Fatal(err)
		}
		return other
	}())
	if err != nil {
		t.Fatal(err)
	}
	if forecast.Fitness.Memory.ActivationAndLogitsBytes*2 > single.Fitness.Memory.ActivationAndLogitsBytes {
		t.Fatalf("activation forecast ignored accumulation/recomputation: %d vs %d", forecast.Fitness.Memory.ActivationAndLogitsBytes, single.Fitness.Memory.ActivationAndLogitsBytes)
	}
}

type trainerResolvingBackend struct {
	backendFunc
	resolve func(training.ResolvedParameters) (training.ResolvedParameters, error)
}

func (backend trainerResolvingBackend) Descriptor() training.Descriptor {
	descriptor := backend.backendFunc.Descriptor()
	descriptor.Capabilities.ActivationCheckpointing = true
	return descriptor
}

func (backend trainerResolvingBackend) ResolveTrainer(_ context.Context, _ json.RawMessage, parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
	return backend.resolve(parameters)
}

func pinnedTrainerResolution(parameters training.ResolvedParameters, warmup int64) training.TransformersTrainerResolution {
	arguments := map[string]any{"per_device_train_batch_size": 8, "per_device_eval_batch_size": 8, "gradient_accumulation_steps": 1, "learning_rate": 5e-05,
		"weight_decay": 0.0, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-08, "lr_scheduler_type": "linear", "lr_scheduler_kwargs": nil,
		"warmup_steps": 0, "max_grad_norm": 1.0, "optim": "adamw_torch_fused", "optim_args": nil, "bf16": false, "fp16": false,
		"gradient_checkpointing": false, "gradient_checkpointing_kwargs": nil, "use_cache": false, "logging_steps": 500, "full_determinism": false}
	for key, value := range parameters.Trainer.Arguments {
		arguments[key] = value
	}
	return training.TransformersTrainerResolution{Arguments: arguments, WarmupSteps: warmup}
}

func TestTransformersRunBOMPinsTrainerResolutionBeforeExecution(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	var executed training.ResolvedParameters
	backend := trainerResolvingBackend{
		backendFunc: func(_ context.Context, request training.Request) (training.Observation, error) {
			executed = request.Parameters
			return training.Observation{}, fmt.Errorf("stop after capturing the request")
		},
		resolve: func(parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
			return parameters.WithTrainerResolution(pinnedTrainerResolution(parameters, 1))
		},
	}
	root := t.TempDir()
	builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "accumulation", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil {
		t.Fatal("expected captured backend failure")
	}
	inspection, err := Inspect(root, "accumulation")
	if err != nil {
		t.Fatal(err)
	}
	if len(inspection.RunBOMs) != 1 {
		t.Fatalf("runs = %d", len(inspection.RunBOMs))
	}
	recorded := inspection.RunBOMs[0].Parameters
	if recorded.TrainerResolution == nil || recorded.TrainerResolution.WarmupSteps != 1 || recorded.BatchSize != 2 || recorded.GradientAccumulation != 2 || recorded.ComputePrecision != "bfloat16" || !recorded.ActivationCheckpointing {
		t.Fatalf("run BOM parameters %+v", recorded)
	}
	if !reflect.DeepEqual(executed, recorded) {
		t.Fatalf("executed parameters differ from the immutable run BOM:\n%+v\n%+v", executed, recorded)
	}
}

func TestTransformersTrainerConflictFailsBeforeRunCreation(t *testing.T) {
	compose := transformersAccumulationCompose(t)
	compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
	for name, resolve := range map[string]func(training.ResolvedParameters) (training.ResolvedParameters, error){
		"host rejects": func(training.ResolvedParameters) (training.ResolvedParameters, error) {
			return training.ResolvedParameters{}, fmt.Errorf("mixed precision requires a GPU in this adapter")
		},
		"pinned drift": func(parameters training.ResolvedParameters) (training.ResolvedParameters, error) {
			resolution := pinnedTrainerResolution(parameters, 1)
			resolution.Arguments["gradient_accumulation_steps"] = 1
			return parameters.WithTrainerResolution(resolution)
		},
	} {
		t.Run(name, func(t *testing.T) {
			called := false
			backend := trainerResolvingBackend{
				backendFunc: func(context.Context, training.Request) (training.Observation, error) {
					called = true
					return training.Observation{}, nil
				},
				resolve: resolve,
			}
			root := t.TempDir()
			builder := Builder{Root: root, Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
				return testSelection(backend), nil
			})}
			if _, err := builder.Compose(context.Background(), "conflict", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil || !strings.Contains(err.Error(), "resolve Trainer settings") {
				t.Fatalf("conflict: %v", err)
			}
			if called {
				t.Fatal("backend executed despite Trainer conflict")
			}
			inspection, err := Inspect(root, "conflict")
			if err == nil && len(inspection.Model.Runs) != 0 {
				t.Fatal("run created before Trainer resolution failed")
			}
		})
	}
	compose.Stages[0].Parameters.Trainer.Arguments["gradient_checkpointing"] = false
	backend := backendFunc(func(context.Context, training.Request) (training.Observation, error) {
		return training.Observation{}, nil
	})
	builder := Builder{Root: t.TempDir(), Resolver: training.ResolverFunc(func(context.Context, training.ResolveRequest) (training.Selection, error) {
		return testSelection(backend), nil
	})}
	if _, err := builder.Compose(context.Background(), "unresolvable", compose, []PreparedStage{preparedFixture(t, compose.Stages[0])}); err == nil || !strings.Contains(err.Error(), "cannot resolve Transformers Trainer settings") {
		t.Fatalf("backend without Trainer resolution: %v", err)
	}
}

// Opt-in real checks that the pinned Trainer resolution gates run creation
// and that resolved recomputation is what executes.
func TestTransformersRealTrainerResolution(t *testing.T) {
	if os.Getenv("WALDO_TRANSFORMERS_PYTHON") == "" || os.Getenv("WALDO_TRANSFORMERS_WHEEL") == "" {
		t.Skip("set WALDO_TRANSFORMERS_PYTHON and WALDO_TRANSFORMERS_WHEEL for real Trainer resolution")
	}
	t.Setenv("WALDO_TRANSFORMERS_DEVICE", "cpu")
	load := func(arguments map[string]any) (Compose, PreparedStage) {
		compose, _, err := LoadCompose("../../docs/examples/transformers-smoke.yaml")
		if err != nil {
			t.Fatal(err)
		}
		compose.Stages[0].Corpora = NewCorpusSelections([]string{"example"})
		for key, value := range arguments {
			compose.Stages[0].Parameters.Trainer.Arguments[key] = value
		}
		return compose, preparedFixture(t, compose.Stages[0])
	}
	t.Run("host rejects before run", func(t *testing.T) {
		compose, prepared := load(map[string]any{"fp16": true})
		root := t.TempDir()
		_, err := (Builder{Root: root}).Compose(context.Background(), "cpu-fp16", compose, []PreparedStage{prepared})
		if err == nil || !strings.Contains(err.Error(), "resolve Trainer settings") || !strings.Contains(err.Error(), "mixed precision requires a GPU") {
			t.Fatalf("expected pre-run Trainer rejection: %v", err)
		}
		if entries, err := filepath.Glob(filepath.Join(root, "*", "runs", "*")); err != nil || len(entries) != 0 {
			t.Fatalf("run state created before Trainer rejection: %v %v", entries, err)
		}
	})
	t.Run("recomputation", func(t *testing.T) {
		compose, prepared := load(map[string]any{"gradient_checkpointing": true, "gradient_accumulation_steps": 2, "per_device_train_batch_size": 1, "warmup_steps": 0.5})
		inspection, err := (Builder{Root: t.TempDir()}).Compose(context.Background(), "recompute", compose, []PreparedStage{prepared})
		if err != nil {
			t.Fatal(err)
		}
		parameters := inspection.RunBOMs[0].Parameters
		if !parameters.ActivationCheckpointing || parameters.GradientAccumulation != 2 || parameters.TrainerResolution == nil || parameters.TrainerResolution.WarmupSteps != 1 || parameters.TrainerResolution.Arguments["optim"] != "adamw_torch_fused" {
			t.Fatalf("run BOM parameters %+v", parameters)
		}
		var evidence struct {
			Execution struct {
				Checkpointing bool  `json:"gradient_checkpointing"`
				MicroBatches  int64 `json:"micro_batches"`
			} `json:"execution"`
		}
		for _, artifact := range inspection.Runs[0].Observation.Artifacts {
			if strings.HasSuffix(artifact.Path, "/runtime.json") {
				data, err := os.ReadFile(filepath.Join(inspection.Path, "runs", runDirectoryName(inspection.Model.Runs[0]), artifact.Path))
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, &evidence); err != nil {
					t.Fatal(err)
				}
			}
		}
		if !evidence.Execution.Checkpointing || evidence.Execution.MicroBatches != 4 {
			t.Fatalf("executed %+v", evidence.Execution)
		}
	})
}
