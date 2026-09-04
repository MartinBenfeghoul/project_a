from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import lm_eval_script
import meta_learning as meta
from cache import CompressedCache, CompressedCacheConfig, MLPValueCacheConfig
from cache.backends.mlp_values import MLPValueCache
from model.meta_learning import LearnedInit
from model.mlp import MLP


def _cfg(**overrides):
    values = {
        "inner_steps": 3,
        "eval_batch_size": 4,
        "eval_target_cr": 4.0,
        "eval_seq_lengths": [4096, 8192],
        "train_on_reconstructed_keys": False,
        "use_residual": False,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_live_learned_init_avoids_checkpoint_loading(monkeypatch):
    mlp = MLP(num_heads=2, head_dim=8)
    learned_init = LearnedInit.from_modules([mlp])

    def unexpected_load(*args, **kwargs):
        raise AssertionError("in-memory initialization must not call torch.load")

    monkeypatch.setattr(torch, "load", unexpected_load)
    cache = MLPValueCache(
        target_cr=4.0,
        learned_init=learned_init,
    )
    configured_cache = CompressedCache(
        config=CompressedCacheConfig(
            value=MLPValueCacheConfig(
                target_compression_ratio=4.0,
                learned_init=learned_init,
            )
        ),
        verbose=False,
    )

    assert cache.learned_init is learned_init
    assert configured_cache.value_cache.learned_init is learned_init
    name, source = next(iter(learned_init.for_layer(0).weights.items()))
    before = source.clone()
    with torch.no_grad():
        mlp.state_dict()[name].add_(1)
    torch.testing.assert_close(source, before + 1)


def test_eval_runtime_reuses_wrapper_tasks_and_managers(monkeypatch):
    wrapper_inits = []
    config_updates = []
    manager_metadata = []
    resolved_tasks = []
    prepared_tasks = []
    evaluations = []

    class Wrapper:
        def __init__(self, *, cache_config, **kwargs):
            wrapper_inits.append(cache_config)

        def set_cache_config(self, cache_config):
            config_updates.append(cache_config)

    class TaskManager:
        def __init__(self, *, metadata):
            manager_metadata.append(metadata)

    def get_tasks(tasks, *, print_tasks):
        resolved_tasks.append((tasks, print_tasks))
        return [f"resolved:{task}" for task in tasks]

    def get_task_dict(tasks, task_manager):
        prepared_tasks.append((tasks, task_manager))
        return {task: f"prepared:{task}" for task in tasks}

    def evaluate(lm, tasks, *, batch_size, task_manager, limit):
        evaluations.append((lm, tasks, batch_size, task_manager, limit))
        return {"results": {}}

    import lm_eval.tasks

    monkeypatch.setattr(lm_eval_script, "CompressedCacheHFLM", Wrapper)
    monkeypatch.setattr(lm_eval_script, "evaluate_tasks", evaluate)
    monkeypatch.setattr(lm_eval_script, "get_tasks", get_tasks)
    monkeypatch.setattr(lm_eval.tasks, "TaskManager", TaskManager)
    monkeypatch.setattr(lm_eval.tasks, "get_task_dict", get_task_dict)

    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    runtime = meta.MetaEvalRuntime(model, "model", object(), _cfg())

    assert len(wrapper_inits) == 1
    assert wrapper_inits[0].value.learned_init is None

    first = LearnedInit.from_modules([MLP(num_heads=2, head_dim=8)])
    runtime.evaluate("ruler", ["ruler_task"], 5, first)
    runtime.evaluate("ruler", ["ruler_task"], 7, first)

    # Tasks and their manager are resolved once and reused.
    assert manager_metadata == [
        {"tokenizer": "model", "max_seq_lengths": [4096, 8192]}
    ]
    assert resolved_tasks == [(["ruler_task"], False)]
    assert len(prepared_tasks) == 1
    assert evaluations[0][3] is evaluations[1][3]
    assert len(wrapper_inits) == 1

    # A second benchmark resolves its own tasks without RULER's metadata.
    runtime.evaluate("longbench", ["longbench_task"], 2, first)
    assert manager_metadata[-1] == {"tokenizer": "model"}

    # Each evaluation swaps in the current weights, no checkpoint round-trip.
    current = LearnedInit.from_modules([MLP(num_heads=2, head_dim=8)])
    runtime.evaluate("ruler", ["ruler_task"], 5, current)
    assert config_updates[-1].value.learned_init is current
    assert config_updates[-1].value.meta_weights_path is None


def test_meta_train_builds_one_runtime_for_repeated_evaluations(monkeypatch):
    runtime_inits = []
    benchmark_calls = []

    class Runtime:
        def __init__(self, *args):
            runtime_inits.append(args)

    class Model:
        def eval(self):
            return self

    cfg = _cfg(
        inner_adaptation_dtype="float32",
        inner_steps=1,
        num_meta_epochs=3,
        eval_interval=1,
        checkpoint_interval=10,
        eval_ruler=True,
        eval_ruler_samples=2,
        eval_ruler_tasks=["ruler_task"],
        eval_longbench=False,
        eval_longbench_samples=2,
        eval_longbench_tasks=None,
    )
    mlps = [MLP(num_heads=2, head_dim=8)]

    monkeypatch.setattr(meta, "MetaEvalRuntime", Runtime)
    monkeypatch.setattr(meta, "setup_optimizer", lambda *args: object())
    monkeypatch.setattr(
        meta,
        "run_epoch",
        lambda *args, **kwargs: (
            {
                "initial_support_loss": 1.0,
                "final_support_loss": 1.0,
                "meta_objective": 1.0,
            },
            1,
            kwargs["optimiser_steps"] + 1,
        ),
    )
    monkeypatch.setattr(meta, "log_epoch_metrics", lambda *args: None)
    monkeypatch.setattr(meta, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(
        meta,
        "eval_benchmark",
        lambda runtime, **kwargs: benchmark_calls.append((runtime, kwargs)),
    )

    meta.meta_train(
        model=Model(),
        name="model",
        mlps=mlps,
        dataloader=[],
        device="cpu",
        cfg=cfg,
        ckpt_path="meta_mlps.pt",
        tokenizer=object(),
    )

    assert len(runtime_inits) == 1
    assert len(benchmark_calls) == 3
    assert all(
        runtime is benchmark_calls[0][0] for runtime, _ in benchmark_calls
    )
    # Each evaluation gets the weights as they stand at that epoch.
    assert all(kwargs["learned_init"] is not None for _, kwargs in benchmark_calls)
