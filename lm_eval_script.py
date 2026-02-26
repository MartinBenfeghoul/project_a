import os
import json
import argparse
import time
import torch

from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

from utils import CompressedCache, Logger, get_model_and_tokenizer

# TODO: not sure if this is needed, isn't it defined in the task config yaml?
# Also, if needed, these will change for ruler and longbench, need to update
GEN_KWARGS = {
    "do_sample": False,
    "use_cache": True,
    "max_new_tokens": 512,
}


def get_device(model):
    try:
        device = model.device
    except AttributeError:
        try:
            device = model.model.device
        except AttributeError:
            device = "cuda:0"
    return device


def get_output_path(output_path):
    for i in range(100):
        if not os.path.exists(output_path.format(i)):
            return output_path.format(i)


def get_device_type():
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} CUDA chips")
        return "cuda"
    print("WARNING: CUDA not available. Continuing with CPU.")
    return "cpu"


def list_of_strings(arg):
    return arg.split(",")

def make_hook(logger):                                                                                                                                                                                                                                                                                                                            
    """Forward hook matching the one in niah.py."""                                                                                                                                                                                                                                                                                               
    def hook(module, args, kwargs, output):                                                                                                                                                                                                                                                                                                       
        input_ids = kwargs["input_ids"]                                                                                                                                                                                                                                                                                                           
        seq_len = input_ids.size(-1)                                                                                                                                                                                                                                                                                                              
        pkv = output.past_key_values                                                                                                                                                                                                                                                                                                              
        if seq_len > 1:                                                                                                                                                                                                                                                                                                                           
            nll = output.loss                                                                                                                                                                                                                                                                                                                     
            logits = output.logits                                                                                                                                                                                                                                                                                                                
            if nll is None:                                                                                                                                                                                                                                                                                                                       
                nll = module.loss_function(                                                                                                                                                                                                                                                                                                       
                    logits=logits,                                                                                                                                                                                                                                                                                                                
                    labels=input_ids,                                                                                                                                                                                                                                                                                                             
                    vocab_size=module.config.vocab_size,                                                                                                                                                                                                                                                                                          
                    **kwargs,                                                                                                                                                                                                                                                                                                                     
                )                                                                                                                                                                                                                                                                                                                                 
            ppl = torch.exp(nll)                                                                                                                                                                                                                                                                                                                  
            print(f"Prefill: nll={nll.item():.1f}, ppl={ppl.item():.1f}, seq_len={seq_len}")                                                                                                                                                                                                                                                      
            if hasattr(pkv, "update_events"):                                                                                                                                                                                                                                                                                                     
                pkv.update_events(logits[..., :-1, :], input_ids[..., 1:])                                                                                                                                                                                                                                                                        
        else:
            if hasattr(pkv, "comp_ratio") and not logger.recorded_cr:                                                                                                                                                                                                                                                                             
                cr = pkv.comp_ratio                                                                                                                                                                                                                                                                                                               
                if cr is not None:                                                                                                                                                                                                                                                                                                                
                    print(f"Compression ratio: {cr:.2f}")                                                                                                                                                                                                                                                                                         
                    logger.add_log("crs", cr)                                                                                                                                                                                                                                                                                                     
                    logger.recorded_cr = True                                                                                                                                                                                                                                                                                                     
    return hook                                      


class CompressedCacheHFLM(HFLM):
    """HFLM subclass that injects a fresh CompressedCache into every model call."""

    def __init__(self, key_cache_kwargs, value_cache_kwargs, logger, **kwargs):
        super().__init__(**kwargs)
        self._key_cache_kwargs = key_cache_kwargs
        self._value_cache_kwargs = value_cache_kwargs
        self._logger = logger

    def _make_cache(self):
        return CompressedCache(
            config=self.model.config,
            key_cache_kwargs=self._key_cache_kwargs,
            value_cache_kwargs=self._value_cache_kwargs,
        )

    def _model_call(self, inps, attn_mask=None, labels=None):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            if attn_mask is not None or labels is not None:
                # seq2seq path — pass through unchanged
                assert attn_mask is not None and labels is not None
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            return self.model(inps, past_key_values=self._make_cache()).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        self._logger.recorded_cr = False
        generation_kwargs["past_key_values"] = self._make_cache()
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


@torch.no_grad()
def main(args):
    device_type = get_device_type()
    device = torch.device(device_type)

    model, tokenizer = get_model_and_tokenizer(args.model_name, device)
    logger = Logger()
    # model.register_forward_hook(make_hook(logger), with_kwargs=True) # TODO: is this needed?

    key_cache_kwargs = {
        "cache_type": args.cache_type,
        "decomposition_method": args.decomposition_method,
        "comp_ratio": args.comp_ratio,
        "energy_threshold": args.energy_threshold,
        "rank_selection": args.rank_selection,
        "lr": args.lr,
        "n_iter": args.n_iter,
        "gamma": 3.0,
        "min_size": 8.0,
    }
    value_cache_kwargs = {"cache_type": args.v_cache_type}

    model.eval()
    lm = CompressedCacheHFLM(
        key_cache_kwargs=key_cache_kwargs,
        value_cache_kwargs=value_cache_kwargs,
        logger=logger,
        pretrained=model,
        tokenizer=tokenizer,
        truncation=False,
        trust_remote_code=True,
    )

    print(f"Evaluating tasks: {args.tasks}")
    tm = TaskManager(metadata={"tokenizer": args.model_name})

    if args.log_efficiency_metrics:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

    results = evaluator.simple_evaluate(
        model=lm,
        gen_kwargs=GEN_KWARGS,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=1,
        max_batch_size=1,
        device=get_device(lm),
        task_manager=tm,
        limit=args.limit,
    )

    if args.log_efficiency_metrics:
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        wall_time_sec = time.perf_counter() - start_time
        efficiency_metrics = {
            "eval_wall_time_seconds": wall_time_sec,
            "eval_wall_time_minutes": wall_time_sec / 60.0,
            "gpu_peak_mem_bytes": peak_mem_bytes,
            "gpu_peak_mem_gib": peak_mem_bytes / (1024 ** 3),
        }
        print("Efficiency metrics:", efficiency_metrics)
        results["results"]["efficiency_metrics"] = efficiency_metrics

    print(make_table(results))

    if args.debug:
        print("Debug mode — not saving results.")
        return results

    output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lm_eval_{args.cache_type}" + "_{}.json")
    output_path = get_output_path(output_path)
    with open(output_path, "w") as f:
        json.dump(results["results"], f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="LM eval harness script")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("-o", "--output_dir", type=str, default="./results")
    parser.add_argument("-t", "--tasks", type=list_of_strings, default="lm_eval")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples per task.")
    parser.add_argument("--log_efficiency_metrics", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # key cache
    parser.add_argument("-c", "--cache_type", type=str, default="surprise_lr")
    parser.add_argument("--decomposition_method", type=str, default="svd", choices=["svd", "lora"])
    parser.add_argument("-r", "--comp_ratio", type=float, default=2.0)
    parser.add_argument("-e", "--energy_threshold", type=float, default=0.95)
    parser.add_argument("--rank_selection", type=str, default="comp_ratio", choices=["comp_ratio", "energy"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_iter", type=int, default=3)

    # value cache
    parser.add_argument("--v_cache_type", type=str, default="baseline")

    # meta-learning

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
