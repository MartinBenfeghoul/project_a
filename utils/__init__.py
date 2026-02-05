from .kv_cache import generate_kv_batched, generate_outputs_single_pass
from .metrics import eval_model, avg_nll, clean
from .model import get_model_and_tokenizer
from .matrix_decomposition import truncated_svd, full_svd, learn_lora_matrix
from .dataloader import PackedTokens, load_data, collate
from .cache import DynamicCache, SVDCache, SurpriseSVDCache
from .args import str2bool, list_of_strings, list_of_floats, args_type


CACHE_CLASSES = {
    'baseline': DynamicCache,
    'svd': SVDCache,
    'surprise_svd': SurpriseSVDCache,
}