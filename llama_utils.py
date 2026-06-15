# Import libraries
import logging
from llama_cpp import (
    # Model
    Llama,

    # Cache types
    LlamaDiskCache,
    LlamaRAMCache,

    # Split modes
    llama_split_mode,

    # ROPE scaling types
    llama_rope_scaling_type,

    # Attention types
    llama_attention_type,
    llama_flash_attn_type,

    # Context types
    llama_context_type,

    # Pooling types
    LLAMA_POOLING_TYPE_CLS as POOLING_CLS,
    LLAMA_POOLING_TYPE_MEAN as POOLING_MEAN,
    LLAMA_POOLING_TYPE_LAST as POOLING_LAST,
    LLAMA_POOLING_TYPE_NONE as POOLING_NONE,
    LLAMA_POOLING_TYPE_RANK as POOLING_RANK,
    LLAMA_POOLING_TYPE_UNSPECIFIED as POOLING_UNSPECIFIED,

    # Ftypes
    llama_ftype,

    # Other
    llama_get_memory,
    llama_memory_seq_rm
)
from typing import Any
import time

__FTYPES__: dict[str | tuple[str, ...], int] = {
    ("f32", "fp32"): llama_ftype.LLAMA_FTYPE_ALL_F32,
    "bf16": llama_ftype.LLAMA_FTYPE_MOSTLY_BF16,
    ("f16", "fp16"): llama_ftype.LLAMA_FTYPE_MOSTLY_F16,
    "q8_0": llama_ftype.LLAMA_FTYPE_MOSTLY_Q8_0,
    "q6_k": llama_ftype.LLAMA_FTYPE_MOSTLY_Q6_K,
    ("q5_k_m", "q5_k"): llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_K_M,
    "q5_k_s": llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_K_S,
    ("q4_k_m", "q4_k"): llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_K_M,
    "q4_k_s": llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_K_S,
    "q3_k_l": llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_L,
    ("q3_k_m", "q3_k"): llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_M,
    "q3_k_s": llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_S,
    "q2_k": llama_ftype.LLAMA_FTYPE_MOSTLY_Q2_K,
    "q2_k_s": llama_ftype.LLAMA_FTYPE_MOSTLY_Q2_K_S,

    "q5_1": llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_1,
    "q5_0": llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_0,
    "q4_1": llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_1,
    "q4_0": llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_0,
    "q1_0": llama_ftype.LLAMA_FTYPE_MOSTLY_Q1_0,

    "tq1_0": llama_ftype.LLAMA_FTYPE_MOSTLY_TQ1_0,
    "tq2_0": llama_ftype.LLAMA_FTYPE_MOSTLY_TQ2_0,

    "iq1_s": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ1_S,
    "iq1_m": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ1_M,
    "iq2_xxs": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_XXS,
    "iq2_xs": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_XS,
    "iq2_s": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_S,
    "iq2_m": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_M,
    "iq3_xxs": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_XXS,
    "iq3_xs": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_XS,
    "iq3_s": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_S,
    "iq3_m": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_M,
    "iq4_xs": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_XS,
    "iq4_nl": llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_NL,

    ("mxfp4", "mxfp4_moe", "mxfp4moe"): llama_ftype.LLAMA_FTYPE_MOSTLY_MXFP4_MOE,
    "nvfp4": llama_ftype.LLAMA_FTYPE_MOSTLY_NVFP4
}
__SPLIT_MODES__: dict[str | tuple[str, ...], int] = {
    "layer": llama_split_mode.LLAMA_SPLIT_MODE_LAYER,
    "row": llama_split_mode.LLAMA_SPLIT_MODE_ROW,
    "tensor": llama_split_mode.LLAMA_SPLIT_MODE_TENSOR,
    "none": llama_split_mode.LLAMA_SPLIT_MODE_NONE
}
__ROPE_SCALING_TYPES__: dict[str | tuple[str, ...], int] = {
    "linear": llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_LINEAR,
    "longrope": llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    ("max_value", "max-value", "max value"): llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_MAX_VALUE,
    "none": llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_NONE,
    "unspecified": llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
    "yarn": llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN
}
__POOLING_TYPES__: dict[str | tuple[str, ...], int] = {
    "cls": POOLING_CLS,
    "mean": POOLING_MEAN,
    "last": POOLING_LAST,
    "none": POOLING_NONE,
    "rank": POOLING_RANK,
    "unspecified": POOLING_UNSPECIFIED
}
__ATTN_TYPES__: dict[str | tuple[str, ...], int] = {
    "casual": llama_attention_type.LLAMA_ATTENTION_TYPE_CAUSAL,
    ("non_casual", "non-casual", "non casual"): llama_attention_type.LLAMA_ATTENTION_TYPE_NON_CAUSAL,
    "unspecified": llama_attention_type.LLAMA_ATTENTION_TYPE_UNSPECIFIED
}
__FLASH_ATTN_TYPES__: dict[str | tuple[str, ...], int] = {
    ("auto", "automatic"): llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO,
    "enabled": llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED,
    "disabled": llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED
}
__CTX_TYPES__: dict[str, tuple[str, ...], int] = {
    "default": llama_context_type.LLAMA_CONTEXT_TYPE_DEFAULT,
    "mtp": llama_context_type.LLAMA_CONTEXT_TYPE_MTP
}

def ClearLlamaCache(Model: Llama) -> None:
    kv = llama_get_memory(Model.ctx)
    llama_memory_seq_rm(kv, -1, -1, -1)

def __get_value_from_dictionary__(Key: Any, Dictionary: dict[Any | list[Any] | tuple[Any, ...], Any], Default: Any | None = None) -> tuple[Any, int] | (Any | None):
    """
    Retrieves the value associated with a key in a dictionary, where the keys can be either a single object or a list of possible keys.

    Args:
        Key (Any): The key to search for.
        Dictionary (dict[Any | list[Any], Any]): The dictionary to search in.
        Default (Any | None): The default value to return if the key is not found.
    
    Returns:
        tuple[Any, int] | (Any | None)
    """
    # For each item in the dictionary
    for key, value in Dictionary.items():
        # Check the key type
        if (isinstance(key, list)):
            # Check if the key is found
            if (Key in key):
                return (value, list(Dictionary.keys()).index(key))
        elif (isinstance(key, tuple)):
            k = list(key)

            if (Key in k):
                return (value, list(Dictionary.keys()).index(key))
        elif (Key == key):
            return (value, list(Dictionary.keys()).index(key))
    
    # Key not found, return the default value
    return Default

def StringToFtype(Ftype: str | None) -> int | None:
    return __get_value_from_dictionary__(Ftype, __FTYPES__, None)[0]

def StringToSplitMode(SplitMode: str | None) -> int | None:
    return __get_value_from_dictionary__(SplitMode, __SPLIT_MODES__, llama_split_mode.LLAMA_SPLIT_MODE_LAYER)[0]

def StringToRopeScalingType(RopeScalingType: str | None) -> int | None:
    return __get_value_from_dictionary__(RopeScalingType, __ROPE_SCALING_TYPES__, llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED)[0]

def StringToPoolingType(PoolingType: str | None) -> int | None:
    return __get_value_from_dictionary__(PoolingType, __POOLING_TYPES__, POOLING_UNSPECIFIED)[0]

def StringToCacheType(CacheType: str | None, CapacityInBytes: int = 2 ^ 30) -> LlamaDiskCache | LlamaRAMCache | None:
    """
    Converts a string (cache type name) into a class.

    Args:
        CacheType (str): The cache type name.
        CapacityInBytes (int): The capacity of the cache.
    
    Returns:
        LlamaDiskCache | LlamaRAMCache | None
    """
    # Get and return the cache type
    if (CacheType == "disk"):
        return LlamaDiskCache(capacity_bytes = CapacityInBytes)
    elif (CacheType == "ram"):
        return LlamaRAMCache(capacity_bytes = CapacityInBytes)
    
    return None

def StringToAttnType(AttnType: str | None) -> int:
    return __get_value_from_dictionary__(AttnType, __ATTN_TYPES__, llama_attention_type.LLAMA_ATTENTION_TYPE_UNSPECIFIED)[0]

def StringToFlashAttnType(FlashAttnType: str | None) -> int:
    return __get_value_from_dictionary__(FlashAttnType, __FLASH_ATTN_TYPES__, llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO)[0]

def StringToCtxType(CtxType: str | None) -> int:
    return __get_value_from_dictionary__(CtxType, __CTX_TYPES__, llama_context_type.LLAMA_CONTEXT_TYPE_DEFAULT)[0]

def LoadLlamaModel(Configuration: dict[str, Any]) -> dict[str, Llama | Any]:
    """
    Loads a llama.cpp model.

    Args:
        Configuration (dict[str, Any]): Configuration for the model.
    
    Returns:
        dict[str, Llama | Any]
    """
    # Get verbose parameters
    verbose = Configuration.get("_private_verbose", False)
    mmprojVerbose = Configuration.get("_private_mmproj_verbose", False)

    # Get model path (and mmproj if provided)
    modelPathConfig = Configuration.get("_private_model_path", {})
    
    if (isinstance(modelPathConfig, dict)):
        modelPath = modelPathConfig.get("llm", None)
        mmproj = modelPathConfig.get("mmproj", None)
    else:
        modelPath = str(modelPathConfig)
        mmproj = None
    
    # Get mmproj GPU usage
    mmprojGPU = Configuration.get("_private_mmproj_use_gpu", True)

    # Get mmproj min and max tokens
    mmprojMinImageTokens = Configuration.get("mmproj_min_image_tokens", -1)
    mmprojMaxImageTokens = Configuration.get("mmproj_max_image_tokens", -1)
    
    # Get LoRA
    loraConfig = Configuration.get("_private_lora", {})
    loraPath = loraConfig.get("path", None)
    loraBase = loraConfig.get("base_path", None)
    loraScale = loraConfig.get("scale", 1)
    
    # Get GPU layers
    gpuLayers = Configuration.get("_private_gpu_layers", -1)
    
    # Get split mode
    splitMode = Configuration.get("_private_split_mode", None)
    splitMode = StringToSplitMode(splitMode)
    
    # Get main GPU
    mainGPU = Configuration.get("_private_main_gpu", 0)
    
    # Get mmap
    mmap = Configuration.get("_private_use_mmap", True)
    
    # Get mlock
    mlock = Configuration.get("_private_use_mlock", False)
    
    # Get ctx
    ctx = Configuration.get("ctx", 2048)

    # Get ctx checkpoints
    ctxCheckpoints = Configuration.get("_private_ctx_checkpoints", 16)
    ctxCheckpointsInterval = Configuration.get("_private_ctx_checkpoints_interval", 4096)
    ctxCheckpointsOnDevice = Configuration.get("_private_ctx_checkpoints_on_device", False)

    # Get ctx tyoe
    ctxType = Configuration.get("_private_ctx_type", None)
    ctxType = StringToCtxType(ctxType)
    
    # Get batch
    batch = Configuration.get("_private_batch", 512)
    
    # Get ubatch
    ubatch = Configuration.get("_private_ubatch", 512)
    
    # Get threads
    threads = Configuration.get("_private_threads", None)
    
    # Get batch threads
    batchThreads = Configuration.get("_private_batch_threads", None)
    
    # Get rope scaling type
    ropeScalingType = Configuration.get("_private_rope_scaling_type", None)
    ropeScalingType = StringToRopeScalingType(ropeScalingType)
    
    # Get rope freq base
    ropeFreqBase = Configuration.get("_private_rope_freq_base", 0)
    
    # Get rope freq scale
    ropeFreqScale = Configuration.get("_private_rope_freq_scale", 0)
    
    # Get yarn ext factor
    yarnExtFactor = Configuration.get("_private_yarn_ext_factor", -1)
    
    # Get yarn attn factor
    yarnAttnFactor = Configuration.get("_private_yarn_attn_factor", 1)
    
    # Get yarn beta fast
    yarnBetaFast = Configuration.get("_private_yarn_beta_fast", 32)
    
    # Get yarn beta slow
    yarnBetaSlow = Configuration.get("_private_yarn_beta_slow", 1)
    
    # Get yarn orig ctx
    yarnOrigCtx = Configuration.get("_private_yarn_orig_ctx", 0)
    
    # Get pooling type
    poolingType = Configuration.get("_private_pooling_type")
    poolingType = StringToPoolingType(poolingType)
    
    # Get offload KQV
    offloadKqv = Configuration.get("_private_offload_kqv", True)
    
    # Get offload op
    offloadOp = Configuration.get("_private_offload_op", None)

    # Get attn type
    attn = Configuration.get("_private_attn", None)
    attn = StringToAttnType(attn)
    
    # Get flash attn type
    flashAttn = Configuration.get("_private_flash_attn", None)
    flashAttn = StringToFlashAttnType(flashAttn)
    
    # Get swa full
    swaFull = Configuration.get("_private_swa_full", None)
    
    # Get FType K
    ftypeK = Configuration.get("ftype_k", None)
    ftypeK = StringToFtype(ftypeK)

    if (ftypeK is None or ftypeK not in [
        llama_ftype.LLAMA_FTYPE_ALL_F32, llama_ftype.LLAMA_FTYPE_MOSTLY_F16, llama_ftype.LLAMA_FTYPE_MOSTLY_BF16,
        llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_0, llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_1,
        llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_0, llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_1,
        llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_NL
    ]):
        ftypeK = None
        logging.warning("[llama_utils] `ftype_k` not found or invalid. Set to None.")
    
    # Get FType V
    ftypeV = Configuration.get("ftype_v", None)
    ftypeV = StringToFtype(ftypeV)

    if (ftypeV is None or ftypeV not in [
        llama_ftype.LLAMA_FTYPE_ALL_F32, llama_ftype.LLAMA_FTYPE_MOSTLY_F16, llama_ftype.LLAMA_FTYPE_MOSTLY_BF16,
        llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_0, llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_1,
        llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_0, llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_1,
        llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_NL
    ]):
        ftypeV = None
        logging.warning("[llama_utils] `ftype_v` not found or invalid. Set to None.")
    
    # Get spm infill
    spmInfill = Configuration.get("_private_spm_infill", False)
    
    # Get cache type
    cacheType = Configuration.get("_private_cache_type", None)
    cacheType = StringToCacheType(cacheType)
    
    # Get multimodal type
    multimodal = Configuration.get("multimodal", "text")

    if (isinstance(multimodal, str)):
        multimodal = multimodal.split(" ")
    elif (not isinstance(multimodal, list)):
        multimodal = ["text"]

    for mul in multimodal:
        if (
            mul != "text" and
            mul != "image" and
            mul != "video" and
            mul != "audio"
        ):
            logging.warning(f"[llama_utils] Multimodal type '{mul}' not supported.")
            continue
    
    # Get cpu moe
    cpuMoE = Configuration.get("_private_cpu_moe", False)
    nCPUMoE = Configuration.get("_private_n_cpu_moe", 0)
    
    # Get keep
    nKeep = Configuration.get("_private_n_keep", 256)

    # Get KV unified
    kvUnified = Configuration.get("_private_kv_unified", None)

    # Get check tensors
    checkTensors = Configuration.get("_private_check_tensors", False)

    # Get extra bufts usage
    useExtraBufts = Configuration.get("_private_use_extra_bufts", False)

    # Get direct IO usage
    useDirectIO = Configuration.get("_private_use_direct_io", False)

    # Get numa usage
    useNuma = Configuration.get("_private_numa", False)
    
    # Save the parameters in a dictionary
    modelParamsLCPP = {
        "model_path": modelPath,
        "mmproj_path": mmproj,
        "chat_handler_kwargs": {
            "use_gpu": mmprojGPU,
            "image_min_tokens": mmprojMinImageTokens,
            "image_max_tokens": mmprojMaxImageTokens,
            "verbose": mmprojVerbose
        },
        "n_gpu_layers": gpuLayers,
        "cpu_moe": cpuMoE,
        "n_cpu_moe": nCPUMoE,
        "split_mode": splitMode,
        "main_gpu": mainGPU,
        "vocab_only": False,
        "use_mmap": mmap,
        "use_mlock": mlock,
        "seed": -1,
        "n_ctx": ctx,
        "n_batch": batch,
        "n_ubatch": ubatch,
        "n_threads": threads,
        "n_threads_batch": batchThreads,
        "rope_scaling_type": ropeScalingType,
        "rope_freq_base": ropeFreqBase,
        "rope_freq_scale": ropeFreqScale,
        "yarn_ext_factor": yarnExtFactor,
        "yarn_attn_factor": yarnAttnFactor,
        "yarn_beta_fast": yarnBetaFast,
        "yarn_beta_slow": yarnBetaSlow,
        "yarn_orig_ctx": yarnOrigCtx,
        "pooling_type": poolingType,
        "logits_all": False,
        "embeddings": False,
        "offload_kqv": offloadKqv,
        "op_offload": offloadOp,
        "swa_full": swaFull,
        "no_perf": False,
        "type_k": ftypeK,
        "type_v": ftypeV,
        "spm_infill": spmInfill,
        "cache_type": cacheType,
        "lora_path": loraPath,
        "lora_base": loraBase,
        "lora_scale": loraScale,
        "verbose": verbose,
        "use_direct_io": useDirectIO,
        "check_tensors": checkTensors,
        "use_extra_bufts": useExtraBufts,
        "n_keep": nKeep,
        "ctx_type": ctxType,
        "attention_type": attn,
        "flash_attn_type": flashAttn,
        "kv_unified": kvUnified,
        "ctx_checkpoints": ctxCheckpoints,
        "checkpoint_interval": ctxCheckpointsInterval,
        "checkpoint_on_device": ctxCheckpointsOnDevice,
        "numa": useNuma,
        "draft_model": None  # TODO: Support speculative decoding
    }

    # Load the model
    logging.info("[llama_utils] Loading model...")
    loadingTime = time.time()

    model = Llama(**modelParamsLCPP)
    model.set_cache(cacheType)

    loadingTime = time.time() - loadingTime
    loadingTime = round(loadingTime, 3)

    logging.info(f"[llama_utils] Model loaded in {loadingTime} seconds.")
    return {
        "_private_model": model,
        "_private_type": "lcpp"
    }