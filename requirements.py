from typing import Any
import Utilities.install_requirements as req
import Utilities.gpu_utils as gpu

REQ_VERSION = 1

def Install(GPU: gpu.GPUType, Vulkan: bool, Env: dict[str, Any], Args: list[str]) -> None:
    f16 = int(Env.get("CHATBOT_F16", True))
    blasBuild = int(Env.get("CHATBOT_BLAS", False))
    blasVendor = Env.get("CHATBOT_BLAS_VENDOR", "OpenBLAS")
    openVINOBuild = int(Env.get("CHATBOT_OPENVINO", False))
    nativeCPU = int(Env.get("CHATBOT_NATIVE", False))

    lcppCmake = (
        "-DCMAKE_BUILD_TYPE=Release "
        f"-DGGML_NATIVE={nativeCPU} "
        f"-DGGML_OPENBLAS={blasBuild} -DGGML_BLAS_VENDOR={blasVendor} "
        f"-DGGML_OPENVINO={openVINOBuild} "
        f"-DGGML_CUDA={int(GPU == gpu.GPUType.NVIDIA)} -DGGML_CUDA_F16={f16} "
        f"-DGGML_HIPBLAS={int(GPU == gpu.GPUType.AMD)} -DGGML_HIPBLAS_F16={f16} "
        f"-DGGML_SYCL={int(GPU == gpu.GPUType.INTEL)} -DGGML_SYCL_F16={f16} "
        f"-DGGML_VULKAN={int(GPU == gpu.GPUType.NO_GPU and Vulkan)} "
    )

    if (GPU == gpu.GPUType.INTEL):
        lcppCmake += f"-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx "
    
    req.InstallPackage(
        Packages = ["git+https://github.com/TAO71-AI/llama-cpp-python-JamePeng.git@main"],
        EnvVars = {
            "CMAKE_ARGS": lcppCmake.strip()
        },
        PIPOptions = Args
    )

if (__name__ == "__main__"):
    Install()