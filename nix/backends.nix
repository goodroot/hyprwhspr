# Optional backends. One attribute each, keyed by transcription_backend:
#   description  appended to the package description
#   runtime      store path for the backend's sidecar, or null
#   python       extra Python packages, as a function of the app's package set
#   wrapperArgs  extra makeWrapper arguments for hyprwhspr and hyprwhspr-tray
{ pkgs, llamaPkgs }:
{
  qwen3-asr = rec {
    description = "a Qwen3-ASR sidecar";
    # nixos-26.05 ships llama.cpp older than the Qwen3-ASR support in b10516.
    runtime = pkgs.callPackage ./qwen3-asr-runtime.nix {
      inherit (llamaPkgs) llama-cpp llama-cpp-vulkan;
    };
    python = ps: [ ];
    # vulkaninfo lets qwen3_asr_device = auto pick Vulkan only with a real GPU.
    wrapperArgs = [
      "--set" "HYPRWHSPR_QWEN3_ASR_RUNTIME" "${runtime}"
      "--prefix" "PATH" ":" "${pkgs.vulkan-tools}/bin"
    ];
  };
}
