{ pkgs }:
{
  qwen3-asr = {
    names = [ "qwen3-asr" ];
    pythonPackages = [ ];
    wrapperArgs = runtime: [
      "--set" "HYPRWHSPR_QWEN3_ASR_RUNTIME" "${runtime}"
      "--set" "HYPRWHSPR_QWEN3_ASR_DEVICES" "cpu,vulkan"
    ];
    runtime = pkgs.callPackage ./qwen3-asr-runtime.nix { };
  };
}
