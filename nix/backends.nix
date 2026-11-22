{ pkgs, llama-cpp ? null, llama-cpp-vulkan ? null }:
{
  qwen3-asr = {
    names = [ "qwen3-asr" ];
    pythonPackages = [ ];
    wrapperArgs = runtime: [
      "--set" "HYPRWHSPR_QWEN3_ASR_RUNTIME" "${runtime}"
      "--set" "HYPRWHSPR_QWEN3_ASR_DEVICES" "cpu,vulkan"
    ];
    runtime =
      if llama-cpp == null || llama-cpp-vulkan == null
      then throw "qwen3-asr runtime requires nixpkgs unstable llama-cpp and llama-cpp-vulkan"
      else pkgs.callPackage ./qwen3-asr-runtime.nix { inherit llama-cpp llama-cpp-vulkan; };
  };
}
