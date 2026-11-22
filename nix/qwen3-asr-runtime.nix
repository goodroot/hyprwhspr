{ stdenvNoCC, llama-cpp, llama-cpp-vulkan }:
stdenvNoCC.mkDerivation {
  pname = "hyprwhspr-qwen3-asr-runtime";
  version = llama-cpp.version;
  dontUnpack = true;
  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    ln -s ${llama-cpp} "$out/cpu"
    ln -s ${llama-cpp-vulkan} "$out/vulkan"
    runHook postInstall
  '';
}
